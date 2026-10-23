"""Lossless relayer for *single flat layer* Docker-save images (stdlib only).

The source archive stays untouched. No pull/push, extraction, or production changes.
Usage: python tools/shared_base.py originals.tar --output results/shared_base_trial
The output is an OCI archive loadable with `docker load -i images.oci.tar`.
All file metadata (including timestamps/PAX xattrs) participates in sharing.
Hardlinks AND their referents stay in the task layer. Unsupported layers fail closed.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import posixpath
import shutil
import sqlite3
import tarfile
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE images (id INTEGER PRIMARY KEY, tag TEXT UNIQUE, config TEXT, layer TEXT,
 compressed_bytes INTEGER, raw_bytes INTEGER, diff_id TEXT);
CREATE TABLE entries (image INTEGER, path TEXT, signature TEXT, kind TEXT, size INTEGER,
 link TEXT, PRIMARY KEY(image,path));
CREATE INDEX entries_sig ON entries(path,signature);
CREATE TABLE shared (path TEXT PRIMARY KEY);
"""


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(name):
    if name.startswith("/") or "\x00" in name or "\\" in name:
        raise ValueError(f"unsafe tar path: {name!r}")
    parts = name.split("/")
    if ".." in parts:
        raise ValueError(f"unsafe tar path: {name!r}")
    return posixpath.normpath(name)


def metadata(member):
    # Keep exact effective metadata, not incidental tar header format/offsets.
    result = {key: getattr(member, key) for key in (
        "mode", "uid", "gid", "size", "mtime", "linkname", "uname", "gname",
        "devmajor", "devminor"
    )} | {"type": member.type.hex()}
    # GNU longname vs PAX path/linkpath is an encoding choice, not a filesystem
    # attribute. TarInfo already resolves both into name/linkname. Keep all
    # other PAX keys, including xattrs and sub-second timestamps, byte-for-byte.
    result["pax_headers"] = {k: v for k, v in member.pax_headers.items()
                             if k not in {"path", "linkpath"}}
    return result


def fingerprint(member, stream=None):
    if member.sparse is not None or any(k.startswith("GNU.sparse") for k in member.pax_headers):
        raise ValueError(f"sparse entry unsupported: {member.name}")
    if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
        raise ValueError(f"special entry unsupported: {member.name}")
    content = hashlib.file_digest(stream, "sha256").hexdigest() if member.isfile() else ""
    return hashlib.sha256(json_bytes([metadata(member), content])).hexdigest()


def index_layer(db, image, layer):
    rows = []
    with tarfile.open(layer, "r:") as source:
        for member in source:
            name = canonical(member.name)
            if any(part.startswith(".wh.") for part in name.split("/")):
                raise ValueError("source must be flat: whiteout found")
            if member.islnk():
                canonical(member.linkname)
            signature = fingerprint(member, source.extractfile(member) if member.isfile() else None)
            kind = "file" if member.isfile() else "dir" if member.isdir() else "hardlink" if member.islnk() else "symlink"
            rows.append((image, name, signature, kind, member.size, canonical(member.linkname) if member.islnk() else member.linkname))
            if len(rows) >= 1000:
                db.executemany("INSERT INTO entries VALUES (?,?,?,?,?,?)", rows)
                rows.clear()
    db.executemany("INSERT INTO entries VALUES (?,?,?,?,?,?)", rows)
    db.commit()


def select_shared(db, count):
    # Re-emit every directory in each delta so each task's directory metadata wins.
    # No hardlinked inode is split across layers (including regular referents).
    db.execute("""INSERT INTO shared
      SELECT path FROM entries GROUP BY path
      HAVING COUNT(*)=? AND COUNT(DISTINCT signature)=1
      AND MIN(kind) IN ('file','symlink')
      AND path NOT IN (SELECT link FROM entries WHERE kind='hardlink')""", (count,))
    # A common file is shareable only when every explicit ancestor directory
    # also has identical metadata. This avoids implicit-parent metadata and
    # prevents one image from deciding another image's base directory metadata.
    shared = {r[0] for r in db.execute("SELECT path FROM shared")}
    compatible = {}
    file_parents = {}
    for path in shared:
        ancestors = set()
        parent = posixpath.dirname(path)
        while parent and parent != ".":
            ancestors.add(parent)
            if parent not in compatible:
                rows = db.execute("SELECT kind,signature FROM entries WHERE path=?", (parent,)).fetchall()
                compatible[parent] = (len(rows) == count and all(row[0] == "dir" for row in rows)
                                      and len({row[1] for row in rows}) == 1)
            parent = posixpath.dirname(parent)
        file_parents[path] = ancestors
    excluded = [path for path, ancestors in file_parents.items()
                if not all(compatible[parent] for parent in ancestors)]
    for path in excluded:
        db.execute("DELETE FROM shared WHERE path=?", (path,))
        shared.remove(path)
    parents = set().union(*(file_parents[path] for path in shared)) if shared else set()
    db.commit()
    return shared, parents


def write_layer(source_path, dest, selected):
    with tarfile.open(source_path, "r:") as source, tarfile.open(dest, "w", format=tarfile.PAX_FORMAT) as target:
        for member in source:
            if selected(member):
                target.addfile(copy.copy(member), source.extractfile(member) if member.isfile() else None)


def blob(layout, raw=None, path=None, compress=False):
    directory = layout / "blobs" / "sha256"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "pending"
    if compress:
        with Path(path).open("rb") as source, temporary.open("wb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=6) as zipped:
                shutil.copyfileobj(source, zipped, 1024 * 1024)
    else:
        temporary.write_bytes(raw)
    digest = sha_file(temporary)
    result = {"digest": "sha256:" + digest, "size": temporary.stat().st_size}
    final = directory / digest
    if final.exists():
        temporary.unlink()
    else:
        temporary.rename(final)
    return result


def verify_composition(db, image, base, delta):
    # Disk-backed proof over all effective metadata/content and the complete path set.
    db.execute("DROP TABLE IF EXISTS actual")
    db.execute("CREATE TEMP TABLE actual(path TEXT PRIMARY KEY, signature TEXT)")
    for layer in (base, delta):
        with tarfile.open(layer, "r:") as source:
            batch = []
            for member in source:
                signature = fingerprint(member, source.extractfile(member) if member.isfile() else None)
                batch.append((canonical(member.name), signature))
                if len(batch) >= 1000:
                    db.executemany("INSERT OR REPLACE INTO actual VALUES (?,?)", batch)
                    batch.clear()
            db.executemany("INSERT OR REPLACE INTO actual VALUES (?,?)", batch)
    differences = db.execute("""SELECT COUNT(*) FROM (
      SELECT path,signature FROM entries WHERE image=? EXCEPT SELECT * FROM actual
      )""", (image,)).fetchone()[0]
    extra = db.execute("""SELECT COUNT(*) FROM (
      SELECT * FROM actual EXCEPT SELECT path,signature FROM entries WHERE image=?
      )""", (image,)).fetchone()[0]
    if differences or extra:
        raise ValueError(f"composition mismatch image={image}: missing/changed={differences} extra={extra}")
    return {"entries": db.execute("SELECT COUNT(*) FROM actual").fetchone()[0], "missing_or_changed": differences, "extra": extra}


def canonical_image_name(name):
    first = name.split("/", 1)[0]
    if "/" not in name:
        return "docker.io/library/" + name
    if "." not in first and ":" not in first and first != "localhost":
        return "docker.io/" + name
    return name


def pack_oci(layout, archive):
    # Envelope metadata must not make otherwise identical OCI artifacts drift.
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as packed:
        for path in sorted(layout.rglob("*")):
            if path.is_file():
                member = tarfile.TarInfo(path.relative_to(layout).as_posix())
                member.size = path.stat().st_size
                member.mode = 0o644
                member.mtime = 0
                with path.open("rb") as stream:
                    packed.addfile(member, stream)


def read_manifest(source, repository):
    manifest = json.load(source.extractfile("manifest.json"))
    if len(manifest) < 2:
        raise ValueError("at least two source images required")
    tails = [entry["RepoTags"][0].rsplit(":", 1)[-1]
             for entry in manifest if len(entry["RepoTags"]) == 1]
    targets = set()
    for entry in manifest:
        if len(entry["Layers"]) != 1 or len(entry["RepoTags"]) != 1:
            raise ValueError("only single-layer, single-tag input images supported")
        config = json.load(source.extractfile(entry["Config"]))
        if len(config["rootfs"]["diff_ids"]) != 1:
            raise ValueError("invalid source diff_ids")
        original_tag = entry["RepoTags"][0]
        tail = original_tag.rsplit(":", 1)[1]
        if tails.count(tail) > 1:
            tail = tail[:100] + "-" + hashlib.sha256(original_tag.encode()).hexdigest()[:16]
        target = repository + ":" + tail
        if target in targets:
            raise ValueError(f"duplicate target tag: {target}")
        targets.add(target)
        entry["target"] = target
    return manifest


def convert(archive, output, repository="local-train/scaleswe-shared"):
    from contextlib import closing
    archive, output = Path(archive).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"output already exists (never overwrite evidence): {output}")
    with tarfile.open(archive, "r:") as source:
        read_manifest(source, repository)
    output.mkdir(parents=True)
    try:
        with closing(sqlite3.connect(output / "index.sqlite3")) as db:
            try:
                return _convert(archive, output, repository, db)
            except sqlite3.IntegrityError as error:
                raise ValueError(f"duplicate source path or tag: {error}") from error
    except BaseException:
        # `output` is the resolved, previously absent directory created above.
        # The connection is already closed; never remove an existing evidence dir.
        shutil.rmtree(output)
        raise


def _convert(archive, output, repository, db):
    started = time.perf_counter()
    db.executescript(SCHEMA)
    originals = []
    with tarfile.open(archive, "r:") as source:
        manifest = read_manifest(source, repository)
        for index, entry in enumerate(manifest):
            config = json.load(source.extractfile(entry["Config"]))
            layer = output / f"original-{index}.tar"
            info = source.getmember(entry["Layers"][0])
            incoming = source.extractfile(info)
            magic = incoming.read(2)
            incoming.seek(0)
            stream = gzip.GzipFile(fileobj=incoming) if magic == b"\x1f\x8b" else incoming
            with layer.open("wb") as target:
                shutil.copyfileobj(stream, target, 1024 * 1024)
            diff_id = "sha256:" + sha_file(layer)
            if diff_id != config["rootfs"]["diff_ids"][0]:
                raise ValueError("source diff_id mismatch")
            db.execute("INSERT INTO images VALUES (?,?,?,?,?,?,?)", (index, entry["RepoTags"][0], json.dumps(config), str(layer), info.size, layer.stat().st_size, diff_id))
            index_layer(db, index, layer)
            originals.append({"tag": entry["RepoTags"][0], "target": entry["target"], "config": config, "layer": layer, "compressed_bytes": info.size})
            print(f"indexed {entry['RepoTags'][0]}", flush=True)
    shared, parents = select_shared(db, len(originals))
    base = output / "base.tar"
    base_paths = shared | parents
    write_layer(originals[0]["layer"], base, lambda member: canonical(member.name) in base_paths)
    layout = output / "oci"
    layout.mkdir()
    base_descriptor = blob(layout, path=base, compress=True) | {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip"}
    base_diff_id = "sha256:" + sha_file(base)
    manifests, reports = [], []
    for index, original in enumerate(originals):
        delta = output / f"delta-{index}.tar"
        write_layer(original["layer"], delta, lambda member: member.isdir() or canonical(member.name) not in shared)
        proof = verify_composition(db, index, base, delta)
        delta_descriptor = blob(layout, path=delta, compress=True) | {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip"}
        config = copy.deepcopy(original["config"])
        config["rootfs"]["diff_ids"] = [base_diff_id, "sha256:" + sha_file(delta)]
        config["history"] = [{"created_by": "shared_base: exact common files"}, {"created_by": "shared_base: exact task delta"}]
        config_descriptor = blob(layout, raw=json_bytes(config)) | {"mediaType": "application/vnd.oci.image.config.v1+json"}
        image_manifest = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json", "config": config_descriptor, "layers": [base_descriptor, delta_descriptor]}
        tag = original["target"]
        descriptor = blob(layout, raw=json_bytes(image_manifest)) | {"mediaType": image_manifest["mediaType"], "annotations": {"org.opencontainers.image.ref.name": tag.rsplit(":", 1)[1], "io.containerd.image.name": canonical_image_name(tag)}, "platform": {"os": config["os"], "architecture": config["architecture"]}}
        manifests.append(descriptor)
        report = {"source": original["tag"], "target": tag, "source_compressed_bytes": original["compressed_bytes"], "source_raw_bytes": original["layer"].stat().st_size, "delta_compressed_bytes": delta_descriptor["size"], "delta_raw_bytes": delta.stat().st_size, "config_preserved": config["config"] == original["config"]["config"], "filesystem_proof": proof}
        reports.append(report)
        print(json.dumps(report), flush=True)
    (layout / "index.json").write_bytes(json_bytes({"schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json", "manifests": manifests}))
    (layout / "oci-layout").write_bytes(json_bytes({"imageLayoutVersion": "1.0.0"}))
    oci_archive = output / "images.oci.tar"
    pack_oci(layout, oci_archive)
    total_original = sum(row["source_compressed_bytes"] for row in reports)
    total_shared = base_descriptor["size"] + sum(row["delta_compressed_bytes"] for row in reports)
    report = {"format_version": 1, "source_archive_sha256": sha_file(archive), "shared_entries": len(shared), "base_parent_directories": len(parents), "base_compressed_bytes": base_descriptor["size"], "base_raw_bytes": base.stat().st_size, "base_diff_id": base_diff_id, "original_compressed_bytes": total_original, "shared_compressed_bytes": total_shared, "compressed_saving_fraction": 1 - total_shared / total_original, "conversion_seconds": time.perf_counter() - started, "images": reports, "archive": str(oci_archive)}
    (output / "report.json").write_bytes(json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", default="local-train/scaleswe-shared")
    args = parser.parse_args()
    report = convert(args.archive, args.output, args.repository)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
