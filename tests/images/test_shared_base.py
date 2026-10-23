"""Unit tests for tools/shared_base.py (single-layer Docker-save -> OCI relayer).

Fixtures are synthetic Docker-save archives built with the standard library only:
two single-layer images that share byte-identical common files (content, metadata
and PAX xattrs) and differ in their task-specific files.

Stdlib unittest is used on purpose so this file runs without the repo's runtime
dependencies:  python -m unittest tests.images.test_shared_base -v
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import sqlite3
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "shared_base.py"


def _load_module():
    """Import tools/shared_base.py by path so `python -m unittest` needs no sys.path setup."""
    spec = importlib.util.spec_from_file_location("shared_base_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shared_base = _load_module()


# --------------------------------------------------------------------------- tar fixture helpers


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_path(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_info(
    name: str,
    data: bytes = b"",
    mode: int = 0o644,
    mtime: int = 1_600_000_000,
    uid: int = 0,
    gid: int = 0,
    pax_headers: dict[str, str] | None = None,
) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    info.mtime = mtime
    info.uid = uid
    info.gid = gid
    info.uname = "root"
    info.gname = "root"
    if pax_headers:
        info.pax_headers = dict(pax_headers)
    return info


def dir_info(name: str, mode: int = 0o755, mtime: int = 1_600_000_000) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = mode
    info.mtime = mtime
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info


def symlink_info(name: str, target: str, mode: int = 0o777, mtime: int = 1_600_000_000) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    info.mode = mode
    info.mtime = mtime
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info


def hardlink_info(name: str, target: str, mtime: int = 1_600_000_000) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.LNKTYPE
    info.linkname = target
    info.mode = 0o644
    info.mtime = mtime
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info


class Entry:
    """One layer member: a TarInfo plus the bytes that follow it in the tar stream."""

    def __init__(self, info: tarfile.TarInfo, data: bytes = b""):
        self.info = info
        self.data = data


def F(name: str, data: bytes = b"", **kwargs) -> Entry:
    """Regular file entry."""
    return Entry(file_info(name, data, **kwargs), data)


def D(name: str, **kwargs) -> Entry:
    """Directory entry."""
    return Entry(dir_info(name, **kwargs))


def L(name: str, target: str, **kwargs) -> Entry:
    """Symlink entry."""
    return Entry(symlink_info(name, target, **kwargs))


def H(name: str, target: str, **kwargs) -> Entry:
    """Hardlink entry."""
    return Entry(hardlink_info(name, target, **kwargs))


def build_tar(entries, fmt: int = tarfile.PAX_FORMAT) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=fmt) as archive:
        for entry in entries:
            payload = io.BytesIO(entry.data) if entry.info.isfile() else None
            archive.addfile(entry.info, payload)
    return buffer.getvalue()


def member_record(member: tarfile.TarInfo, payload: bytes = b"") -> dict:
    """Extracted view of a member, including everything the tool fingerprints."""
    return {
        "type": member.type.hex(),
        "mode": member.mode,
        "uid": member.uid,
        "gid": member.gid,
        "size": member.size,
        "mtime": member.mtime,
        "uname": member.uname,
        "gname": member.gname,
        "linkname": member.linkname,
        "devmajor": member.devmajor,
        "devminor": member.devminor,
        "pax_headers": dict(member.pax_headers),
        "content": payload,
    }


def tar_members(path: Path) -> dict[str, dict]:
    """Map normalised member name -> member_record for a layer tar on disk."""
    result: dict[str, dict] = {}
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            payload = archive.extractfile(member).read() if member.isfile() else b""
            result[shared_base.canonical(member.name)] = member_record(member, payload)
    return result


def tar_members_from_bytes(data: bytes) -> dict[str, dict]:
    path_like = io.BytesIO(data)
    result: dict[str, dict] = {}
    with tarfile.open(fileobj=path_like, mode="r:") as archive:
        for member in archive:
            payload = archive.extractfile(member).read() if member.isfile() else b""
            result[shared_base.canonical(member.name)] = member_record(member, payload)
    return result


def signature_of(record: dict) -> str:
    """Recompute the tool's fingerprint from an extracted member (independent ground truth)."""
    metadata = {key: record[key] for key in (
        "mode", "uid", "gid", "size", "mtime", "linkname", "uname", "gname",
        "devmajor", "devminor", "pax_headers",
    )} | {"type": record["type"]}
    content = sha_bytes(record["content"]) if record["type"] == tarfile.REGTYPE.hex() else ""
    return sha_bytes(shared_base.json_bytes([metadata, content]))


def assert_equivalent(test: unittest.TestCase, expected: dict[str, dict], actual: dict[str, dict]) -> None:
    test.assertEqual(sorted(expected), sorted(actual), "path set mismatch")
    for name, record in expected.items():
        test.assertEqual(signature_of(actual[name]), signature_of(record), f"member drift: {name}")


# --------------------------------------------------------------------------- source image fixtures


XATTR = {"SCHILY.xattr.user.swe": "factory", "SCHILY.xattr.user.run": "unit"}


def common_entries() -> list[Entry]:
    """Files both images share byte-for-byte, including xattrs and directory metadata."""
    return [
        D("."),
        D("etc"),
        D("etc/ssl"),
        D("opt"),
        D("opt/app"),
        F("etc/os-release", b"ID=ubuntu\nVERSION=22.04\n"),
        F("etc/passwd", b"root:x:0:0:root:/root:/bin/bash\n" * 8),
        F("etc/ssl/certs.pem", b"-----BEGIN CERTIFICATE-----\n" + b"A" * 4096),
        F("opt/app/config.json", b'{"shared": true}\n', mode=0o600, mtime=1_650_000_000),
        F("opt/app/data.bin", bytes(range(256)) * 64, mode=0o640, mtime=1_650_000_123),
        F("opt/app/legacy.bin", b"legacy payload\n", uid=1000, gid=1000),
        F("etc/marked", b"xattrs matter\n", pax_headers=XATTR),
        L("etc/localtime", "../usr/share/zoneinfo/UTC"),
        L("opt/app/shared-link", "config.json"),
    ]


def tail_entries(task_path: str, task_payload: bytes, image_key: str = "a") -> list[Entry]:
    """Members that must stay in the task delta even though both layers carry them.

    Directory timestamps differ per image on purpose: a directory with divergent
    metadata is *not* shareable, so it has to stay where each task's metadata wins.
    """
    task_paths = {"a": TASK_PATH_A, "b": TASK_PATH_B}
    other_task = task_paths["b" if image_key == "a" else "a"]
    return [
        # Hardlink whose referent is itself a shared file: neither side may leave the delta.
        F("opt/app/hl-a", b"hardlinked bytes\n"),
        H("opt/app/hl-b", "opt/app/hl-a"),
        # Dangling hardlink: its referent only exists in the other image.
        H("opt/app/wants", "opt/app/only-me"),
        # Directories are re-emitted per delta so each task's directory metadata wins.
        D("srv", mtime=SVR_MTIME[image_key]),
        D("srv/taskdir", mtime=TASKDIR_MTIME[image_key]),
        F(task_path, task_payload),
        F("srv/taskdir/answer.txt", b"answer\n" * 4),
    ]


TASK_PATH_A = "opt/app/task-a.py"
TASK_PATH_B = "opt/app/task-b.py"
TASK_PATH = TASK_PATH_A  # convenience alias for the primary image
SVR_MTIME = {"a": 1_111_111_111, "b": 1_444_444_444}
TASKDIR_MTIME = {"a": 1_222_222_222, "b": 1_555_555_555}


class Fixture:
    """Synthetic docker-save archive whose per-image extras can be customised."""

    TAGS = {"a": "orig-a:latest", "b": "orig-b:latest"}

    def __init__(self, root: Path, extra_a=(), extra_b=(), tags: dict[str, str] | None = None,
                 tails: dict[str, list["Entry"]] | None = None):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.extra_a = list(extra_a)
        self.extra_b = list(extra_b)
        self.tags = dict(self.TAGS if tags is None else tags)
        self.tails = tails or {}
        self.archive = root / "originals.tar"
        self.layer_a = root / "layer-a.tar"
        self.layer_b = root / "layer-b.tar"
        self.gz_a = root / "layer-a.tar.gz"
        self.gz_b = root / "layer-b.tar.gz"
        self.config_files = {"a": root / "config-a.json", "b": root / "config-b.json"}
        self.manifest_file = root / "manifest.json"
        self.layer_a.write_bytes(build_tar(common_entries() + self._tail("a") + self.extra_a))
        self.layer_b.write_bytes(build_tar(common_entries() + self._tail("b") + self.extra_b))
        for key, layer in (("a", self.layer_a), ("b", self.layer_b)):
            self.config_files[key].write_text(json.dumps({
                "architecture": "amd64",
                "os": "linux",
                "created": "2024-05-06T07:08:09.123456789Z",
                "config": {
                    "Env": ["PATH=/usr/local/bin:/usr/bin:/bin", "LANG=C.UTF-8"],
                    "Cmd": ["/bin/bash"],
                    "WorkingDir": "/opt/app",
                    "Labels": {"swe.factory": "shared-base", "image.index": key},
                    "Entrypoint": ["/bin/sh", "-c"],
                    "StopSignal": "SIGTERM",
                },
                "rootfs": {"type": "layers", "diff_ids": ["sha256:" + sha_path(layer)]},
                "history": [
                    {"created": "2024-05-06T07:08:09.123456789Z",
                     "created_by": "/bin/sh -c #(nop) ADD file:handwritten in config",
                     "empty_layer": False},
                ],
            }), encoding="utf-8")
        self.repack()

    # -- serialisation

    def _tail(self, key: str) -> list[Entry]:
        """Per-image task members: defaults, or a custom tail (which replaces the defaults)."""
        if key in self.tails:
            return list(self.tails[key])
        return tail_entries(TASK_PATH_A if key == "a" else TASK_PATH_B,
                            b"print('task a')\n" if key == "a" else b"print('task b')\n", key)

    def repack(self) -> None:
        """(Re)build the gzip layers and the docker-save archive from the current on-disk pieces."""
        for layer, gz in ((self.layer_a, self.gz_a), (self.layer_b, self.gz_b)):
            gz.write_bytes(gzip.compress(layer.read_bytes(), mtime=0))
        if not self.manifest_file.exists():
            self.manifest_file.write_text(json.dumps([
                {"Config": "config-a.json", "RepoTags": [self.tags["a"]], "Layers": ["layer-a.tar.gz"]},
                {"Config": "config-b.json", "RepoTags": [self.tags["b"]], "Layers": ["layer-b.tar.gz"]},
            ]), encoding="utf-8")
        with tarfile.open(self.archive, "w") as archive:
            for name in ("manifest.json", "config-a.json", "config-b.json",
                         "layer-a.tar.gz", "layer-b.tar.gz"):
                data = (self.root / name).read_bytes()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = 1_700_000_000
                archive.addfile(info, io.BytesIO(data))

    def rewrite_layer(self, key: str, entries) -> None:
        """Replace one source layer, keeping its config diff_id consistent with the new bytes."""
        layer = self.layer_a if key == "a" else self.layer_b
        layer.write_bytes(build_tar(entries))
        config = json.loads(self.config_files[key].read_text(encoding="utf-8"))
        config["rootfs"]["diff_ids"] = ["sha256:" + sha_path(layer)]
        self.config_files[key].write_text(json.dumps(config), encoding="utf-8")
        self.repack()

    def config(self, key: str) -> dict:
        return json.loads(self.config_files[key].read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- OCI output reader


class OCIImage:
    """Strict reader for the produced OCI archive: every blob is hash-verified."""

    def __init__(self, output: Path):
        self.output = output
        self.layout = output / "oci"
        self.index = json.loads((self.layout / "index.json").read_text())
        self.report = json.loads((output / "report.json").read_text())

    def raw_blob(self, descriptor: dict) -> bytes:
        digest = descriptor["digest"].split(":")[1]
        data = (self.layout / "blobs" / "sha256" / digest).read_bytes()
        assert sha_bytes(data) == digest, f"blob digest mismatch: {descriptor['digest']}"
        assert len(data) == descriptor["size"], f"descriptor size mismatch: {descriptor['digest']}"
        return data

    def json_blob(self, descriptor: dict) -> dict:
        return json.loads(self.raw_blob(descriptor))

    def manifests(self) -> list[dict]:
        return [self.json_blob(descriptor) for descriptor in self.index["manifests"]]

    def configs(self) -> list[dict]:
        return [self.json_blob(manifest["config"]) for manifest in self.manifests()]

    def layer_raw(self, descriptor: dict) -> bytes:
        raw = self.raw_blob(descriptor)
        return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw

    def layer(self, descriptor: dict) -> dict[str, dict]:
        return tar_members_from_bytes(self.layer_raw(descriptor))

    def layer_diff_id(self, descriptor: dict) -> str:
        return "sha256:" + sha_bytes(self.layer_raw(descriptor))

    def composed(self, position: int) -> dict[str, dict]:
        base, delta = self.manifests()[position]["layers"]
        return {**self.layer(base), **self.layer(delta)}


def convert(fixture: Fixture, output: Path, repository: str = "local-train/scaleswe-shared") -> OCIImage:
    shared_base.convert(fixture.archive, output, repository)
    return OCIImage(output)


class TempCase(unittest.TestCase):
    """Each test gets a scratch directory; nothing ever touches results/ or data/."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="shared-base-test-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def fixture(self, name: str = "fixture", **kwargs) -> Fixture:
        return Fixture(self.tmp / name, **kwargs)


# --------------------------------------------------------------------------- conversion invariants


class ConversionTest(TempCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._class_tmp = tempfile.TemporaryDirectory(prefix="shared-base-conversion-")
        root = Path(cls._class_tmp.name)
        cls.fixture_obj = Fixture(root / "fixture")
        cls.fixture_sha = sha_path(cls.fixture_obj.archive)
        cls.source_layers = {key: cls.fixture_obj.layer_a if key == "a" else cls.fixture_obj.layer_b
                             for key in ("a", "b")}
        cls.source_bytes = {key: path.read_bytes() for key, path in cls.source_layers.items()}
        cls.source_members = {key: tar_members(path) for key, path in cls.source_layers.items()}
        cls.base_signatures = {
            shared_base.canonical(entry.info.name): signature_of(member_record(entry.info, entry.data))
            for entry in common_entries()
        }
        cls.output = root / "out"
        shared_base.convert(cls.fixture_obj.archive, cls.output)
        cls.oci = OCIImage(cls.output)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._class_tmp.cleanup()

    # -- OCI structure

    def test_index_layout_and_tags(self) -> None:
        self.assertEqual(self.oci.index["schemaVersion"], 2)
        self.assertEqual(self.oci.index["mediaType"], "application/vnd.oci.image.index.v1+json")
        self.assertEqual(len(self.oci.index["manifests"]), 2)
        self.assertEqual(json.loads((self.oci.layout / "oci-layout").read_text()),
                         {"imageLayoutVersion": "1.0.0"})
        annotations = [descriptor["annotations"] for descriptor in self.oci.index["manifests"]]
        # Two images may not share one tag: `docker load` would silently keep only the last.
        tags = [item["org.opencontainers.image.ref.name"] for item in annotations]
        self.assertEqual(len(set(tags)), 2, f"duplicate output tag: {tags}")
        names = [item["io.containerd.image.name"] for item in annotations]
        self.assertEqual(len(set(names)), 2, f"duplicate output image name: {names}")
        targets = [image["target"] for image in self.oci.report["images"]]
        self.assertEqual(sorted(targets), sorted(f"local-train/scaleswe-shared:{tag}" for tag in tags))
        self.assertTrue((self.oci.output / "images.oci.tar").is_file())

    def test_both_manifests_reference_the_identical_base_layer(self) -> None:
        # Do not assume an image shape: require identical layer *content* where it must match.
        def layers(position: int) -> list[dict[str, dict]]:
            return [self.oci.layer(descriptor) for descriptor in self.oci.manifests()[position]["layers"]]

        first, second = layers(0), layers(1)
        self.assertTrue(first and second, "each image needs at least one layer")
        shared_prefix = 0
        for left, right in zip(first, second):
            if {name: signature_of(record) for name, record in left.items()} != \
                    {name: signature_of(record) for name, record in right.items()}:
                break
            shared_prefix += 1
        self.assertGreaterEqual(shared_prefix, 1, "both images must share a byte-identical base layer")
        # The genuinely common files must live in that shared base, not be duplicated per image.
        for name in ("etc/os-release", "etc/passwd", "opt/app/data.bin"):
            self.assertIn(name, first[0], f"{name} must sit in the shared base")

    def test_manifest_platform_matches_source_config(self) -> None:
        for descriptor in self.oci.index["manifests"]:
            self.assertEqual(descriptor["platform"], {"os": "linux", "architecture": "amd64"})

    # -- what lands in the shared base

    def test_base_holds_exactly_the_shared_paths_with_identical_metadata(self) -> None:
        base = self.oci.layer(self.oci.manifests()[0]["layers"][0])
        self.assertTrue(base, "base layer must not be empty")
        for name in (TASK_PATH_A, TASK_PATH_B, "opt/app/hl-a", "opt/app/hl-b",
                     "opt/app/wants", "srv", "srv/taskdir", "srv/taskdir/answer.txt"):
            self.assertNotIn(name, base, f"task-specific path leaked into base: {name}")
        for name, record in base.items():
            self.assertIn(name, self.base_signatures, f"unexpected path in base: {name}")
            self.assertEqual(signature_of(record), self.base_signatures[name], f"metadata drift: {name}")
        self.assertTrue({"etc/os-release", "etc/passwd", "etc/marked", "etc/localtime"} <= set(base))

    def test_whiteouts_never_appear_in_output_layers(self) -> None:
        for position in range(2):
            composed = self.oci.composed(position)
            for name in composed:
                self.assertFalse(name.rsplit("/", 1)[-1].startswith(".wh."), f"whiteout emitted: {name}")

    # -- config fidelity

    def test_source_config_fields_survive_verbatim(self) -> None:
        # Every source field outside the derived ones (rootfs diff_ids, history) survives.
        derived = {"rootfs", "history"}
        for position, key in enumerate(("a", "b")):
            source = self.fixture_obj.config(key)
            produced = self.oci.configs()[position]
            for field, value in source.items():
                if field in derived:
                    continue
                self.assertEqual(produced[field], value, f"{key}:{field}")
            self.assertNotEqual(set(source), set(), "fixture config must not be empty")
            self.assertEqual(sorted(set(source) - set(produced)), [], f"{key}: fields dropped")

    def test_unknown_config_keys_are_carried_over(self) -> None:
        # The source config is the OCI config; unknown keys must not be silently dropped.
        root = self.tmp / "odd"
        fixture = Fixture(root)
        config = fixture.config("a")
        config["os.version"] = "10.0.22631.3593"
        config["config"]["User"] = "1001:1001"
        fixture.config_files["a"].write_text(json.dumps(config), encoding="utf-8")
        fixture.repack()
        oci = convert(fixture, self.tmp / "odd-out")
        produced = oci.configs()[0]
        self.assertEqual(produced["os.version"], "10.0.22631.3593")
        self.assertEqual(produced["config"]["User"], "1001:1001")

    def test_diff_ids_are_reproducible_from_layer_bytes(self) -> None:
        for manifest, config in zip(self.oci.manifests(), self.oci.configs()):
            recomputed = [self.oci.layer_diff_id(descriptor) for descriptor in manifest["layers"]]
            self.assertEqual(config["rootfs"]["diff_ids"], recomputed)
        self.assertEqual(self.oci.configs()[0]["rootfs"]["diff_ids"][0],
                         self.oci.configs()[1]["rootfs"]["diff_ids"][0])
        self.assertNotEqual(self.oci.configs()[0]["rootfs"]["diff_ids"][1],
                            self.oci.configs()[1]["rootfs"]["diff_ids"][1])

    def test_history_matches_the_emitted_layers(self) -> None:
        # Derived from the sources but rewritten: the output must stay loadable and honest.
        for position in range(2):
            produced = self.oci.configs()[position]
            diff_ids = produced["rootfs"]["diff_ids"]
            self.assertEqual(len(diff_ids), 2, "base + task delta")
            history = produced["history"]
            self.assertTrue(history, "history must not be empty")
            non_empty = [entry for entry in history if not entry.get("empty_layer")]
            self.assertEqual(len(non_empty), len(diff_ids), "one history step per layer")
            for entry in non_empty:
                self.assertTrue(entry.get("created_by"), "every step needs created_by")
                self.assertIn("shared_base", entry["created_by"])

    # -- composition is lossless

    def test_every_source_member_round_trips_byte_exactly(self) -> None:
        # Independent of the tool's own proof: re-read the emitted layers ourselves.
        for position, key in enumerate(("a", "b")):
            assert_equivalent(self, self.source_members[key], self.oci.composed(position))

    def test_directories_are_reemitted_in_every_delta(self) -> None:
        for position, key in enumerate(("a", "b")):
            delta = self.oci.layer(self.oci.manifests()[position]["layers"][1])
            for name, record in self.source_members[key].items():
                if record["type"] == tarfile.DIRTYPE.hex():
                    self.assertIn(name, delta, f"directory {name} missing from delta {position}")

    def test_directory_mtime_differences_survive(self) -> None:
        # Every image keeps its own directory timestamps (none of them may collapse to base's).
        for position, key in enumerate(("a", "b")):
            source = self.source_members[key]
            composed = self.oci.composed(position)
            for name in ("srv", "srv/taskdir", "."):
                self.assertEqual(composed[name]["mtime"], source[name]["mtime"],
                                 f"{name} mtime drifted in image {key}")
            self.assertNotEqual(SVR_MTIME["a"], SVR_MTIME["b"], "fixture must make the dirs differ")

    def test_conflicting_directory_metadata_is_never_decided_by_image_zero(self) -> None:
        base = self.oci.layer(self.oci.manifests()[0]["layers"][0])
        for name in ("srv", "srv/taskdir"):
            self.assertNotIn(name, base, f"{name} has per-image metadata and must not be shared")

    def test_hardlinks_and_their_referents_never_cross_layers(self) -> None:
        for position in range(2):
            base = self.oci.layer(self.oci.manifests()[position]["layers"][0])
            delta = self.oci.layer(self.oci.manifests()[position]["layers"][1])
            self.assertNotIn("opt/app/hl-a", base)
            self.assertNotIn("opt/app/hl-b", base)
            self.assertIn("opt/app/hl-a", delta)
            self.assertIn("opt/app/hl-b", delta)
            self.assertEqual(delta["opt/app/hl-b"]["linkname"], "opt/app/hl-a")
            names = set(delta)
            for name, record in delta.items():
                if record["type"] == tarfile.LNKTYPE.hex():
                    target = shared_base.canonical(record["linkname"])
                    if name == "opt/app/wants":
                        # Dangling in the source too: no image contains the referent.
                        self.assertNotIn(target, names)
                        continue
                    self.assertIn(target, names,
                                  f"hardlink {name} in the delta points outside its own layer")

    def test_hardlink_content_is_bit_exact(self) -> None:
        composed = self.oci.composed(0)
        source = self.source_members["a"]
        self.assertEqual(composed["opt/app/hl-a"], source["opt/app/hl-a"])
        self.assertEqual(composed["opt/app/hl-b"]["linkname"], source["opt/app/hl-b"]["linkname"])

    def test_xattrs_survive_the_delta(self) -> None:
        composed = self.oci.composed(0)
        self.assertEqual(composed["etc/marked"]["pax_headers"].get("SCHILY.xattr.user.swe"), "factory")

    # -- inputs stay untouched, evidence stays honest

    def test_source_archive_is_never_modified(self) -> None:
        self.assertEqual(sha_path(self.fixture_obj.archive), self.fixture_sha)
        for key, path in self.source_layers.items():
            self.assertEqual(path.read_bytes(), self.source_bytes[key], f"source layer {key} mutated")
        with tarfile.open(self.fixture_obj.archive, "r:") as archive:
            self.assertEqual(sorted(archive.getnames()),
                             ["config-a.json", "config-b.json", "layer-a.tar.gz",
                              "layer-b.tar.gz", "manifest.json"])
        for index, key in enumerate(("a", "b")):
            self.assertEqual(sha_path(self.oci.output / f"original-{index}.tar"),
                             sha_path(self.source_layers[key]))

    def test_report_matches_the_bytes_on_disk(self) -> None:
        report = self.oci.report
        self.assertEqual(report["format_version"], 1)
        self.assertEqual(report["source_archive_sha256"], sha_path(self.fixture_obj.archive))
        self.assertEqual(len(report["images"]), 2)
        self.assertEqual(report["base_diff_id"], self.oci.layer_diff_id(self.oci.manifests()[0]["layers"][0]))
        for image in report["images"]:
            self.assertEqual(image["filesystem_proof"]["missing_or_changed"], 0)
            self.assertEqual(image["filesystem_proof"]["extra"], 0)
            self.assertGreater(image["filesystem_proof"]["entries"], 0)
            self.assertTrue(image["config_preserved"])
        deltas = [image["delta_compressed_bytes"] for image in report["images"]]
        self.assertEqual(report["shared_compressed_bytes"], report["base_compressed_bytes"] + sum(deltas))
        self.assertGreater(report["compressed_saving_fraction"], 0)

    def test_index_sqlite_records_both_images(self) -> None:
        db = sqlite3.connect(self.oci.output / "index.sqlite3")
        try:
            rows = db.execute("SELECT id, tag FROM images ORDER BY id").fetchall()
            self.assertEqual(rows, [(0, "orig-a:latest"), (1, "orig-b:latest")])
            self.assertEqual(db.execute("SELECT COUNT(*) FROM shared").fetchone()[0],
                             self.oci.report["shared_entries"])
            for path in ("opt/app/task-a.py", "opt/app/task-b.py"):
                self.assertEqual(db.execute("SELECT COUNT(*) FROM entries WHERE path=?", (path,)).fetchone()[0], 1)
        finally:
            db.close()


class EquivalentLayoutsTest(TempCase):
    """Different task path sets (and identical inputs) must convert equally well."""

    def _make_archive(self, root: Path, layer_payloads: list[tuple[bytes, bytes]], extra=None) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        manifest = []
        names = []
        for index, (layer, config_payload) in enumerate(layer_payloads):
            gz_name = f"l{index}.tar.gz"
            (root / gz_name).write_bytes(gzip.compress(layer, mtime=0))
            (root / f"c{index}.json").write_bytes(config_payload)
            names += [f"c{index}.json", gz_name]
            manifest.append({"Config": f"c{index}.json", "RepoTags": [f"img:{index}"], "Layers": [gz_name]})
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        archive = root / "originals.tar"
        with tarfile.open(archive, "w") as packed:
            for name in ["manifest.json"] + names:
                data = (root / name).read_bytes()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                packed.addfile(info, io.BytesIO(data))
        return archive

    @staticmethod
    def _config(layer: bytes) -> bytes:
        return json.dumps({
            "architecture": "arm64",
            "os": "linux",
            "config": {"Env": ["A=1"], "Cmd": ["sh"]},
            "rootfs": {"type": "layers", "diff_ids": ["sha256:" + sha_bytes(layer)]},
            "history": [{"created": "2024-01-01T00:00:00Z", "created_by": "layer"}],
        }).encode()

    def _check(self, archive: Path, layers: list[bytes]) -> OCIImage:
        oci = convert(_ArchiveOnly(archive), self.tmp / "out")
        self.assertEqual(oci.manifests()[0]["layers"][0], oci.manifests()[1]["layers"][0])
        for position, layer in enumerate(layers):
            expected = tar_members_from_bytes(layer)
            assert_equivalent(self, expected, oci.composed(position))
        return oci

    def test_same_task_path_set_with_different_content(self) -> None:
        first = build_tar(common_entries() + [
            F("opt/app/entry.py", b"print('a')\n"),
            F("opt/app/helper.py", b"a = 1\n"),
            D("opt/app/pkg"),
        ])
        second = build_tar(common_entries() + [
            F("opt/app/entry.py", b"print('b')\n"),
            F("opt/app/helper.py", b"b = 2\n"),
            D("opt/app/pkg"),
        ])
        archive = self._make_archive(self.tmp / "same", [(first, self._config(first)), (second, self._config(second))])
        self._check(archive, [first, second])

    def test_disjoint_task_path_sets(self) -> None:
        first = build_tar(common_entries() + [F("opt/app/only-a", b"a\n")])
        second = build_tar(common_entries() + [H("opt/app/only-b", "opt/app/only-b"), D("srv")])
        archive = self._make_archive(self.tmp / "disjoint",
                                     [(first, self._config(first)), (second, self._config(second))])
        self._check(archive, [first, second])

    def test_identical_input_images(self) -> None:
        layer = build_tar(common_entries() + [F("opt/app/task.py", b"s\n")])
        archive = self._make_archive(self.tmp / "identical",
                                     [(layer, self._config(layer)), (layer, self._config(layer))])
        oci = self._check(archive, [layer, layer])
        self.assertEqual(oci.configs()[0]["architecture"], "arm64")
        self.assertEqual(oci.configs()[0]["config"], {"Env": ["A=1"], "Cmd": ["sh"]})


class _ArchiveOnly:
    """Minimal stand-in accepted by shared_base.convert (it only needs .archive)."""

    def __init__(self, archive: Path):
        self.archive = archive


class MetadataSensitiveSharingTest(TempCase):
    """Equal bytes with unequal metadata must never be shared."""

    def _convert(self, extra_a, extra_b, name: str = "fx", tails=None) -> OCIImage:
        fixture = Fixture(self.tmp / name, extra_a=extra_a, extra_b=extra_b, tails=tails)
        return convert(fixture, self.tmp / f"{name}-out")

    def _assert_not_shared(self, oci: OCIImage, name: str) -> None:
        for position in range(2):
            base = oci.layer(oci.manifests()[position]["layers"][0])
            delta = oci.layer(oci.manifests()[position]["layers"][1])
            self.assertNotIn(name, base, f"{name} must not be shared")
            self.assertIn(name, delta, f"{name} must stay in the delta")

    def test_same_content_different_mode_is_not_shared(self) -> None:
        payload = b"identical bytes\n"
        oci = self._convert([F("opt/app/mode-only", payload, mode=0o644)],
                            [F("opt/app/mode-only", payload, mode=0o755)])
        self._assert_not_shared(oci, "opt/app/mode-only")
        self.assertNotEqual(oci.configs()[0]["rootfs"]["diff_ids"][1],
                            oci.configs()[1]["rootfs"]["diff_ids"][1])

    def test_same_content_different_mtime_is_not_shared(self) -> None:
        payload = b"identical bytes\n"
        oci = self._convert([F("opt/app/mtime-only", payload, mtime=1_700_000_001)],
                            [F("opt/app/mtime-only", payload, mtime=1_700_000_002)])
        self._assert_not_shared(oci, "opt/app/mtime-only")

    def test_same_content_different_xattr_is_not_shared(self) -> None:
        payload = b"identical bytes\n"
        oci = self._convert([F("opt/app/xattr-only", payload, pax_headers={"SCHILY.xattr.user.foo": "one"})],
                            [F("opt/app/xattr-only", payload, pax_headers={"SCHILY.xattr.user.foo": "two"})])
        self._assert_not_shared(oci, "opt/app/xattr-only")

    def test_same_content_different_owner_is_not_shared(self) -> None:
        payload = b"identical bytes\n"
        oci = self._convert([F("opt/app/owner-only", payload, uid=0, gid=0)],
                            [F("opt/app/owner-only", payload, uid=1000, gid=1000)])
        self._assert_not_shared(oci, "opt/app/owner-only")

    def test_file_versus_symlink_is_not_shared(self) -> None:
        oci = self._convert([F("opt/app/pair", b"target\n")], [L("opt/app/pair", "target")])
        self._assert_not_shared(oci, "opt/app/pair")

    def test_task_directory_with_divergent_metadata_is_not_shared(self) -> None:
        # A directory with per-image metadata must never be pinned into the shared base.
        oci = self._convert([], [], tails={
            "a": [D("srv", mtime=1_100), D("srv/taskdir", mtime=1_000), F("srv/taskdir/a.txt", b"a\n")],
            "b": [D("srv", mtime=1_500), D("srv/taskdir", mtime=2_000), F("srv/taskdir/a.txt", b"a\n")],
        })
        for position in range(2):
            base = oci.layer(oci.manifests()[position]["layers"][0])
            self.assertNotIn("srv", base, "the base must not carry a directory image 0 decided")
            self.assertNotIn("srv/taskdir", base)
        self.assertEqual(oci.composed(0)["srv/taskdir"]["mtime"], 1_000)
        self.assertEqual(oci.composed(1)["srv/taskdir"]["mtime"], 2_000)
        # Documents the deliberately conservative rule: a file whose ancestor metadata
        # differs is not shared either, so it stays in both deltas.
        for position in range(2):
            delta = oci.layer(oci.manifests()[position]["layers"][1])
            self.assertIn("srv/taskdir/a.txt", delta)


# --------------------------------------------------------------------------- fail-closed behaviour


class FailClosedTest(TempCase):
    def _refuse(self, fixture: Fixture, needle: str | None = None) -> None:
        output = self.tmp / "refused-out"
        with self.assertRaises(ValueError) as caught:
            shared_base.convert(fixture.archive, output)
        if needle is not None:
            self.assertIn(needle, str(caught.exception))
        self.assertFalse((output / "images.oci.tar").exists(), "no OCI archive may be emitted")
        # Fail closed means fail *before* evidence is produced: no layer, layout or index.
        for name in ("base.tar", "delta-0.tar", "original-0.tar", "oci", "index.sqlite3", "report.json"):
            self.assertFalse((output / name).exists(), f"refused conversion left {name} behind")

    def _with_member(self, info: tarfile.TarInfo, data: bytes = b"", key: str = "a") -> Fixture:
        fixture = self.fixture()
        original = (fixture.layer_a if key == "a" else fixture.layer_b).read_bytes()
        layer = build_tar(tar_members_to_entries(original) + [Entry(info, data)])
        fixture.rewrite_layer(key, tar_members_to_entries(layer))
        return fixture

    # -- destructive / unsupported members

    def test_whiteout_file_is_rejected(self) -> None:
        fixture = self.fixture()
        layer = build_tar(tar_members_to_entries(fixture.layer_a.read_bytes()) + [F("etc/.wh.deleted", b"")])
        fixture.rewrite_layer("a", tar_members_to_entries(layer))
        self._refuse(fixture, "flat")

    def test_whiteout_directory_prefix_is_rejected(self) -> None:
        fixture = self.fixture()
        layer = build_tar(tar_members_to_entries(fixture.layer_a.read_bytes()) + [
            D(".wh.gone"), F(".wh.gone/f", b"x"),
        ])
        fixture.rewrite_layer("a", tar_members_to_entries(layer))
        self._refuse(fixture, "flat")

    def test_character_device_is_rejected(self) -> None:
        fixture = self.fixture()
        info = tarfile.TarInfo("dev/null")
        info.type = tarfile.CHRTYPE
        info.mode = 0o666
        info.mtime = 1_600_000_000
        info.devmajor, info.devminor = 1, 3
        layer = build_tar(tar_members_to_entries(fixture.layer_a.read_bytes()) + [Entry(info)])
        fixture.rewrite_layer("a", tar_members_to_entries(layer))
        self._refuse(fixture, "special")

    def test_fifo_is_rejected(self) -> None:
        fixture = self.fixture()
        info = tarfile.TarInfo("run/pipe")
        info.type = tarfile.FIFOTYPE
        info.mode = 0o644
        info.mtime = 1_600_000_000
        layer = build_tar(tar_members_to_entries(fixture.layer_b.read_bytes()) + [Entry(info)])
        fixture.rewrite_layer("b", tar_members_to_entries(layer))
        self._refuse(fixture, "special")

    def test_pax_sparse_entry_is_rejected(self) -> None:
        fixture = self.fixture()
        info = file_info("opt/app/sparse-file", b"sparse\n")
        info.pax_headers = {
            "GNU.sparse.major": "1", "GNU.sparse.minor": "0",
            "GNU.sparse.name": "opt/app/sparse-file", "GNU.sparse.size": "1024",
            "GNU.sparse.numblocks": "1",
        }
        layer = build_tar(tar_members_to_entries(fixture.layer_b.read_bytes()) + [Entry(info, b"sparse\n")])
        fixture.rewrite_layer("b", tar_members_to_entries(layer))
        self._refuse(fixture, "sparse")

    def test_duplicate_path_is_rejected(self) -> None:
        fixture = self.fixture()
        layer = build_tar(tar_members_to_entries(fixture.layer_a.read_bytes()) + [F("etc//passwd", b"evil\n")])
        fixture.rewrite_layer("a", tar_members_to_entries(layer))
        self._refuse(fixture, "duplicate")

    def test_duplicate_path_beyond_one_insert_batch_is_rejected(self) -> None:
        # The indexer flushes every 1000 rows, so a duplicate far apart must still be caught.
        filler = [F(f"opt/app/filler/{index:04d}.txt", b"x\n") for index in range(1200)]
        fixture = self.fixture()
        layer = build_tar(tar_members_to_entries(fixture.layer_a.read_bytes()) + filler
                          + [F("etc//passwd", b"evil\n")])
        fixture.rewrite_layer("a", tar_members_to_entries(layer))
        self._refuse(fixture, "duplicate")

    def test_absolute_and_traversing_member_names_are_rejected(self) -> None:
        for name in ("/etc/shadow", "etc/../../etc/shadow"):
            fixture = self.fixture()
            layer = build_tar(tar_members_to_entries(fixture.layer_a.read_bytes()) + [F(name, b"evil\n")])
            fixture.rewrite_layer("a", tar_members_to_entries(layer))
            self._refuse(fixture, "unsafe")

    # -- unsupported image shapes

    def test_multilayer_image_is_rejected(self) -> None:
        fixture = self.fixture()
        manifest = json.loads(fixture.manifest_file.read_text(encoding="utf-8"))
        manifest[1]["Layers"] = ["layer-b.tar.gz", "layer-a.tar.gz"]
        fixture.manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
        config = fixture.config("b")
        config["rootfs"]["diff_ids"] = [config["rootfs"]["diff_ids"][0], "sha256:" + sha_path(fixture.layer_a)]
        fixture.config_files["b"].write_text(json.dumps(config), encoding="utf-8")
        fixture.repack()
        self._refuse(fixture, "single-layer")

    def test_single_image_archive_is_rejected(self) -> None:
        fixture = self.fixture()
        manifest = json.loads(fixture.manifest_file.read_text(encoding="utf-8"))
        fixture.manifest_file.write_text(json.dumps(manifest[:1]), encoding="utf-8")
        fixture.repack()
        self._refuse(fixture, "two source images")

    def test_duplicate_output_tag_is_rejected(self) -> None:
        # Two sources tagged identically would produce one target name for two images.
        fixture = self.fixture(tags={"a": "orig-x:v1", "b": "orig-x:v1"})
        self._refuse(fixture, "duplicate target tag")

    def test_diff_id_mismatch_is_rejected(self) -> None:
        fixture = self.fixture()
        config = fixture.config("a")
        config["rootfs"]["diff_ids"] = ["sha256:" + "0" * 64]
        fixture.config_files["a"].write_text(json.dumps(config), encoding="utf-8")
        fixture.repack()
        self._refuse(fixture, "diff_id")

    def test_two_diff_ids_for_one_layer_is_rejected(self) -> None:
        fixture = self.fixture()
        config = fixture.config("a")
        config["rootfs"]["diff_ids"] = config["rootfs"]["diff_ids"] + ["sha256:" + "1" * 64]
        fixture.config_files["a"].write_text(json.dumps(config), encoding="utf-8")
        fixture.repack()
        self._refuse(fixture, "diff_ids")

    # -- evidence protection

    def test_existing_output_is_never_overwritten(self) -> None:
        fixture = self.fixture()
        output = self.tmp / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("do not touch\n")
        with self.assertRaises(ValueError) as caught:
            shared_base.convert(fixture.archive, output)
        self.assertIn("already exists", str(caught.exception))
        self.assertEqual(marker.read_text(), "do not touch\n")
        self.assertEqual(sorted(path.name for path in output.iterdir()), ["keep.txt"])


def tar_members_to_entries(data: bytes) -> list[Entry]:
    """Re-read a layer tar into Entry objects so tests can append/mutate members."""
    entries: list[Entry] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for member in archive:
            payload = archive.extractfile(member).read() if member.isfile() else b""
            entries.append(Entry(member, payload))
    return entries


# --------------------------------------------------------------------------- helper units


class CanonicalPathTest(unittest.TestCase):
    def test_normalises_redundant_separators(self) -> None:
        self.assertEqual(shared_base.canonical("a//b"), "a/b")
        self.assertEqual(shared_base.canonical("a/./b"), "a/b")

    def test_rejects_absolute_traversal_and_nul(self) -> None:
        for name in ("/etc/passwd", "a/../etc/passwd", "a\x00b"):
            with self.assertRaises(ValueError):
                shared_base.canonical(name)

    def test_empty_name_normalises_to_dot(self) -> None:
        # Documents current behaviour: an empty member name collapses to ".".
        self.assertEqual(shared_base.canonical(""), ".")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
