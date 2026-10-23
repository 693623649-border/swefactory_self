"""Paired, offline smoke and public-test comparison for shared_base.py output.

Never calls an LLM, a hidden judge, or production. Keeps raw receipts in --output.
The transfer test is a controlled local HTTP blob-cache test, NOT Docker Hub speed.
"""
from __future__ import annotations
import argparse
import hashlib
import http.server
import json
import shutil
import statistics
import subprocess
import tarfile
import threading
import time
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

SMOKE = '''import json,subprocess,sys
meta={"python":sys.version,"head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"git_status":subprocess.check_output(["git","status","--porcelain=v1","--untracked-files=no"],text=True),"packages":subprocess.check_output(["python","-m","pip","freeze","--all"],text=True).splitlines()}
print("SHARED_BASE_META="+json.dumps(meta,sort_keys=True),flush=True)
sys.exit(subprocess.call(["python","-m","pytest","-q","--disable-warnings","--color=no","--junitxml=/tmp/shared-base-tests.xml"]))
'''


def run(args, timeout=240):
    start = time.perf_counter()
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return result, time.perf_counter() - start


def must(args, timeout=240):
    result, elapsed = run(args, timeout)
    if result.returncode:
        raise RuntimeError(f"command failed: {args!r}\n{result.stderr}")
    return result.stdout, elapsed


def inspect(tag):
    text, _ = must(["docker", "image", "inspect", tag])
    return json.loads(text)[0]


def smoke(tag, directory, script):
    name = "shared-base-check-" + uuid.uuid4().hex[:12]
    directory.mkdir()
    created = False
    try:
        _, create_s = must(["docker", "create", "--network", "none", "--name", name, tag,
                            "python", "/tmp/shared-base-smoke.py"])
        created = True
        must(["docker", "cp", str(script), name + ":/tmp/shared-base-smoke.py"])
        result, test_s = run(["docker", "start", "--attach", name])
        (directory / "stdout.txt").write_text(result.stdout, encoding="utf-8")
        (directory / "stderr.txt").write_text(result.stderr, encoding="utf-8")
        state, _ = must(["docker", "inspect", "--format", "{{json .State}}", name])
        state = json.loads(state)
        copy_result, _ = run(["docker", "cp", name + ":/tmp/shared-base-tests.xml", str(directory / "tests.xml")])
        if copy_result.returncode:
            raise RuntimeError("public pytest did not produce JUnit XML: " + result.stdout[-2000:])
        cases = []
        for case in ET.parse(directory / "tests.xml").iter("testcase"):
            status = next((key for key in ("failure", "error", "skipped") if case.find(key) is not None), "passed")
            cases.append([case.get("classname"), case.get("name"), status])
        meta = next((json.loads(line.split("=", 1)[1]) for line in result.stdout.splitlines() if line.startswith("SHARED_BASE_META=")), None)
        if meta is None or not cases:
            raise RuntimeError("missing metadata or public testcase evidence")
        return {"metadata": meta, "cases": sorted(cases), "exit_code": state["ExitCode"], "create_seconds": create_s, "test_seconds": test_s, "state_error": state.get("Error", "")}
    finally:
        if created:
            # Only this function's explicitly named disposable container is removed.
            run(["docker", "rm", "--force", name])


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def transfer_comparison(original_archive, converted, output):
    """Real HTTP GETs with empty independent per-arm digest caches; full SHA checks."""
    source_dir = output / "original-blobs"
    source_dir.mkdir()
    original_layers = []
    with tarfile.open(original_archive, "r:") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        for row in manifest:
            layers = []
            for name in row["Layers"]:
                digest = name.rsplit("/", 1)[1]
                path = source_dir / digest
                with archive.extractfile(name) as src, path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                layers.append(("original-blobs/" + digest, digest))
            original_layers.append(layers)
    # Serve both arms under one root. Copy only the small OCI index/manifests;
    # shared blobs are hardlinked on the same volume, never read into memory.
    shared_dir = output / "shared-blobs"
    shared_dir.mkdir()
    for source in (converted / "oci" / "blobs" / "sha256").iterdir():
        (shared_dir / source.name).hardlink_to(source)
    index = json.loads((converted / "oci" / "index.json").read_bytes())
    shared_layers = []
    for descriptor in index["manifests"]:
        manifest = json.loads((shared_dir / descriptor["digest"].split(":")[1]).read_bytes())
        shared_layers.append([("shared-blobs/" + layer["digest"].split(":")[1], layer["digest"].split(":")[1]) for layer in manifest["layers"]])
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *a, **kw: QuietHandler(*a, directory=str(output), **kw))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    results = {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for arm, images in (("original", original_layers), ("shared", shared_layers)):
            cache = output / (arm + "-http-cache")
            cache.mkdir()
            times, transferred, hits = [], [], []
            for layers in images:
                count = 0
                cached = 0
                started = time.perf_counter()
                for path, digest in layers:
                    destination = cache / digest
                    if destination.exists():
                        cached += 1
                        continue
                    hasher = hashlib.sha256()
                    with opener.open(f"http://127.0.0.1:{server.server_port}/{path}", timeout=60) as response, destination.open("wb") as target:
                        while block := response.read(1024 * 1024):
                            target.write(block)
                            hasher.update(block)
                            count += len(block)
                    if hasher.hexdigest() != digest:
                        raise ValueError("HTTP blob digest mismatch")
                times.append(time.perf_counter() - started)
                transferred.append(count)
                hits.append(cached)
            results[arm] = {"seconds": times, "bytes": transferred, "cache_hits": hits, "total_seconds": sum(times), "total_bytes": sum(transferred)}
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    results["scope"] = "local HTTP, sequential images, independent empty caches; layer payload only; not a Docker Hub or Docker pull throughput claim"
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--original-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir()  # Refuse overwriting earlier evidence.
    converted = args.converted.resolve()
    conversion = json.loads((converted / "report.json").read_bytes())
    script = output / "smoke.py"
    script.write_text(SMOKE, encoding="utf-8")
    load_text, load_s = must(["docker", "load", "--input", str(converted / "images.oci.tar")], timeout=600)
    (output / "load.txt").write_text(load_text, encoding="utf-8")
    pairs = []
    for index, row in enumerate(conversion["images"]):
        original, shared = inspect(row["source"]), inspect(row["target"])
        config_equal = original["Config"] == shared["Config"]
        if not config_equal:
            (output / f"config-{index}.json").write_text(json.dumps([original["Config"], shared["Config"]], indent=2), encoding="utf-8")
            raise ValueError(f"runtime Config mismatch: {row['source']}")
        if len(shared["RootFS"]["Layers"]) != 2 or shared["RootFS"]["Layers"][0] != conversion["base_diff_id"]:
            raise ValueError("Docker did not retain expected shared base")
        left = smoke(row["source"], output / f"original-{index}", script)
        right = smoke(row["target"], output / f"shared-{index}", script)
        result = {"source": row["source"], "target": row["target"], "config_equal": config_equal, "metadata_equal": left["metadata"] == right["metadata"], "public_testcases_equal": left["cases"] == right["cases"], "original": left, "shared": right, "layers": shared["RootFS"]["Layers"]}
        pairs.append(result)
        (output / "pairs.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
        print(json.dumps({k: result[k] for k in ("source", "config_equal", "metadata_equal", "public_testcases_equal")}), flush=True)
        if not all(result[k] for k in ("config_equal", "metadata_equal", "public_testcases_equal")) or left["exit_code"] != right["exit_code"]:
            raise ValueError("paired smoke mismatch")
    transfer = transfer_comparison(args.original_archive.resolve(), converted, output)
    result = {"load_seconds": load_s, "load_scope": "new shared images loaded into existing Docker cache; no cold original load comparison", "pairs": pairs, "transfer": transfer}
    (output / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"load_seconds": load_s, "transfer": transfer}, indent=2))


if __name__ == "__main__":
    main()
