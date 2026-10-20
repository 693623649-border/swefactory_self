"""Fetch the Scale-SWE parquet shards listed on the Hub without cloning the repo.

Downloads only *.parquet files under the default config into
data/scaleswe_raw/, streaming to disk so memory stays flat. The dataset is
20,181 rows / ~271.8 MB of parquet (verified via datasets-server /size), so this
is metadata-scale storage, not the per-instance Docker images.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "AweAI-Team/Scale-SWE"
OUT = Path(__file__).resolve().parents[1] / "data" / "scaleswe_raw"
UA = {"User-Agent": "swefactory-incremental-fetch/1.0"}


def api(path: str, **params) -> dict:
    base = path if path.startswith("http") else f"https://huggingface.co/{path}"
    url = base + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def list_parquet() -> list[str]:
    page = api("https://datasets-server.huggingface.co/parquet", dataset=DATASET)
    return [f["url"] for f in page.get("parquet_files", [])]


def download(url: str) -> int:
    dest = OUT / url.rsplit("/", 1)[-1]
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r, dest.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest.stat().st_size


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    urls = list_parquet()
    print(f"parquet shards: {len(urls)}")
    manifest = {}
    total = 0
    for url in urls:
        size = download(url)
        manifest[url.rsplit("/", 1)[-1]] = size
        total += size
        print(f"  {url.rsplit('/', 1)[-1]}: {size/1e6:.1f} MB")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"total {total/1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
