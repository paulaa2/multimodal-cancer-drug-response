"""Download frozen public inputs with verified checksums and atomic writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(source: dict[str, Any]) -> dict[str, Any]:
    """Reuse only verified files; incomplete downloads never become final inputs."""

    path = Path(source["path"])
    algorithm, expected = source["checksum"].split(":", 1)
    cached = path.exists() and digest_file(path, algorithm) == expected
    if not cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(source["url"], headers={"User-Agent": "mcdrp/0.1"})
        with (
            urllib.request.urlopen(request, timeout=90) as response,
            temporary.open("wb") as out,
        ):
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                out.write(chunk)
        actual = digest_file(temporary, algorithm)
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {path}: expected {expected}, got {actual}."
            )
        temporary.replace(path)
    return {
        **source,
        "bytes": path.stat().st_size,
        "sha256": digest_file(path, "sha256"),
        "cached": cached,
    }


def download_sources(config_path: str, summary: str, workers: int = 2) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        sources = list(pool.map(download_source, config["sources"]))
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources}
    Path(summary).parent.mkdir(parents=True, exist_ok=True)
    Path(summary).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/public_sources.json")
    parser.add_argument("--summary", default="data/reports/public_sources_summary.json")
    parser.add_argument("--workers", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = download_sources(args.config, args.summary, args.workers)
    print(json.dumps([{"path": x["path"], "bytes": x["bytes"]} for x in report["sources"]]))


if __name__ == "__main__":
    main()
