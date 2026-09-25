#!/usr/bin/env python3
"""Download catalogued public products atomically, retaining source checksums."""

import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    manifest.update(catalog=str(args.catalog.resolve()), catalog_sha256=sha256(args.catalog))
    entries = [("readme", catalog["readme_url"])]
    for region, row in catalog["regions"].items():
        entries.extend((f"{region}/{kind}", url) for kind, url in row["urls"].items())
    for key, url in entries:
        target = args.output / (key + (".txt" if key == "readme" else ".tif"))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            old = manifest["files"].get(key)
            if not old or old["url"] != url or old["sha256"] != sha256(target):
                raise ValueError(f"Existing product lacks matching manifest: {target}")
        else:
            partial = target.with_suffix(target.suffix + ".partial")
            subprocess.run(
                [
                    "curl",
                    "-fL",
                    "--retry",
                    "4",
                    "--retry-all-errors",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    "900",
                    url,
                    "-o",
                    str(partial),
                ],
                check=True,
            )
            partial.replace(target)
        previous = manifest["files"].get(key, {})
        manifest["files"][key] = {
            "path": str(target.resolve()),
            "url": url,
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "downloaded_at": previous.get(
                "downloaded_at", datetime.datetime.now(datetime.UTC).isoformat()
            ),
        }
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(manifest_path)
        print(key, target.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
