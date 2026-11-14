import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


roots = [
    Path("pyproject.toml"),
    Path("data/raw/spambase.csv"),
    Path("data/synthetic/spambase_synthetic.csv"),
    Path("configs"),
    Path("src"),
    Path("scripts"),
    Path("reports/LAB1"),
    Path("reports/LAB2"),
    Path("reports/LAB3"),
    Path("reports/LAB4"),
    Path("reports/LAB5"),
    Path("reports/LAB6"),
    Path("reports/LAB7"),
    Path("reports/LAB8"),
]

entries = []
for root in roots:
    if not root.exists():
        continue
    files = (
        [root]
        if root.is_file()
        else sorted(
            p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        )
    )
    for fpath in files:
        if fpath.name == "artifacts_manifest.json":
            continue
        entries.append(
            {
                "path": fpath.as_posix(),
                "size_bytes": fpath.stat().st_size,
                "sha256": sha256_file(fpath),
            }
        )

manifest = {
    "variant": 45,
    "dataset": "UCI Spambase",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "total_files": len(entries),
    "files": entries,
}

out_path = Path("reports/LAB8/artifacts_manifest.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"[OK] Manifest generated: {out_path} ({len(entries)} files)")
