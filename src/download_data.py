import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

# Официальный архив датасета из репозитория UCI
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/94/spambase.zip"
RAW_DIR = Path("data/raw")
DATA_FILE = RAW_DIR / "spambase.csv"
MANIFEST_PATH = Path("reports/LAB1/hash_manifest.json")


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Загрузка архива с UCI...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    request = urllib.request.Request(UCI_ZIP_URL, headers=headers)

    with urllib.request.urlopen(request, timeout=30) as response:
        zip_bytes = response.read()

    print("Извлечение spambase.data из архива...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        # Внутри архива UCI лежит сырой файл spambase.data без заголовков
        data_bytes = z.read("spambase.data")
        DATA_FILE.write_bytes(data_bytes)

    sha256_hash = compute_sha256(DATA_FILE)
    size_bytes = DATA_FILE.stat().st_size

    manifest = {
        "dataset_name": "UCI Spambase",
        "source_url": UCI_ZIP_URL,
        "files": [
            {
                "path": str(DATA_FILE).replace("\\", "/"),
                "sha256": sha256_hash,
                "size_bytes": size_bytes,
            }
        ],
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Готово! Данные сохранены в: {DATA_FILE}")
    print(f"Размер: {size_bytes} байт")
    print(f"SHA-256: {sha256_hash}")
    print(f"Манифест записан в: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
