import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "spambase/spambase.data"
)
OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "spambase.csv"
EXPECTED_SHA256 = "b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12"


def compute_sha256(file_path: Path) -> str:
    """Вычисление хеша SHA-256 файла."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_dataset(
    url: str = DEFAULT_URL,
    target_path: Path = OUTPUT_FILE,
) -> int:
    """Загрузка данных, валидация хеша и программный возврат кода (0 или 1)."""
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Загрузка сырых данных: {url}")

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            data = response.read()

        with open(target_path, "wb") as out_f:
            out_f.write(data)

        # Валидация контрольной суммы
        actual_hash = compute_sha256(target_path)
        if actual_hash.lower() != EXPECTED_SHA256.lower():
            sys.stderr.write(
                f"[ERROR] Нарушение целостности скачанного файла!\n"
                f"Ожидался SHA-256: {EXPECTED_SHA256}\n"
                f"Получен SHA-256:  {actual_hash}\n"
            )
            return 1

        print(f"[INFO] Данные успешно сохранены в: {target_path}")
        print(f"[INFO] Хеш SHA-256 совпадает: {actual_hash}")
        return 0

    except urllib.error.HTTPError as exc:
        sys.stderr.write(
            f"[ERROR] HTTP ошибка при скачивании: {exc.code} {exc.reason}\n"
        )
        return 1
    except urllib.error.URLError as exc:
        sys.stderr.write(f"[ERROR] Сетевая ошибка подключения: {exc.reason}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"[ERROR] Непредвиденный сбой загрузки: {exc}\n")
        return 1


def main() -> int:
    """Точка входа CLI."""
    parser = argparse.ArgumentParser(description="Загрузка данных Spambase.")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help="URL источника датасета",
    )
    args = parser.parse_args()
    return download_dataset(url=args.url)


if __name__ == "__main__":
    sys.exit(main())
