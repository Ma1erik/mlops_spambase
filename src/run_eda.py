import hashlib
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

RAW_DATA_PATH = Path("data/raw/spambase.csv")
MANIFEST_PATH = Path("reports/LAB1/hash_manifest.json")
STATS_PATH = Path("reports/LAB1/eda_stats.csv")
PLOT_PATH = Path("reports/LAB1/problem_distribution.png")


def compute_sha256(file_path: Path) -> str:
    """Вычисление хеша SHA-256 файла."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_integrity(
    data_path: Path = RAW_DATA_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> tuple[bool, str]:
    """Проверка целостности сырых данных против манифеста."""
    if not data_path.exists():
        return False, f"Файл данных не найден: {data_path}"
    if not manifest_path.exists():
        return False, f"Манифест хешей не найден: {manifest_path}"

    try:
        content = manifest_path.read_text(encoding="utf-8")
        match = re.search(r"\b[a-fA-F0-9]{64}\b", content)
        if not match:
            return False, "В манифесте не найден 64-значный SHA-256 хеш."
        expected_hash = match.group(0).lower()
    except Exception as exc:
        return False, f"Ошибка чтения манифеста: {exc}"

    actual_hash = compute_sha256(data_path).lower()

    if actual_hash != expected_hash:
        err_msg = (
            f"Несовпадение SHA-256!\n"
            f"Ожидался: {expected_hash}\n"
            f"Получен:  {actual_hash}"
        )
        return False, err_msg

    return True, "Контрольная сумма SHA-256 совпадает с манифестом."


def run_eda(data_path: Path = RAW_DATA_PATH) -> dict:
    """Выполнение EDA и сохранение артефактов."""
    col_names = [f"f_{i}" for i in range(57)] + ["target"]
    df = pd.read_csv(data_path, header=None, names=col_names)

    total_rows = len(df)
    total_cols = len(df.columns)
    missing_count = int(df.isnull().sum().sum())

    target_counts = df["target"].value_counts(normalize=True) * 100
    ham_pct = float(target_counts.get(0, 0.0))
    spam_pct = float(target_counts.get(1, 0.0))

    feature_cols = [c for c in df.columns if c != "target"]
    skews = df[feature_cols].skew()
    max_skew_col = skews.abs().idxmax()
    max_skew_val = float(skews[max_skew_col])

    stats_df = df.describe().T
    stats_df["skewness"] = df.skew()
    stats_df.to_csv(STATS_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    classes = ["Не-спам (0)", "Спам (1)"]
    shares = [ham_pct, spam_pct]
    bars = axes[0].bar(classes, shares, color=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Баланс классов целевой переменной")
    axes[0].set_ylabel("Доля, %")
    axes[0].set_ylim(0, 100)
    for bar, val in zip(bars, shares):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.2f}%",
            ha="center",
            fontweight="bold",
        )

    f55 = df["f_55"]
    axes[1].hist(
        f55,
        bins=50,
        color="#3498db",
        edgecolor="black",
        log=True,
    )
    axes[1].set_title("Распределение f_55 (длина цепочек заглавных букв)")
    axes[1].set_xlabel("Значение признака")
    axes[1].set_ylabel("Количество (log-шкала)")
    axes[1].axvline(
        f55.mean(),
        color="red",
        linestyle="--",
        label=f"Среднее: {f55.mean():.2f}",
    )
    axes[1].axvline(
        f55.median(),
        color="orange",
        linestyle=":",
        label=f"Медиана: {f55.median():.2f}",
    )
    axes[1].legend()

    plt.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()

    print("=== Результаты первичного анализа (EDA) ===")
    print(f"Объем выборки: {total_rows} строк, {total_cols} признаков")
    print(f"Пропуски в данных: {missing_count}")
    print(f"Баланс классов: не-спам = {ham_pct:.2f}%, спам = {spam_pct:.2f}%")
    print(f"Максимальная асимметрия: {max_skew_col} (skew = {max_skew_val:.2f})")
    print(f"График сохранен: {PLOT_PATH}")
    print(f"Статистика сохранена: {STATS_PATH}")

    return {
        "rows": total_rows,
        "cols": total_cols,
        "missing": missing_count,
        "ham_pct": ham_pct,
        "spam_pct": spam_pct,
        "max_skew": max_skew_val,
    }


def main() -> int:
    """Точка входа: валидация, обработка ошибок и код возврата."""
    try:
        is_valid, message = verify_integrity()
        if not is_valid:
            sys.stderr.write(f"[ERROR] Нарушение целостности данных:\n{message}\n")
            return 1

        run_eda()
        print("[INFO] Первичный анализ данных (EDA) успешно завершен.")
        return 0

    except Exception as exc:
        sys.stderr.write(f"[ERROR] Непредвиденный сбой выполнения: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
