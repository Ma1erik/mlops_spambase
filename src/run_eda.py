import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = Path("data/raw/spambase.csv")
REPORT_DIR = Path("reports/LAB1")
STATS_PATH = REPORT_DIR / "eda_stats.csv"
PLOT_PATH = REPORT_DIR / "problem_distribution.png"
MANIFEST_PATH = REPORT_DIR / "hash_manifest.json"

COLUMNS = [f"f_{i}" for i in range(57)] + ["target"]


def verify_hash() -> bool:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    expected_hash = manifest["files"][0]["sha256"]

    hasher = hashlib.sha256()
    with open(DATA_PATH, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest()
    return actual_hash == expected_hash


def main() -> None:
    if not verify_hash():
        err_msg = "Ошибка: хеш сырых данных не совпадает с манифестом!"
        raise ValueError(err_msg)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)

    desc = df.describe().T
    desc["missing_count"] = df.isnull().sum()
    desc["dtype"] = [str(t) for t in df.dtypes]
    desc.to_csv(STATS_PATH, index_label="feature_name")

    total_rows = len(df)
    counts = df["target"].value_counts()
    ham_count = counts.get(0, 0)
    spam_count = counts.get(1, 0)
    ham_pct = (ham_count / total_rows) * 100
    spam_pct = (spam_count / total_rows) * 100

    max_run_max = df["f_55"].max()
    skew_max = df.skew().max()

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["0: Ham", "1: Spam"]
    bars = axes[0].bar(
        labels,
        [ham_count, spam_count],
        color=["#27ae60", "#e74c3c"],
    )
    axes[0].set_title("Дисбаланс классов Spambase", fontsize=12)
    axes[0].set_ylabel("Количество писем", fontsize=10)
    bar_txt = [
        f"{ham_count} ({ham_pct:.1f}%)",
        f"{spam_count} ({spam_pct:.1f}%)",
    ]
    axes[0].bar_label(bars, labels=bar_txt, padding=3)

    sns.boxplot(
        ax=axes[1],
        x="target",
        y="f_55",
        hue="target",
        data=df,
        palette=["#27ae60", "#e74c3c"],
        legend=False,
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Выбросы: цепочки заглавных букв (f_55)", fontsize=12)
    axes[1].set_xlabel("Класс", fontsize=10)
    axes[1].set_ylabel("Длина (лог. шкала)", fontsize=10)

    stat_box = (
        f"Макс. цепочка: {int(max_run_max)}\n" f"Макс. асимметрия: {skew_max:.1f}"
    )
    axes[1].text(
        0.5,
        0.9,
        stat_box,
        ha="center",
        transform=axes[1].transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    print("=== Результаты EDA Spambase ===")
    print(f"Всего строк: {total_rows}")
    print(f"Всего колонок: {df.shape[1]}")
    print(f"Пропусков: {df.isnull().sum().sum()}")
    print(f"Класс 0 (Ham): {ham_count} ({ham_pct:.2f}%)")
    print(f"Класс 1 (Spam): {spam_count} ({spam_pct:.2f}%)")
    print(f"Максимальная асимметрия: {skew_max:.2f}")
    print(f"EDA-статистики записаны в: {STATS_PATH}")
    print(f"График сохранен в: {PLOT_PATH}")


if __name__ == "__main__":
    main()
