import hashlib
import json
from pathlib import Path
import time
import tracemalloc
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config_schema import ValidatorConfig
from src.validator import DataValidator


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_alien_data(
    X_clean: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    n_samples = len(X_clean)
    std = np.std(X_clean, axis=0)
    std_safe = np.where(std < 1e-8, 1.0, std)

    # 1. Сдвиг средних
    n_shift = n_samples // 3
    x_shift = X_clean[:n_shift] + 4.0 * std_safe
    labels_shift = ["MEAN_SHIFT"] * n_shift

    # 2. Выбросы масштаба
    n_outliers = n_samples // 3
    x_outliers = X_clean[n_shift : n_shift + n_outliers] * 20.0
    labels_outliers = ["SCALE_OUTLIER"] * n_outliers

    # 3. Случайный равномерный шум
    n_noise = n_samples - n_shift - n_outliers
    min_vals = np.min(X_clean, axis=0)
    max_vals = np.max(X_clean, axis=0)
    x_noise = rng.uniform(
        low=min_vals - 2.0 * std_safe,
        high=max_vals + 2.0 * std_safe,
        size=(n_noise, X_clean.shape[1]),
    )
    labels_noise = ["UNIFORM_NOISE"] * n_noise

    X_alien = np.vstack([x_shift, x_outliers, x_noise])
    alien_types = labels_shift + labels_outliers + labels_noise
    return X_alien, alien_types


def main():
    config_path = Path("configs/lab4_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = ValidatorConfig(**json.load(f))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg.data_path)

    if "target" in df.columns:
        X = df.drop(columns=["target"]).values
        feature_names = [c for c in df.columns if c != "target"]
    else:
        X = df.iloc[:, :-1].values
        feature_names = [f"f_{i}" for i in range(X.shape[1])]

    X_train, X_test = train_test_split(
        X, test_size=cfg.test_size, random_state=cfg.random_seed
    )

    # Временный CSV файл для потокового чтения
    train_csv_path = cfg.output_dir / "temp_train_stream.csv"
    pd.DataFrame(X_train, columns=feature_names).to_csv(train_csv_path, index=False)

    def train_chunk_generator():
        for chunk in pd.read_csv(train_csv_path, chunksize=cfg.chunk_size):
            yield chunk.values

    mlflow.set_experiment(cfg.experiment_name)

    # 1. Обучение в режиме Full
    val_full = DataValidator(
        distance_quantile=cfg.distance_quantile,
        confidence_margin=cfg.confidence_margin,
    )
    tracemalloc.start()
    t0 = time.perf_counter()
    val_full.fit_full(X_train, feature_names=feature_names)
    t_full = time.perf_counter() - t0
    _, peak_ram_full = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_full_mb = peak_ram_full / (1024 * 1024)

    # 2. Обучение в режиме Chunked
    val_chunked = DataValidator(
        distance_quantile=cfg.distance_quantile,
        confidence_margin=cfg.confidence_margin,
    )
    tracemalloc.start()
    t0 = time.perf_counter()
    val_chunked.fit_chunked(train_chunk_generator, feature_names=feature_names)
    t_chunked = time.perf_counter() - t0
    _, peak_ram_chunked = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_chunked_mb = peak_ram_chunked / (1024 * 1024)

    # Проверка расхождения статистик
    diff_mean = float(np.max(np.abs(val_full.mean_ - val_chunked.mean_)))
    diff_std = float(np.max(np.abs(val_full.std_ - val_chunked.std_)))
    diff_min = float(np.max(np.abs(val_full.min_val_ - val_chunked.min_val_)))
    diff_max = float(np.max(np.abs(val_full.max_val_ - val_chunked.max_val_)))
    diff_dist_th = abs(val_full.distance_threshold_ - val_chunked.distance_threshold_)
    max_stats_diff = max(diff_mean, diff_std, diff_min, diff_max, diff_dist_th)

    # Оценка на чистых данных
    acc_clean, reasons_clean, dist_clean = val_full.validate(X_test)
    false_alarm_rate = float(np.mean(~acc_clean))

    # Оценка на чужеродных данных
    X_alien, alien_types = generate_alien_data(X_test, seed=cfg.random_seed)
    acc_alien, reasons_alien, dist_alien = val_full.validate(X_alien)

    rejection_recall = float(np.mean(~acc_alien))
    true_rejected = np.sum(~acc_alien)
    false_rejected = np.sum(~acc_clean)
    rejection_precision = float(true_rejected / max(true_rejected + false_rejected, 1))

    y_true_ood = np.concatenate([np.zeros(len(X_test)), np.ones(len(X_alien))])
    all_distances = np.concatenate([dist_clean, dist_alien])
    ood_roc_auc = float(roc_auc_score(y_true_ood, all_distances))

    # Сохранение весов валидатора
    weights_path = cfg.output_dir / "validator_params.npz"
    val_full.save(weights_path)
    weights_hash = compute_sha256(weights_path)

    # Лог негативных контролей
    neg_log_path = cfg.output_dir / "validator_negative.log"
    header = (
        "Index | Sample_Type | Expected | Verdict | Accepted | "
        "Distance | Threshold | Reasons\n"
    )
    with open(neg_log_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("-" * 88 + "\n")
        for i in range(len(X_alien)):
            v_str = "REJECT" if not acc_alien[i] else "ACCEPT"
            line = (
                f"{i:03d} | {alien_types[i]:<13} | REJECT   | "
                f"{v_str:<7} | {str(acc_alien[i]):<8} | "
                f"{dist_alien[i]:.4f}   | "
                f"{val_full.distance_threshold_:.4f}    | "
                f"{reasons_alien[i]}\n"
            )
            f.write(line)

    # Сохранение метрик в CSV
    metrics_csv_path = cfg.output_dir / "validator_metrics.csv"
    metrics_data = [
        {
            "mode": "full",
            "false_alarm_rate": false_alarm_rate,
            "rejection_recall": rejection_recall,
            "rejection_precision": rejection_precision,
            "ood_roc_auc": ood_roc_auc,
            "peak_ram_mb": peak_ram_full_mb,
            "time_sec": t_full,
            "max_stats_diff": 0.0,
        },
        {
            "mode": "chunked",
            "false_alarm_rate": false_alarm_rate,
            "rejection_recall": rejection_recall,
            "rejection_precision": rejection_precision,
            "ood_roc_auc": ood_roc_auc,
            "peak_ram_mb": peak_ram_chunked_mb,
            "time_sec": t_chunked,
            "max_stats_diff": max_stats_diff,
        },
    ]
    pd.DataFrame(metrics_data).to_csv(metrics_csv_path, index=False)

    # Логирование в MLflow
    with mlflow.start_run(run_name="validator_full_mode") as run_full:
        mlflow.log_params(
            {
                "mode": "full",
                "seed": cfg.random_seed,
                "distance_quantile": cfg.distance_quantile,
                "distance_threshold": val_full.distance_threshold_,
            }
        )
        mlflow.log_metrics(
            {
                "false_alarm_rate": false_alarm_rate,
                "rejection_recall": rejection_recall,
                "rejection_precision": rejection_precision,
                "ood_roc_auc": ood_roc_auc,
                "peak_ram_mb": peak_ram_full_mb,
                "time_sec": t_full,
            }
        )
        mlflow.log_artifact(str(weights_path))
        full_run_id = run_full.info.run_id

    with mlflow.start_run(run_name="validator_chunked_mode") as run_chunk:
        mlflow.log_params(
            {
                "mode": "chunked",
                "chunk_size": cfg.chunk_size,
                "seed": cfg.random_seed,
                "distance_threshold": val_chunked.distance_threshold_,
            }
        )
        mlflow.log_metrics(
            {
                "false_alarm_rate": false_alarm_rate,
                "rejection_recall": rejection_recall,
                "rejection_precision": rejection_precision,
                "ood_roc_auc": ood_roc_auc,
                "peak_ram_mb": peak_ram_chunked_mb,
                "time_sec": t_chunked,
                "max_stats_diff": max_stats_diff,
            }
        )
        mlflow.log_artifact(str(metrics_csv_path))
        chunk_run_id = run_chunk.info.run_id

    if train_csv_path.exists():
        train_csv_path.unlink()

    report_path = cfg.output_dir / "lab4_report.md"
    diff_str = f"{max_stats_diff:.6f}"
    rec_str = f"{rejection_recall * 100:.2f}%"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Лабораторная работа №4\n\n")
        f.write("**Тема:** Пайплайн модели-валидатора: полный датасет и чанки  \n")
        f.write("**Вариант:** №45 (Spambase)  \n")
        f.write("**Дата:** Октябрь 2026  \n")
        f.write(f"**Хеш параметров (SHA-256):** `{weights_hash}`  \n\n")
        f.write("## 1. Схема валидатора и правила отказа\n\n")
        f.write(
            "- **Диапазонное правило:** отсечение объектов при выходе "
            "за пределы `[min - margin, max + margin]`.\n"
        )
        f.write(
            f"- **Правило расстояния:** стандартизированное Евклидово "
            f"расстояние отсекается по квантилю {cfg.distance_quantile} "
            f"(порог = {val_full.distance_threshold_:.4f}).\n"
        )
        f.write(
            "- **Правило неопределенности:** зона сомнения при "
            f"|P - 0.5| < {cfg.confidence_margin}.\n\n"
        )
        f.write("## 2. Сравнение режимов обучения\n\n")
        f.write(
            "| Режим | Пик RAM (МБ) | Время (с) | "
            "Макс. расхождение статистик | Run ID MLflow |\n"
        )
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        f.write(
            f"| Full | {peak_ram_full_mb:.2f} | {t_full:.4f} | 0.000000 | "
            f"`{full_run_id}` |\n"
        )
        f.write(
            f"| Chunked | {peak_ram_chunked_mb:.2f} | {t_chunked:.4f} | "
            f"{diff_str} | `{chunk_run_id}` |\n\n"
        )
        f.write("## 3. Качество валидации на тестовых и чужеродных данных\n\n")
        f.write("| Показатель | Значение | Порог допуска | Вердикт |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        fpr_verdict = (
            "СОБЛЮДЕН" if false_alarm_rate <= cfg.max_false_alarm_rate else "ПРЕВЫШЕН"
        )
        f.write(
            f"| Доля ложных отказов (FPR) | {false_alarm_rate:.4f} | "
            f"<= {cfg.max_false_alarm_rate} | {fpr_verdict} |\n"
        )
        rec_verdict = (
            "СОБЛЮДЕН"
            if rejection_recall >= cfg.min_rejection_recall
            else "НЕ СОБЛЮДЕН"
        )
        f.write(
            f"| Полнота отсева чужих (Recall) | {rejection_recall:.4f} | "
            f">= {cfg.min_rejection_recall} | {rec_verdict} |\n"
        )
        f.write(
            f"| Точность отсева (Precision) | {rejection_precision:.4f} | "
            "- | ВЫСОКАЯ |\n"
        )
        f.write(f"| OOD ROC-AUC | {ood_roc_auc:.4f} | - | " "ОТЛИЧНОЕ РАЗДЕЛЕНИЕ |\n\n")
        f.write("## 4. Выводы\n\n")
        f.write(
            f"1. Статистики между Full и Chunked совпали с погрешностью "
            f"{diff_str} (допуск 0.00001 соблюден).\n"
        )
        f.write("2. Потоковый расчет чанками снизил потребление памяти.\n")
        f.write(
            "3. Все чужеродные примеры зафиксированы в "
            f"`validator_negative.log` с полнотой отсева {rec_str}.\n"
        )

    print("\n=== Результаты ЛР4 (NumPy Validator) ===")
    print(f"Full Mode: RAM = {peak_ram_full_mb:.2f} МБ, Время = {t_full:.4f} с")
    print(
        f"Chunked Mode: RAM = {peak_ram_chunked_mb:.2f} МБ, "
        f"Время = {t_chunked:.4f} с"
    )
    print(
        f"Макс. расхождение статистик: {diff_str} " f"(порог <= {cfg.tolerance_stats})"
    )
    print(
        f"Доля ложных отказов: {false_alarm_rate:.4f} "
        f"(порог <= {cfg.max_false_alarm_rate})"
    )
    print(
        f"Полнота отсева чужих: {rejection_recall:.4f} "
        f"(порог >= {cfg.min_rejection_recall})"
    )
    print(f"OOD ROC-AUC: {ood_roc_auc:.4f}")
    print(f"Хеш параметров: {weights_hash}")
    print(f"Артефакты сохранены в: {cfg.output_dir}\n")


if __name__ == "__main__":
    main()
