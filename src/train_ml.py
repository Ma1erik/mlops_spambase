import hashlib
import json
from pathlib import Path
import sys
import time
import tracemalloc

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.config_schema import PipelineConfig
except ModuleNotFoundError:
    from config_schema import PipelineConfig  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "lab2_config.json"
REPORT_DIR = PROJECT_ROOT / "reports" / "LAB2"
METRICS_PATH = REPORT_DIR / "ml_metrics.csv"
MODEL_PATH = REPORT_DIR / "ml_model.joblib"


def compute_sha256(file_path: Path) -> str:
    """Вычисление хеша SHA-256 файла."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict:
    """Расчет метрик с 95% бутстрэп доверительными интервалами."""
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)

    boot_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
    }

    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        yt_sample = y_true[idx]
        if len(np.unique(yt_sample)) < 2:
            continue
        yp_sample = y_pred[idx]
        ypr_sample = y_prob[idx]

        boot_metrics["accuracy"].append(accuracy_score(yt_sample, yp_sample))
        boot_metrics["precision"].append(
            precision_score(yt_sample, yp_sample, zero_division=0)
        )
        boot_metrics["recall"].append(
            recall_score(yt_sample, yp_sample, zero_division=0)
        )
        boot_metrics["f1"].append(f1_score(yt_sample, yp_sample, zero_division=0))
        boot_metrics["roc_auc"].append(roc_auc_score(yt_sample, ypr_sample))

    result = {}
    for metric_name, values in boot_metrics.items():
        mean_val = float(np.mean(values))
        low_val = float(np.percentile(values, 2.5))
        high_val = float(np.percentile(values, 97.5))
        result[metric_name] = (mean_val, low_val, high_val)

    return result


def train_full_dataset(
    config: PipelineConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[Pipeline, float, float]:
    """Обучение в режиме полного датасета в памяти.

    Принимает ТОЛЬКО обучающую выборку. Никакого x_test!
    Возвращает единый обученный объект Pipeline.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SGDClassifier(
                    loss=config.loss_function,
                    alpha=config.alpha,
                    max_iter=config.max_iter_full,
                    random_state=config.random_seed,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    train_time = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_bytes / (1024 * 1024)

    return pipeline, train_time, peak_mem_mb


def train_chunked(
    config: PipelineConfig,
    train_csv_path: Path,
) -> tuple[Pipeline, float, float]:
    """Инкрементальное обучение чанками по потоку данных.

    Принимает ТОЛЬКО путь к обучающему потоку данных. Никакого x_test!
    Возвращает собранный обученный объект Pipeline.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    scaler = StandardScaler()
    for chunk in pd.read_csv(train_csv_path, chunksize=config.chunk_size):
        x_chunk = chunk.iloc[:, :-1].values
        scaler.partial_fit(x_chunk)

    model = SGDClassifier(
        loss=config.loss_function,
        alpha=config.alpha,
        random_state=config.random_seed,
    )
    classes = np.array([0, 1])

    for chunk in pd.read_csv(train_csv_path, chunksize=config.chunk_size):
        x_chunk = chunk.iloc[:, :-1].values
        y_chunk = chunk.iloc[:, -1].values
        x_chunk_scaled = scaler.transform(x_chunk)
        model.partial_fit(x_chunk_scaled, y_chunk, classes=classes)

    train_time = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_bytes / (1024 * 1024)

    # Собираем обученные скейлер и классификатор в единый Pipeline
    pipeline = Pipeline([("scaler", scaler), ("model", model)])
    return pipeline, train_time, peak_mem_mb


def evaluate_pipeline(
    pipeline: Pipeline,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> dict:
    """Оценка качества обученного пайплайна на тестовой выборке.

    Пайплайн сам применяет сохраненный scaler к x_test без утечек данных.
    """
    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]
    return compute_bootstrap_ci(y_test, y_pred, y_prob, seed=seed)


def run_pipeline() -> int:
    """Точка входа запуска полного конвейера ЛР 2."""
    if not CONFIG_PATH.exists():
        sys.stderr.write(f"[ERROR] Конфиг не найден: {CONFIG_PATH}\n")
        return 1

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
            raw_cfg = json.load(f)
        config = PipelineConfig(**raw_cfg)
    except Exception as exc:
        sys.stderr.write(f"[ERROR] Невалидный конфиг: {exc}\n")
        return 1

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    col_names = [f"f_{i}" for i in range(57)] + ["target"]
    df = pd.read_csv(config.data_path, header=None, names=col_names)

    x_all = df.iloc[:, :-1].values
    y_all = df.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=y_all,
    )

    # Временный поток данных содержит строго обучающую выборку
    temp_train_path = REPORT_DIR / "temp_train_stream.csv"
    train_df = pd.DataFrame(np.column_stack([x_train, y_train]), columns=col_names)
    train_df.to_csv(temp_train_path, index=False)

    # Наивный бейзлайн
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    dummy_pred = dummy.predict(x_test)
    dummy_acc = accuracy_score(y_test, dummy_pred)
    dummy_f1 = f1_score(y_test, dummy_pred, zero_division=0)

    mlflow.set_experiment("Lab2_ML_Spambase")
    rows_metrics = []

    # 1. Полноразмерный режим
    with mlflow.start_run(run_name="full_dataset_mode") as run_full:
        full_id = run_full.info.run_id
        mlflow.log_params(config.model_dump())
        mlflow.log_param("mode", "full_dataset")

        # Чистое обучение (без x_test) -> возвращает Pipeline
        full_pipeline, f_time, f_mem = train_full_dataset(config, x_train, y_train)

        # Оценка вынесена отдельно
        f_metrics = evaluate_pipeline(
            full_pipeline, x_test, y_test, seed=config.random_seed
        )

        for m_name, (val, ci_low, ci_high) in f_metrics.items():
            mlflow.log_metric(f"{m_name}_mean", val)
            mlflow.log_metric(f"{m_name}_ci_low", ci_low)
            mlflow.log_metric(f"{m_name}_ci_high", ci_high)
            rows_metrics.append(
                {
                    "mode": "full_dataset",
                    "run_id": full_id,
                    "metric": m_name,
                    "mean": round(val, 4),
                    "ci_low_95": round(ci_low, 4),
                    "ci_high_95": round(ci_high, 4),
                    "time_sec": round(f_time, 4),
                    "peak_ram_mb": round(f_mem, 4),
                }
            )

        mlflow.log_metric("time_seconds", f_time)
        mlflow.log_metric("peak_ram_mb", f_mem)

    # 2. Инкрементальный чанковый режим
    with mlflow.start_run(run_name="chunked_mode") as run_chunk:
        chunk_id = run_chunk.info.run_id
        mlflow.log_params(config.model_dump())
        mlflow.log_param("mode", "chunked")

        # Чистое обучение чанками (без x_test) -> возвращает Pipeline
        chunk_pipeline, c_time, c_mem = train_chunked(config, temp_train_path)

        # Оценка пайплайна
        c_metrics = evaluate_pipeline(
            chunk_pipeline, x_test, y_test, seed=config.random_seed
        )

        for m_name, (val, ci_low, ci_high) in c_metrics.items():
            mlflow.log_metric(f"{m_name}_mean", val)
            mlflow.log_metric(f"{m_name}_ci_low", ci_low)
            mlflow.log_metric(f"{m_name}_ci_high", ci_high)
            rows_metrics.append(
                {
                    "mode": "chunked",
                    "run_id": chunk_id,
                    "metric": m_name,
                    "mean": round(val, 4),
                    "ci_low_95": round(ci_low, 4),
                    "ci_high_95": round(ci_high, 4),
                    "time_sec": round(c_time, 4),
                    "peak_ram_mb": round(c_mem, 4),
                }
            )

        mlflow.log_metric("time_seconds", c_time)
        mlflow.log_metric("peak_ram_mb", c_mem)

    # Сохранение полного пайплайна (со скейлером внутри!)
    joblib.dump(full_pipeline, MODEL_PATH)
    model_sha256 = compute_sha256(MODEL_PATH)

    metrics_df = pd.DataFrame(rows_metrics)
    metrics_df.to_csv(METRICS_PATH, index=False)

    if temp_train_path.exists():
        temp_train_path.unlink()

    print("Результаты ЛР2")
    print(f"MLflow Run ID (Full):    {full_id}")
    print(f"MLflow Run ID (Chunked): {chunk_id}")
    print(f"Наивный бейзлайн (Dummy): Accuracy={dummy_acc:.4f}, F1={dummy_f1:.4f}")
    print(
        f"Full Mode:    ROC-AUC={f_metrics['roc_auc'][0]:.4f}, "
        f"Time={f_time:.3f}s, RAM={f_mem:.2f}MB"
    )
    print(
        f"Chunked Mode: ROC-AUC={c_metrics['roc_auc'][0]:.4f}, "
        f"Time={c_time:.3f}s, RAM={c_mem:.2f}MB"
    )
    print(f"Модель сохранена: {MODEL_PATH} (SHA-256: {model_sha256})")
    print(f"Метрики сохранены: {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())
