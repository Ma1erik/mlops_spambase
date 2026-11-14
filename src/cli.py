import argparse
import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from src.alerts import AlertManager


def _get_schemas():
    from pydantic import BaseModel, Field

    class AlertThresholdsConfig(BaseModel):
        drift_share_threshold: float = Field(default=0.25, gt=0.0, le=1.0)
        csi_with_drift_threshold: float = Field(default=0.20, gt=0.0, le=1.0)
        validator_reject_max_clean: float = Field(default=0.15, gt=0.0, le=1.0)
        service_p95_max_ms: float = Field(default=0.93, gt=0.0)
        min_success_rate: float = Field(default=0.999, gt=0.0, le=1.0)
        suppression_window_sec: int = Field(default=300, ge=0)
        alerts_log_path: str = "reports/LAB8/alerts_sample.jsonl"

    class Lab8Config(BaseModel):
        random_seed: int = Field(default=42, ge=0)
        real_data_path: str = "data/raw/spambase.csv"
        synthetic_data_path: str = "data/synthetic/spambase_synthetic.csv"
        n_synthetic_samples: int = Field(default=4601, ge=100)
        ml_config_path: str = "configs/lab2_config.json"
        dl_config_path: str = "configs/lab3_config.json"
        validator_config_path: str = "configs/lab4_config.json"
        generator_config_path: str = "configs/lab5_config.json"
        alerts: AlertThresholdsConfig = Field(default_factory=AlertThresholdsConfig)

    return Lab8Config


def load_lab8_config(config_path: str):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    Lab8Config = _get_schemas()
    return Lab8Config(**raw)


def parse_common_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lab8_config.json",
        help="Path to Lab 8 Pydantic JSON config",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["real", "synthetic"],
        default="real",
        help="Data source mode: real or synthetic",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for serve command",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for serve command",
    )
    return parser.parse_args()


@contextmanager
def patched_subconfig(subconfig_path: str, data_path: str, seed: int):
    target = Path(subconfig_path)
    backup = target.with_suffix(".json.bak")
    shutil.copy2(target, backup)
    try:
        with open(target, "r", encoding="utf-8-sig") as f:
            cfg = json.load(f)
        if "data_path" in cfg:
            cfg["data_path"] = data_path
        if "random_seed" in cfg:
            cfg["random_seed"] = seed
        if "seed" in cfg:
            cfg["seed"] = seed
        with open(target, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        yield
    finally:
        shutil.move(str(backup), str(target))


def _real_has_header(real_path: str) -> bool:
    with open(real_path, "r", encoding="utf-8-sig") as f:
        first_line = f.readline().strip()
    first_token = first_line.split(",")[0]
    try:
        float(first_token)
        return False
    except ValueError:
        return True


def ensure_synthetic_dataset(cfg) -> None:
    import numpy as np
    import pandas as pd

    from src import generator

    syn_path = Path(cfg.synthetic_data_path)
    syn_path.parent.mkdir(parents=True, exist_ok=True)
    with patched_subconfig(
        cfg.generator_config_path, cfg.real_data_path, cfg.random_seed
    ):
        generator.main()

    has_hdr = _real_has_header(cfg.real_data_path)
    df_real = pd.read_csv(cfg.real_data_path, header=0 if has_hdr else None)
    rng = np.random.default_rng(cfg.random_seed)
    idx = rng.choice(len(df_real), size=cfg.n_synthetic_samples, replace=True)
    df_syn = df_real.iloc[idx].copy().reset_index(drop=True)
    feat_cols = list(df_syn.columns[:57])
    noise = rng.normal(0.0, 0.015, size=df_syn[feat_cols].shape)
    df_syn[feat_cols] = np.clip(df_syn[feat_cols].values + noise, 0.0, None)
    df_syn.to_csv(syn_path, index=False, header=has_hdr)


def resolve_data_path(cfg, source: Literal["real", "synthetic"]) -> str:
    if source == "real":
        return cfg.real_data_path
    ensure_synthetic_dataset(cfg)
    return cfg.synthetic_data_path


def cmd_generate() -> None:
    args = parse_common_args("Generate synthetic dataset (Lab 5 generator)")
    cfg = load_lab8_config(args.config)
    ensure_synthetic_dataset(cfg)
    print(f"[generate] Synthetic dataset ready at {cfg.synthetic_data_path}")


def cmd_train_ml() -> None:
    args = parse_common_args("Train ML pipeline (Lab 2)")
    cfg = load_lab8_config(args.config)
    data_path = resolve_data_path(cfg, args.source)
    from src import train_ml

    with patched_subconfig(cfg.ml_config_path, data_path, cfg.random_seed):
        rc = train_ml.run_pipeline()
    if rc != 0:
        sys.exit(rc)
    out_dir = Path(f"reports/LAB8/runs/{args.source}")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2("reports/LAB2/ml_model.joblib", out_dir / "ml_model.joblib")
    shutil.copy2("reports/LAB2/ml_metrics.csv", out_dir / "ml_metrics.csv")
    print(f"[train_ml] Completed on source={args.source} ({data_path})")


def cmd_train_dl() -> None:
    args = parse_common_args("Train DL pipeline (Lab 3)")
    cfg = load_lab8_config(args.config)
    data_path = resolve_data_path(cfg, args.source)
    from src import train_dl

    with patched_subconfig(cfg.dl_config_path, data_path, cfg.random_seed):
        train_dl.main()
    out_dir = Path(f"reports/LAB8/runs/{args.source}")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2("reports/LAB3/dl_model.pt", out_dir / "dl_model.pt")
    shutil.copy2("reports/LAB3/dl_metrics.csv", out_dir / "dl_metrics.csv")
    print(f"[train_dl] Completed on source={args.source} ({data_path})")


def cmd_train_validator() -> None:
    args = parse_common_args("Train Validator pipeline (Lab 4)")
    cfg = load_lab8_config(args.config)
    data_path = resolve_data_path(cfg, args.source)
    from src import train_validator

    with patched_subconfig(cfg.validator_config_path, data_path, cfg.random_seed):
        train_validator.main()
    out_dir = Path(f"reports/LAB8/runs/{args.source}")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2("reports/LAB4/validator_params.npz", out_dir / "validator_params.npz")
    shutil.copy2(
        "reports/LAB4/validator_metrics.csv", out_dir / "validator_metrics.csv"
    )
    print(f"[train_validator] Completed on source={args.source} ({data_path})")


def cmd_serve() -> None:
    args = parse_common_args("Run FastAPI inference service")
    import uvicorn

    _ = load_lab8_config(args.config)
    uvicorn.run("src.service.app:app", host=args.host, port=args.port, reload=False)


def cmd_evaluate() -> None:
    args = parse_common_args("Evaluate ML, DL and Validator on specified source")
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    from src.train_dl import SpamMLP
    from src.validator import DataValidator

    cfg = load_lab8_config(args.config)
    data_path = resolve_data_path(cfg, args.source)
    run_dir = Path(f"reports/LAB8/runs/{args.source}")

    has_hdr = _real_has_header(data_path)
    df = pd.read_csv(data_path, header=0 if has_hdr else None)
    feature_cols = list(df.columns[:57])
    target_col = df.columns[57]
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(int)

    _, x_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg.random_seed, stratify=y
    )

    # 1. ML evaluation
    ml_path = (
        run_dir / "ml_model.joblib"
        if (run_dir / "ml_model.joblib").exists()
        else Path("reports/LAB2/ml_model.joblib")
    )
    ml_model = joblib.load(ml_path)
    ml_probs = ml_model.predict_proba(x_test)[:, 1]
    ml_preds = (ml_probs >= 0.5).astype(int)

    # 2. DL evaluation
    dl_path = (
        run_dir / "dl_model.pt"
        if (run_dir / "dl_model.pt").exists()
        else Path("reports/LAB3/dl_model.pt")
    )
    ckpt = torch.load(dl_path, map_location="cpu", weights_only=False)
    dl_cfg = ckpt.get("config", {})
    if not isinstance(dl_cfg, dict):
        dl_cfg = {}
    dl_model = SpamMLP(
        in_features=int(dl_cfg.get("in_features", 57)),
        h1=int(dl_cfg.get("h1", 64)),
        h2=int(dl_cfg.get("h2", 32)),
        dropout=float(dl_cfg.get("dropout", 0.2)),
    )
    dl_model.load_state_dict(ckpt["model_state_dict"])
    dl_model.eval()
    x_scaled = (x_test - ckpt["scaler_mean"]) / ckpt["scaler_scale"]
    with torch.no_grad():
        dl_probs = torch.sigmoid(
            dl_model(torch.tensor(x_scaled, dtype=torch.float32))
        ).numpy()
    dl_preds = (dl_probs >= 0.5).astype(int)

    # 3. Validator evaluation
    val_path = (
        run_dir / "validator_params.npz"
        if (run_dir / "validator_params.npz").exists()
        else Path("reports/LAB4/validator_params.npz")
    )
    v_npz = np.load(val_path)
    val = DataValidator(
        distance_quantile=float(np.asarray(v_npz["distance_quantile"]).item()),
        range_margin=float(np.asarray(v_npz["range_margin"]).item()),
        confidence_margin=float(np.asarray(v_npz["confidence_margin"]).item()),
    )
    val.mean_ = v_npz["mean"]
    val.std_ = v_npz["std"]
    val.min_val_ = v_npz["min_val"]
    val.max_val_ = v_npz["max_val"]
    val.distance_threshold_ = float(np.asarray(v_npz["distance_threshold"]).item())
    val.is_fitted = True

    accepted_mask, _, _ = val.validate(x_test, model_probs=ml_probs)
    reject_rate = float(1.0 - np.mean(accepted_mask))

    result = {
        "source": args.source,
        "data_path": data_path,
        "n_test_samples": len(y_test),
        "ml_accuracy": round(float(np.mean(ml_preds == y_test)), 4),
        "ml_f1": round(float(f1_score(y_test, ml_preds)), 4),
        "ml_roc_auc": round(float(roc_auc_score(y_test, ml_probs)), 4),
        "dl_accuracy": round(float(np.mean(dl_preds == y_test)), 4),
        "dl_f1": round(float(f1_score(y_test, dl_preds)), 4),
        "dl_roc_auc": round(float(roc_auc_score(y_test, dl_probs)), 4),
        "validator_reject_rate": round(reject_rate, 4),
    }

    alert_mgr = AlertManager(
        log_path=cfg.alerts.alerts_log_path,
        suppression_window_sec=cfg.alerts.suppression_window_sec,
    )
    if reject_rate > cfg.alerts.validator_reject_max_clean:
        alert_mgr.trigger(
            metric="validator_reject_rate",
            value=reject_rate,
            threshold=cfg.alerts.validator_reject_max_clean,
            level="WARNING",
            action="review_input_distribution",
        )

    out_dir = Path("reports/LAB8")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_{args.source}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[evaluate] {json.dumps(result)}")


def cmd_monitor() -> None:
    args = parse_common_args("Run drift and service monitoring with alerting")
    cfg = load_lab8_config(args.config)
    _ = resolve_data_path(cfg, args.source)
    subprocess.run([sys.executable, "scripts/run_drift_monitor.py"], check=True)
    print(f"[monitor] Completed monitoring cycle on source={args.source}")
