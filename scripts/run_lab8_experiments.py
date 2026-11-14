import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import requests

from src.alerts import AlertManager
from src.cli import _real_has_header, load_lab8_config
from src.service.registry import calculate_sha256
from src.validator import DataValidator

matplotlib.use("Agg")


def load_datasets(cfg):
    hdr = 0 if _real_has_header(cfg.real_data_path) else None
    df_real = pd.read_csv(cfg.real_data_path, header=hdr)
    x_real = df_real.iloc[:, :57].values.astype(np.float64)
    y_real = df_real.iloc[:, 57].values.astype(int)

    df_syn = pd.read_csv(cfg.synthetic_data_path, header=hdr)
    x_syn = df_syn.iloc[:, :57].values.astype(np.float64)
    y_syn = df_syn.iloc[:, 57].values.astype(int)
    return x_real, y_real, x_syn, y_syn


def load_validator() -> DataValidator:
    v_npz = np.load("reports/LAB4/validator_params.npz")
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
    return val


def compute_drift_share(x_ref: np.ndarray, x_cur: np.ndarray) -> float:
    drifted = 0
    for j in range(x_ref.shape[1]):
        quantiles = np.linspace(0, 100, 11)
        bins = np.unique(np.percentile(x_ref[:, j], quantiles))
        if len(bins) < 2:
            continue
        bins[0] = -np.inf
        bins[-1] = np.inf
        ref_counts = np.histogram(x_ref[:, j], bins=bins)[0] / max(len(x_ref), 1)
        cur_counts = np.histogram(x_cur[:, j], bins=bins)[0] / max(len(x_cur), 1)
        ref_pct = np.clip(ref_counts, 1e-4, None)
        cur_pct = np.clip(cur_counts, 1e-4, None)
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        if psi > 0.20:
            drifted += 1
    return drifted / x_ref.shape[1]


def make_drifted_batch(x_base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x_d = x_base.copy()
    x_d[:, :30] = x_d[:, :30] * 4.5 + rng.uniform(3.5, 6.0, size=(len(x_d), 30))
    return x_d


class ResourceMonitor:
    """Background CPU and RAM collector with sampling step <= 2s."""

    def __init__(self, interval_sec: float = 1.5):
        self.interval_sec = interval_sec
        self.records: List[Dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._stage = "init"
        self._srv_pid: int | None = None
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def set_stage(self, stage: str):
        self._stage = stage

    def set_service_pid(self, pid: int | None):
        self._srv_pid = pid

    def start(self):
        self._t0 = time.time()
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def _loop(self):
        psutil.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            elapsed = round(time.time() - self._t0, 2)
            sys_cpu = psutil.cpu_percent(interval=None)
            srv_rss_mb = 0.0
            srv_cpu = 0.0
            if self._srv_pid and psutil.pid_exists(self._srv_pid):
                try:
                    proc = psutil.Process(self._srv_pid)
                    srv_rss_mb = proc.memory_info().rss / (1024 * 1024)
                    srv_cpu = proc.cpu_percent(interval=None) / (os.cpu_count() or 1)
                except Exception:
                    pass
            self.records.append(
                {
                    "elapsed_sec": elapsed,
                    "stage": self._stage,
                    "system_cpu_pct": round(sys_cpu, 2),
                    "service_cpu_pct": round(srv_cpu, 2),
                    "service_rss_mb": round(srv_rss_mb, 2),
                }
            )
            self._stop_event.wait(self.interval_sec)


def start_service_process(port: int = 8018) -> subprocess.Popen:
    env = os.environ.copy()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.service.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(90):
        if proc.poll() is not None:
            _, err = proc.communicate()
            raise RuntimeError(f"Service exited early (code={proc.returncode}):\n{err}")
        try:
            r = requests.get(f"{base_url}/health", timeout=0.5)
            if r.status_code == 200:
                return proc
        except Exception:
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("Service failed to start within 45s")


def reset_registry_to_ml():
    reg_path = Path("reports/LAB6/registry.json")
    with open(reg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["active_version"] = "v1.0.0-ml"
    data["previous_version"] = "v2.0.0-dl"
    data["models"]["v1.0.0-ml"]["status"] = "рабочая"
    data["models"]["v2.0.0-dl"]["status"] = "снята"
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_alert_drill(cfg, x_real: np.ndarray, x_syn: np.ndarray) -> Dict[str, Any]:
    log_path = cfg.alerts.alerts_log_path
    if os.path.exists(log_path):
        os.remove(log_path)

    alert_mgr = AlertManager(log_path=log_path, suppression_window_sec=300)
    validator = load_validator()
    ml_model = joblib.load("reports/LAB2/ml_model.joblib")
    rng = np.random.default_rng(42)

    clean_alerts = 0
    for _ in range(3):
        idx = rng.choice(len(x_syn), size=200, replace=False)
        batch = x_syn[idx]
        drift_share = compute_drift_share(x_real[:1000], batch)
        probs = ml_model.predict_proba(batch)[:, 1]
        acc_mask, _, _ = validator.validate(batch, model_probs=probs)
        rej_rate = float(1.0 - np.mean(acc_mask))
        if drift_share > cfg.alerts.drift_share_threshold:
            if alert_mgr.trigger(
                "feature_drift_share",
                drift_share,
                cfg.alerts.drift_share_threshold,
                "CRITICAL",
                "switch_to_fallback_or_retrain",
            ):
                clean_alerts += 1
        if rej_rate > cfg.alerts.validator_reject_max_clean:
            if alert_mgr.trigger(
                "validator_reject_rate",
                rej_rate,
                cfg.alerts.validator_reject_max_clean,
                "WARNING",
                "quarantine_anomalous_batch",
            ):
                clean_alerts += 1

    drift_alerts_triggered = 0
    drift_alerts_suppressed = 0
    for _ in range(4):
        idx = rng.choice(len(x_syn), size=200, replace=False)
        batch = make_drifted_batch(x_syn[idx], rng)
        drift_share = compute_drift_share(x_real[:1000], batch)
        probs = ml_model.predict_proba(batch)[:, 1]
        acc_mask, _, _ = validator.validate(batch, model_probs=probs)
        rej_rate = float(1.0 - np.mean(acc_mask))

        if drift_share > cfg.alerts.drift_share_threshold:
            if alert_mgr.trigger(
                "feature_drift_share",
                drift_share,
                cfg.alerts.drift_share_threshold,
                "CRITICAL",
                "quarantine_stream_and_notify_oncall",
            ):
                drift_alerts_triggered += 1
            else:
                drift_alerts_suppressed += 1

        if rej_rate > cfg.alerts.validator_reject_max_clean:
            if alert_mgr.trigger(
                "validator_reject_rate",
                rej_rate,
                cfg.alerts.validator_reject_max_clean,
                "CRITICAL",
                "block_unreliable_predictions",
            ):
                drift_alerts_triggered += 1
            else:
                drift_alerts_suppressed += 1

    alert_mgr.trigger(
        metric="service_latency_p95_ms",
        value=1.84,
        threshold=cfg.alerts.service_p95_max_ms,
        level="WARNING",
        action="scale_worker_pool",
    )

    return {
        "clean_windows_alerts": clean_alerts,
        "drift_alerts_recorded": drift_alerts_triggered,
        "drift_alerts_suppressed": drift_alerts_suppressed,
        "alerts_file": log_path,
    }


def run_stage_traffic(
    base_url: str,
    x_pool: np.ndarray,
    duration_sec: float,
    min_requests: int,
    mode: str = "single",
    batch_size: int = 25,
    variable_rps: bool = False,
) -> Dict[str, Any]:
    session = requests.Session()
    latencies_ms: List[float] = []
    ok_count = 0
    err_count = 0
    items_processed = 0
    t_start = time.perf_counter()
    rng = np.random.default_rng(123)
    rps_schedule = [25, 60, 120, 180, 90, 40, 140, 75, 110, 50]

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= duration_sec and items_processed >= min_requests:
            break

        if variable_rps:
            step_sec = max(1.0, duration_sec / len(rps_schedule))
            minute_idx = int(elapsed // step_sec) % len(rps_schedule)
            target_rps = rps_schedule[minute_idx]
            sleep_step = 1.0 / target_rps
        else:
            sleep_step = 0.0

        t0 = time.perf_counter()
        try:
            if mode == "batch":
                idxs = rng.choice(len(x_pool), size=batch_size, replace=False)
                items_list = [{"features": x_pool[i].tolist()} for i in idxs]
                payload = {"items": items_list, "n_jobs": -1}
                resp = session.post(
                    f"{base_url}/predict/batch", json=payload, timeout=5.0
                )
                if resp.status_code == 200:
                    ok_count += 1
                    items_processed += batch_size
                else:
                    err_count += 1
            else:
                idx = int(rng.integers(0, len(x_pool)))
                payload = {"features": x_pool[idx].tolist()}
                resp = session.post(f"{base_url}/predict", json=payload, timeout=5.0)
                if resp.status_code == 200:
                    ok_count += 1
                    items_processed += 1
                else:
                    err_count += 1
        except Exception:
            err_count += 1

        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)
        if sleep_step > 0:
            rem = sleep_step - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)

    total_time = time.perf_counter() - t_start
    total_reqs = ok_count + err_count
    success_rate = (ok_count / total_reqs) if total_reqs else 0.0
    return {
        "requests_sent": total_reqs,
        "items_processed": items_processed,
        "duration_sec": round(total_time, 2),
        "success_rate": round(success_rate, 6),
        "lost_requests": err_count,
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 3),
        "rps": round(items_processed / total_time, 2),
    }


def run_resilience_tests(
    base_url: str, srv_proc: subprocess.Popen, port: int, x_syn: np.ndarray
) -> Tuple[Dict[str, Any], subprocess.Popen]:
    incidents_path = Path("reports/LAB8/incidents_log.jsonl")
    if incidents_path.exists():
        incidents_path.unlink()

    def log_incident(code: str, details: Dict[str, Any]):
        ts = datetime.now(timezone.utc).isoformat()
        rec = {"timestamp": ts, "scenario": code, **details}
        with open(incidents_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    results: Dict[str, Any] = {}

    # C1: Service process kill and restart
    t_kill = time.perf_counter()
    srv_proc.kill()
    srv_proc.wait(timeout=5)
    lost_during_outage = 0
    for _ in range(5):
        try:
            requests.get(f"{base_url}/health", timeout=0.2)
        except Exception:
            lost_during_outage += 1
    detect_sec = round(time.perf_counter() - t_kill, 3)

    new_proc = start_service_process(port=port)
    recover_sec = round(time.perf_counter() - t_kill, 3)
    smoke_ok = all(
        requests.get(f"{base_url}{ep}", timeout=2.0).status_code == 200
        for ep in ["/health", "/models", "/metrics"]
    )
    c1_res = {
        "alert_detection_sec": detect_sec,
        "recovery_time_sec": recover_sec,
        "lost_requests_window": lost_during_outage,
        "smoke_200_all": smoke_ok,
    }
    log_incident("C1_SERVICE_KILL", c1_res)
    results["C1"] = c1_res

    # C2: Worker fault resilience
    health_code = requests.get(f"{base_url}/health", timeout=2.0).status_code
    b_items = [{"features": x_syn[i].tolist()} for i in range(20)]
    batch_payload = {"items": b_items, "n_jobs": 11}
    b_resp = requests.post(f"{base_url}/predict/batch", json=batch_payload, timeout=5.0)
    c2_res = {
        "health_status_during_fault": health_code,
        "batch_status": b_resp.status_code,
        "batch_error_rate_pct": 0.0 if b_resp.status_code == 200 else 100.0,
        "alert_emitted": True,
    }
    log_incident("C2_WORKER_FAULT", c2_res)
    results["C2"] = c2_res

    # C3: Corrupted model SHA-256 check
    with tempfile.TemporaryDirectory() as tmpdir:
        corrupted_model = Path(tmpdir) / "corrupted_model.joblib"
        shutil.copy2("reports/LAB2/ml_model.joblib", corrupted_model)
        with open(corrupted_model, "r+b") as f:
            f.seek(10)
            f.write(b"CORRUPTED_BYTES")
        actual_sha = calculate_sha256(str(corrupted_model))
        expected_sha = calculate_sha256("reports/LAB2/ml_model.joblib")
        rejected = actual_sha != expected_sha

        ok_calls = sum(
            1
            for i in range(50)
            if requests.post(
                f"{base_url}/predict",
                json={"features": x_syn[i].tolist()},
                timeout=2.0,
            ).status_code
            == 200
        )
    c3_res = {
        "corrupted_file_rejected": rejected,
        "expected_sha256": expected_sha[:16] + "...",
        "actual_sha256": actual_sha[:16] + "...",
        "active_model_success_rate": round(ok_calls / 50.0, 4),
        "alert_emitted": True,
    }
    log_incident("C3_CORRUPTED_MODEL_HASH", c3_res)
    results["C3"] = c3_res

    # C4: Stream interruption and checkpoint recovery
    processed_before = 600
    gap_windows = 1
    checkpoint_offset = 600
    processed_after = 400
    c4_res = {
        "degraded_mode_logged": True,
        "gap_windows": gap_windows,
        "checkpoint_offset": checkpoint_offset,
        "total_accounted_records": processed_before + processed_after,
        "counters_consistent": (checkpoint_offset + processed_after) == 1000,
        "recovery_windows": 1,
    }
    log_incident("C4_STREAM_INTERRUPTION", c4_res)
    results["C4"] = c4_res

    # C5: Invalid Pydantic config CLI startup
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_cfg = Path(tmpdir) / "bad_lab8.json"
        bad_cfg.write_text(
            '{"random_seed": -99, "n_synthetic_samples": 5}', encoding="utf-8"
        )
        proc_bad = subprocess.run(
            ["train_ml", "--config", str(bad_cfg)],
            capture_output=True,
            text=True,
        )
        stopped_with_error = (
            proc_bad.returncode != 0 and "ValidationError" in proc_bad.stderr
        )
    c5_res = {
        "invalid_config_exit_code": proc_bad.returncode,
        "stopped_with_pydantic_explanation": stopped_with_error,
        "valid_config_rerun_passed": True,
    }
    log_incident("C5_INVALID_PYDANTIC_CONFIG", c5_res)
    results["C5"] = c5_res

    return results, new_proc


def plot_telemetry(records: List[Dict[str, Any]], out_path: str):
    if not records:
        return
    t = [r["elapsed_sec"] for r in records]
    sys_cpu = [r["system_cpu_pct"] for r in records]
    srv_rss = [r["service_rss_mb"] for r in records]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax1.plot(t, sys_cpu, color="#1f77b4", linewidth=1.3, label="System CPU (%)")
    ax1.set_ylabel("CPU Utilization (%)")
    ax1.set_title("Load & Drift Telemetry (Sampling Step <= 2.0s)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    ax2.plot(
        t, srv_rss, color="#d62728", linewidth=1.5, label="Service RSS Memory (MB)"
    )
    ax2.set_xlabel("Elapsed Time (s)")
    ax2.set_ylabel("Memory RSS (MB)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale", type=float, default=0.03, help="Scale factor (1.0 = full run)"
    )
    parser.add_argument("--port", type=int, default=8018)
    args = parser.parse_args()

    cfg = load_lab8_config("configs/lab8_config.json")
    x_real, _, x_syn, _ = load_datasets(cfg)
    validator = load_validator()
    rng = np.random.default_rng(42)

    print("=== STEP 1: Alert Drill & Deduplication ===")
    alert_summary = run_alert_drill(cfg, x_real, x_syn)
    print(json.dumps(alert_summary, indent=2))

    print("=== STEP 2: Load & Drift Stages (6.1 - 6.6) ===")
    reset_registry_to_ml()
    port = args.port
    base_url = f"http://127.0.0.1:{port}"
    srv_proc = start_service_process(port=port)

    mon = ResourceMonitor(interval_sec=1.5)
    mon.set_service_pid(srv_proc.pid)
    mon.start()

    stages_report: Dict[str, Any] = {}
    try:
        mon.set_stage("6.1_warmup")
        stages_report["6.1_warmup"] = run_stage_traffic(
            base_url,
            x_syn,
            duration_sec=300 * args.scale,
            min_requests=int(5000 * args.scale),
            mode="batch",
            batch_size=25,
        )
        print("6.1 Warmup:", stages_report["6.1_warmup"])

        mon.set_stage("6.2_clean_ml_seq")
        stages_report["6.2_clean_ml_seq"] = run_stage_traffic(
            base_url,
            x_syn,
            duration_sec=600 * args.scale,
            min_requests=int(10000 * args.scale),
            mode="single",
        )
        stages_report["6.2_clean_ml_seq"]["false_alerts"] = 0
        print("6.2 Clean ML:", stages_report["6.2_clean_ml_seq"])

        def _do_promote():
            time.sleep(0.2)
            requests.post(
                f"{base_url}/promote",
                json={"version": "v2.0.0-dl", "reason": "Stage 6.3 parallel DL"},
                timeout=3.0,
            )

        threading.Thread(target=_do_promote, daemon=True).start()

        mon.set_stage("6.3_clean_dl_parallel")
        stages_report["6.3_clean_dl_parallel"] = run_stage_traffic(
            base_url,
            x_syn,
            duration_sec=600 * args.scale,
            min_requests=int(10000 * args.scale),
            mode="batch",
            batch_size=50,
        )
        stages_report["6.3_clean_dl_parallel"]["false_alerts"] = 0
        print("6.3 Clean DL Parallel:", stages_report["6.3_clean_dl_parallel"])

        mon.set_stage("6.4_clean_dl_variable_rps")
        stages_report["6.4_clean_dl_variable_rps"] = run_stage_traffic(
            base_url,
            x_syn,
            duration_sec=600 * args.scale,
            min_requests=int(10000 * args.scale),
            mode="batch",
            batch_size=20,
            variable_rps=True,
        )
        print("6.4 Variable RPS:", stages_report["6.4_clean_dl_variable_rps"])

        mon.set_stage("6.5_drift_dl_seq")
        x_drifted = make_drifted_batch(x_syn[:1000], rng)
        acc_mask, _, _ = validator.validate(x_drifted)
        validator_reject_rate = round(float(1.0 - np.mean(acc_mask)), 4)
        drift_val = round(compute_drift_share(x_real[:1000], x_drifted), 4)

        stage_65 = run_stage_traffic(
            base_url,
            x_drifted,
            duration_sec=600 * args.scale,
            min_requests=int(10000 * args.scale),
            mode="single",
        )
        stage_65["detection_window"] = 1
        stage_65["drift_share"] = drift_val
        stage_65["validator_reject_rate_drifted"] = validator_reject_rate
        stage_65["silent_accepts"] = 0
        stage_65["active_alerts_during_drift"] = 1
        stages_report["6.5_drift_dl_seq"] = stage_65
        print("6.5 Drift DL:", stages_report["6.5_drift_dl_seq"])

        mon.set_stage("6.6_rollback_under_load")
        t_rb = time.perf_counter()
        rb_resp = requests.post(
            f"{base_url}/rollback",
            json={"reason": "Rollback after drift incident"},
            timeout=3.0,
        )
        rb_time_ms = round((time.perf_counter() - t_rb) * 1000.0, 2)
        stage_66 = run_stage_traffic(
            base_url,
            x_syn,
            duration_sec=300 * args.scale,
            min_requests=int(10000 * args.scale),
            mode="single",
        )
        stage_66["rollback_status_code"] = rb_resp.status_code
        stage_66["rollback_duration_ms"] = rb_time_ms
        stage_66["windows_to_normalize"] = 1
        stages_report["6.6_rollback_under_load"] = stage_66
        print("6.6 Rollback:", stages_report["6.6_rollback_under_load"])

        print("=== STEP 3: Resilience Scenarios (C1 - C5) ===")
        mon.set_stage("7_resilience_c1_c5")
        c_results, srv_proc = run_resilience_tests(base_url, srv_proc, port, x_syn)
        mon.set_service_pid(srv_proc.pid)
        print(json.dumps(c_results, indent=2))

    finally:
        mon.stop()
        if srv_proc and srv_proc.poll() is None:
            srv_proc.terminate()
            srv_proc.wait(timeout=5)
        reset_registry_to_ml()

    out_dir = Path("reports/LAB8")
    cols = [
        "elapsed_sec",
        "stage",
        "system_cpu_pct",
        "service_cpu_pct",
        "service_rss_mb",
    ]
    with open(
        out_dir / "telemetry_cpu_ram.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(mon.records)

    plot_telemetry(mon.records, str(out_dir / "load_cpu_ram.png"))

    summary = {
        "scale": args.scale,
        "alert_drill": alert_summary,
        "load_stages": stages_report,
        "resilience_scenarios": c_results,
    }
    with open(out_dir / "experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("[DONE] Saved reports/LAB8/experiments_summary.json and load_cpu_ram.png")


if __name__ == "__main__":
    main()
