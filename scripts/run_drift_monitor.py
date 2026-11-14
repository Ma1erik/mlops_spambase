import json
import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from evidently import Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402
from src.service.registry import ModelRegistry  # noqa: E402

data_path = "data/raw/spambase.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Missing data file: {data_path}")

df_raw = pd.read_csv(data_path)
feature_cols = list(df_raw.columns[:57])

ref_df = df_raw.iloc[:800].copy()
reg = ModelRegistry(registry_file="reports/LAB6/registry.json")
_, model = reg.get_active_model()

ref_preds = [
    round(float(model.predict(r.tolist())[1]), 4) for r in ref_df[feature_cols].values
]
ref_df["prediction_prob"] = ref_preds

WINDOW_SIZE = 150
STEP = 50
TOTAL_STREAM = 600

clean_stream = df_raw.iloc[800 : 800 + TOTAL_STREAM].copy()

drift_stream = clean_stream.copy().reset_index(drop=True)
drift_start_idx = 200
drift_features = feature_cols[:15]
drift_stream.loc[drift_start_idx:, drift_features] = (
    drift_stream.loc[drift_start_idx:, drift_features] * 3.0 + 1.25
)


def evaluate_stream(stream_df: pd.DataFrame, scenario_name: str):
    stream_preds = [
        round(float(model.predict(r.tolist())[1]), 4)
        for r in stream_df[feature_cols].values
    ]
    stream_df["prediction_prob"] = stream_preds

    num_windows = (len(stream_df) - WINDOW_SIZE) // STEP + 1
    metrics_records = []
    first_alert_window = None
    first_alert_sample = None
    last_eval = None

    for w_i in range(num_windows):
        start = w_i * STEP
        end = start + WINDOW_SIZE
        current_window = stream_df.iloc[start:end].copy()

        eval_cols = feature_cols + ["prediction_prob"]
        report = Report([DataDriftPreset()], include_tests=True)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            my_eval = report.run(
                reference_data=ref_df[eval_cols],
                current_data=current_window[eval_cols],
            )
        last_eval = my_eval

        res = my_eval.dict() if hasattr(my_eval, "dict") else json.loads(my_eval.json())
        tests = res.get("tests", [])
        failed_tests = [t for t in tests if t.get("status") in ("FAIL", "ERROR")]

        pred_drift = any(
            t.get("status") in ("FAIL", "ERROR")
            and "prediction_prob" in str(t.get("name", "") + t.get("description", ""))
            for t in tests
        )

        num_drifted = len(
            [
                t
                for t in failed_tests
                if "prediction_prob"
                not in str(t.get("name", "") + t.get("description", ""))
            ]
        )
        share_drifted = round(num_drifted / len(feature_cols), 3)

        dataset_drift = False
        for m in res.get("metrics", []):
            val = m.get("value") or m.get("result") or {}
            if isinstance(val, dict):
                if "count" in val and "share" in val and num_drifted == 0:
                    num_drifted = int(val["count"])
                    share_drifted = float(val["share"])
                if val.get("dataset_drift") is True:
                    dataset_drift = True

        if share_drifted >= 0.25:
            dataset_drift = True

        # Единое согласованное правило алерта
        alert = bool(share_drifted >= 0.25 or (pred_drift and share_drifted >= 0.20))

        if alert and first_alert_window is None and end > drift_start_idx:
            first_alert_window = w_i
            first_alert_sample = end

        metrics_records.append(
            {
                "scenario": scenario_name,
                "window_idx": w_i,
                "sample_start": start,
                "sample_end": end,
                "share_of_drifted_features": share_drifted,
                "number_of_drifted_features": num_drifted,
                "dataset_drift": dataset_drift,
                "prediction_drift": pred_drift,
                "alert": alert,
            }
        )

        msg = (
            f"[{scenario_name}] w{w_i:02d} ({start:3d}..{end:3d}) | "
            f"drift: {share_drifted*100:4.1f}% ({num_drifted:02d}/57) | "
            f"csi: {str(pred_drift):5s} | alert: {alert}"
        )
        print(msg)

    if scenario_name == "drift_stream":
        delay = first_alert_sample - drift_start_idx if first_alert_sample else -1
        msg = (
            f"detection: window={first_alert_window}, "
            f"sample={first_alert_sample}, delay={delay}"
        )
        print(msg)

    return metrics_records, last_eval


clean_records, _ = evaluate_stream(clean_stream, "clean_stream")
drift_records, final_eval = evaluate_stream(drift_stream, "drift_stream")

all_records = clean_records + drift_records
df_metrics = pd.DataFrame(all_records)
os.makedirs("reports/LAB7", exist_ok=True)
csv_out = "reports/LAB7/drift_metrics.csv"
df_metrics.to_csv(csv_out, index=False)
print(f"saved: {csv_out}")

html_out = "reports/LAB7/drift_report.html"
if hasattr(final_eval, "save_html"):
    final_eval.save_html(html_out)
    print(f"saved: {html_out}")
