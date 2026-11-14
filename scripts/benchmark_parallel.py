import os
import sys
import time

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("."))

from src.service.app import app  # noqa: E402

client = TestClient(app)

np.random.seed(42)
TOTAL_POOL = 1000
mock_data = np.abs(np.random.randn(TOTAL_POOL, 57)).tolist()

batch_sizes = [50, 200, 500]
max_cores = os.cpu_count() or 1
workers_list = sorted(list(set([1, 2, min(4, max_cores), max_cores])))
repeats = 5

for _ in range(3):
    client.post(
        "/predict/batch",
        json={
            "items": [{"features": mock_data[i]} for i in range(100)],
            "n_jobs": 1,
        },
    )

records = []

for b_size in batch_sizes:
    batch_items = [{"features": mock_data[i % TOTAL_POOL]} for i in range(b_size)]
    baseline_time = None

    for n_jobs in workers_list:
        latencies_ms = []

        for _ in range(repeats):
            t0 = time.perf_counter()
            resp = client.post(
                "/predict/batch", json={"items": batch_items, "n_jobs": n_jobs}
            )
            elapsed = (time.perf_counter() - t0) * 1000.0

            if resp.status_code == 200:
                latencies_ms.append(elapsed)

        p50 = float(np.percentile(latencies_ms, 50))
        p95 = float(np.percentile(latencies_ms, 95))
        mean_lat = float(np.mean(latencies_ms))
        throughput_samples_sec = round((b_size / (mean_lat / 1000.0)), 2)

        if n_jobs == 1:
            baseline_time = mean_lat
            speedup = 1.0
        else:
            speedup = round(baseline_time / mean_lat, 3) if baseline_time else 1.0

        records.append(
            {
                "batch_size": b_size,
                "workers": n_jobs,
                "repeats": repeats,
                "latency_p50_ms": round(p50, 2),
                "latency_p95_ms": round(p95, 2),
                "latency_mean_ms": round(mean_lat, 2),
                "throughput_samples_per_sec": throughput_samples_sec,
                "speedup": speedup,
            }
        )

        msg = (
            f"Batch: {b_size:3d} | Workers: {n_jobs:2d} | "
            f"p50: {p50:6.2f} ms | p95: {p95:6.2f} ms | "
            f"RPS: {throughput_samples_sec:8.1f} | Speedup: {speedup}x"
        )
        print(msg)

os.makedirs("reports/LAB7", exist_ok=True)
output_path = "reports/LAB7/parallel_benchmark.csv"
pd.DataFrame(records).to_csv(output_path, index=False)
print(f"saved: {output_path}")
