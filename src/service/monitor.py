import os
import time
import requests
import matplotlib.pyplot as plt

SERVICE_URL = "http://127.0.0.1:8000"
LOG_FILE = "reports/LAB6/monitoring_log.txt"
PLOT_FILE = "reports/LAB6/monitoring_status.png"


def run_monitoring(iterations: int = 20, interval_sec: float = 1.0):
    os.makedirs("reports/LAB6", exist_ok=True)
    timestamps = []
    latencies_p95 = []
    memories = []

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"=== Session: {time.ctime()} ===\n")

        for i in range(iterations):
            try:
                resp = requests.get(f"{SERVICE_URL}/metrics", timeout=2.0)
                if resp.status_code == 200:
                    data = resp.json()
                    ts = time.strftime("%H:%M:%S")
                    line = (
                        f"[{ts}] Active: {data['active_version']} | "
                        f"Requests: {data['requests_total']} | "
                        f"Errors: {data['errors_total']} | "
                        f"p95: {data['latency_p95_ms']} ms | "
                        f"RSS: {data['memory_rss_mb']} MB\n"
                    )
                    log.write(line)
                    log.flush()
                    print(line.strip())

                    timestamps.append(i)
                    latencies_p95.append(data["latency_p95_ms"])
                    memories.append(data["memory_rss_mb"])
            except Exception as e:
                log.write(f"[{time.strftime('%H:%M:%S')}] Connection error: {e}\n")
            time.sleep(interval_sec)

    if timestamps:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        color = "tab:blue"
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Latency p95 (ms)", color=color)
        ax1.plot(
            timestamps, latencies_p95, color=color, marker="o", label="Latency p95"
        )
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:green"
        ax2.set_ylabel("Memory RSS (MB)", color=color)
        ax2.plot(timestamps, memories, color=color, linestyle="--", label="RSS MB")
        ax2.tick_params(axis="y", labelcolor=color)

        plt.title("Service Monitoring Metrics")
        fig.tight_layout()
        plt.savefig(PLOT_FILE, dpi=150)
        plt.close()


if __name__ == "__main__":
    run_monitoring(iterations=10, interval_sec=1.0)
