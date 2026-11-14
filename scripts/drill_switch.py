import time
import threading
import requests

SERVICE_URL = "http://127.0.0.1:8000"
NUM_THREADS = 5

success_count = 0
fail_count = 0
dropped_count = 0
lock = threading.Lock()
stop_flag = False


def send_worker(worker_id: int):
    global success_count, fail_count, dropped_count
    payload = {"features": [0.0] * 57}
    while not stop_flag:
        try:
            resp = requests.post(f"{SERVICE_URL}/predict", json=payload, timeout=2.0)
            with lock:
                if resp.status_code == 200:
                    success_count += 1
                else:
                    fail_count += 1
        except Exception:
            with lock:
                dropped_count += 1
        time.sleep(0.005)


def run_drill():
    global stop_flag
    threads = [
        threading.Thread(target=send_worker, args=(i,)) for i in range(NUM_THREADS)
    ]
    for t in threads:
        t.start()

    time.sleep(1.0)
    requests.post(
        f"{SERVICE_URL}/promote", json={"version": "v2.0.0-dl", "reason": "Promote DL"}
    )

    time.sleep(1.5)
    requests.post(f"{SERVICE_URL}/rollback", json={"reason": "Rollback to ML"})

    time.sleep(1.0)
    stop_flag = True
    for t in threads:
        t.join()

    total = success_count + fail_count + dropped_count
    success_rate = (success_count / total * 100.0) if total > 0 else 0.0

    print(f"Total: {total}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Dropped: {dropped_count}")
    print(f"Success rate: {success_rate:.3f}%")

    assert dropped_count == 0
    assert success_rate >= 99.9


if __name__ == "__main__":
    run_drill()
