import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def test_no_data_leakage():
    """Тест-ловушка утечки: сверка статистик train со статистиками all."""
    df = pd.read_csv("data/raw/spambase.csv", header=None)
    x_all = df.iloc[:, :-1].values
    y_all = df.iloc[:, -1].values

    x_train, _, _, _ = train_test_split(
        x_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # 1. Корректный скейлер (fit только на train)
    correct_scaler = StandardScaler().fit(x_train)

    # 2. Ошибочный скейлер с утечкой (fit на x_all)
    leaked_scaler = StandardScaler().fit(x_all)

    # Расхождения параметров
    mean_diff = float(np.max(np.abs(correct_scaler.mean_ - leaked_scaler.mean_)))
    scale_diff = float(np.max(np.abs(correct_scaler.scale_ - leaked_scaler.scale_)))

    print(f"[TEST] Максимальное расхождение средних: {mean_diff:.6f}")
    print(f"[TEST] Максимальное расхождение дисперсий: {scale_diff:.6f}")

    assert mean_diff > 1e-4, "Утечка: средние train совпадают с all!"
    assert scale_diff > 1e-4, "Утечка: дисперсии train совпадают с all!"
    print("[SUCCESS] Негативный контроль пройден: утечка отсутствует.")


if __name__ == "__main__":
    test_no_data_leakage()
