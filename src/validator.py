from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np


class DataValidator:
    """Валидатор табличных данных на чистом NumPy (Профиль P1)."""

    def __init__(
        self,
        distance_quantile: float = 0.95,
        range_margin: float = 0.1,
        confidence_margin: float = 0.05,
    ):
        self.distance_quantile = distance_quantile
        self.range_margin = range_margin
        self.confidence_margin = confidence_margin

        # Статистики распределения (обучаются на train)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.min_val_: Optional[np.ndarray] = None
        self.max_val_: Optional[np.ndarray] = None
        self.distance_threshold_: Optional[float] = None
        self.feature_names_: Optional[list] = None
        self.is_fitted: bool = False

    def _standardized_distance(self, X: np.ndarray) -> np.ndarray:
        std_safe = np.where(self.std_ < 1e-8, 1.0, self.std_)
        scaled_diff = (X - self.mean_) / std_safe
        return np.sqrt(np.sum(scaled_diff**2, axis=1))

    def fit_full(
        self, X: np.ndarray, feature_names: Optional[list] = None
    ) -> "DataValidator":
        X = np.asarray(X, dtype=np.float64)
        self.feature_names_ = feature_names

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.min_val_ = np.min(X, axis=0)
        self.max_val_ = np.max(X, axis=0)

        # Расчет порога расстояния по квантилю train-выборки
        distances = self._standardized_distance(X)
        self.distance_threshold_ = float(np.quantile(distances, self.distance_quantile))
        self.is_fitted = True
        return self

    def fit_chunked(
        self,
        chunk_generator_func,
        feature_names: Optional[list] = None,
    ) -> "DataValidator":
        self.feature_names_ = feature_names

        # === Проход 1: расчет сумм, квадратов, min и max ===
        n_samples = 0
        sum_x = None
        sum_sq = None
        min_v = None
        max_v = None

        for chunk in chunk_generator_func():
            chunk_arr = np.asarray(chunk, dtype=np.float64)
            n_chunk = chunk_arr.shape[0]
            if n_chunk == 0:
                continue

            n_samples += n_chunk
            if sum_x is None:
                sum_x = np.sum(chunk_arr, axis=0)
                sum_sq = np.sum(chunk_arr**2, axis=0)
                min_v = np.min(chunk_arr, axis=0)
                max_v = np.max(chunk_arr, axis=0)
            else:
                sum_x += np.sum(chunk_arr, axis=0)
                sum_sq += np.sum(chunk_arr**2, axis=0)
                min_v = np.minimum(min_v, np.min(chunk_arr, axis=0))
                max_v = np.maximum(max_v, np.max(chunk_arr, axis=0))

        if n_samples == 0:
            raise ValueError("Генератор чанков не вернул данных.")

        self.mean_ = sum_x / n_samples
        variance = np.maximum((sum_sq / n_samples) - (self.mean_**2), 0.0)
        self.std_ = np.sqrt(variance)
        self.min_val_ = min_v
        self.max_val_ = max_v

        # === Проход 2: расчет расстояний по чанкам ===
        all_distances = []
        for chunk in chunk_generator_func():
            chunk_arr = np.asarray(chunk, dtype=np.float64)
            if chunk_arr.shape[0] == 0:
                continue
            dists = self._standardized_distance(chunk_arr)
            all_distances.append(dists)

        concat_distances = np.concatenate(all_distances)
        self.distance_threshold_ = float(
            np.quantile(concat_distances, self.distance_quantile)
        )
        self.is_fitted = True
        return self

    def validate(
        self,
        X: np.ndarray,
        model_probs: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError(
                "Валидатор не обучен. Вызовите fit_full или fit_chunked."
            )

        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        distances = self._standardized_distance(X)

        # 1. Проверка физических диапазонов с запасом (margin)
        span = self.max_val_ - self.min_val_
        lower_bound = self.min_val_ - self.range_margin * np.maximum(span, 1e-6)
        upper_bound = self.max_val_ + self.range_margin * np.maximum(span, 1e-6)

        range_violation = np.any((X < lower_bound) | (X > upper_bound), axis=1)

        # 2. Проверка расстояния (OOD детекция)
        distance_violation = distances > self.distance_threshold_

        # 3. Проверка уверенности модели (|prob - 0.5| < margin)
        if model_probs is not None:
            model_probs = np.asarray(model_probs).ravel()
            confidence_violation = np.abs(model_probs - 0.5) < self.confidence_margin
        else:
            confidence_violation = np.zeros(n_samples, dtype=bool)

        # Объединенное решение: отказ, если нарушено хотя бы одно правило
        is_rejected = range_violation | distance_violation | confidence_violation
        is_accepted = ~is_rejected

        # Формирование описания причин отказа
        reasons = []
        for i in range(n_samples):
            if is_accepted[i]:
                reasons.append("ACCEPTED")
            else:
                fails = []
                if range_violation[i]:
                    fails.append("RANGE_OUTLIER")
                if distance_violation[i]:
                    fails.append("DISTANCE_OOD")
                if confidence_violation[i]:
                    fails.append("LOW_CONFIDENCE")
                reasons.append("; ".join(fails))

        return is_accepted, np.array(reasons), distances

    def save(self, filepath: Union[str, Path]) -> None:
        """Сохранение параметров валидатора в машиночитаемый .npz."""
        if not self.is_fitted:
            raise RuntimeError("Невозможно сохранить необученный валидатор.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            filepath,
            mean=self.mean_,
            std=self.std_,
            min_val=self.min_val_,
            max_val=self.max_val_,
            distance_threshold=np.array([self.distance_threshold_]),
            distance_quantile=np.array([self.distance_quantile]),
            range_margin=np.array([self.range_margin]),
            confidence_margin=np.array([self.confidence_margin]),
        )

    def load(self, filepath: Union[str, Path]) -> "DataValidator":
        """Загрузка параметров валидатора из .npz."""
        data = np.load(filepath)
        self.mean_ = data["mean"]
        self.std_ = data["std"]
        self.min_val_ = data["min_val"]
        self.max_val_ = data["max_val"]
        self.distance_threshold_ = float(data["distance_threshold"][0])
        self.distance_quantile = float(data["distance_quantile"][0])
        self.range_margin = float(data["range_margin"][0])
        self.confidence_margin = float(data["confidence_margin"][0])
        self.is_fitted = True
        return self
