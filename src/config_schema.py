from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class PipelineConfig(BaseModel):
    """(ЛР 2)."""

    data_path: Path = Field(
        default=Path("data/raw/spambase.csv"),
        description="Путь к исходному набору данных в формате CSV",
    )
    chunk_size: int = Field(
        default=500,
        gt=0,
        description="Размер пакета для инкрементальной обработки данных",
    )
    test_size: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Доля выборки для валидации модели",
    )
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Зерно генератора случайных чисел",
    )
    loss_function: Literal["log_loss", "modified_huber"] = Field(
        default="log_loss",
        description="Функция потерь алгоритма SGD",
    )
    alpha: float = Field(
        default=0.0001,
        gt=0.0,
        description="Коэффициент регуляризации для контроля переобучения",
    )
    max_iter_full: int = Field(
        default=1000,
        gt=0,
        description="Максимальное количество итераций на полном датасете",
    )

    @field_validator("data_path")
    @classmethod
    def check_file_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Файл по указанному пути не обнаружен: {v}")
        return v


class DLConfig(BaseModel):
    """Схема параметров пайплайна глубокого обучения PyTorch (ЛР 3)."""

    data_path: Path = Field(
        default=Path("data/raw/spambase.csv"),
        description="Путь к набору данных",
    )
    test_size: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Доля тестовой выборки",
    )
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Зерно генератора случайных чисел",
    )
    batch_size: int = Field(
        default=64,
        gt=0,
        description="Размер мини-батча для обучения",
    )
    chunk_size: int = Field(
        default=500,
        gt=0,
        description="Размер чанка для потокового чтения",
    )
    epochs: int = Field(
        default=30,
        gt=0,
        description="Максимальное количество эпох обучения",
    )
    learning_rate: float = Field(
        default=0.001,
        gt=0.0,
        description="Скорость обучения (learning rate) для Adam",
    )
    early_stopping_patience: int = Field(
        default=5,
        gt=0,
        description="Количество эпох без улучшений до ранней остановки",
    )
    hidden_dim_1: int = Field(
        default=64,
        gt=0,
        description="Размерность первого скрытого слоя MLP",
    )
    hidden_dim_2: int = Field(
        default=32,
        gt=0,
        description="Размерность второго скрытого слоя MLP",
    )
    dropout_rate: float = Field(
        default=0.2,
        ge=0.0,
        lt=1.0,
        description="Коэффициент Dropout",
    )
    tolerance_metric: float = Field(
        default=0.005,
        gt=0.0,
        description="Допустимое расхождение ключевой метрики при воспроизводимости",
    )
    device_parity_tolerance: float = Field(
        default=0.00001,
        gt=0.0,
        description="Допуск расхождения предсказаний между устройствами (CPU/GPU)",
    )

    @field_validator("data_path")
    @classmethod
    def check_file_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Файл по указанному пути не обнаружен: {v}")
        return v
