from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class PipelineConfig(BaseModel):
    """Схема параметров классического пайплайна машинного обучения."""

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
