from pathlib import Path

import pandas as pd
from pydantic import ValidationError
import pytest
import torch

from src.config_schema import DLConfig
from src.train_dl import set_seed, SpamMLP


def test_pydantic_rejects_nonexistent_data():
    """Проверка отказа при отсутствии файла данных."""
    with pytest.raises(ValidationError):
        DLConfig(data_path=Path("data/raw/non_existent_spam.csv"))


def test_pydantic_rejects_invalid_params():
    """Проверка отказа при недопустимых гиперпараметрах."""
    with pytest.raises(ValidationError):
        DLConfig(
            data_path=Path("data/raw/spambase.csv"),
            learning_rate=-0.01,
        )

    with pytest.raises(ValidationError):
        DLConfig(
            data_path=Path("data/raw/spambase.csv"),
            dropout_rate=1.5,
        )


def test_seed_divergence_negative_control():
    """Негативный контроль: без сида веса расходятся."""
    set_seed(42)
    m1 = SpamMLP(57, 64, 32, 0.2)
    w1 = m1.net[0].weight.detach().clone()

    set_seed(42)
    m2 = SpamMLP(57, 64, 32, 0.2)
    w2 = m2.net[0].weight.detach().clone()
    assert torch.equal(w1, w2)

    torch.seed()
    m3 = SpamMLP(57, 64, 32, 0.2)
    w3 = m3.net[0].weight.detach().clone()
    assert not torch.equal(w1, w3)


def test_streaming_memory_bounded():
    """Контроль ограничения памяти стриминга."""
    mem_path = Path("reports/LAB3/memory_compare.csv")
    assert mem_path.exists()
    df = pd.read_csv(mem_path)
    full_ram = df[df["mode"] == "full_dataset_mode"]["ram_mb"].values[0]
    chunked_ram = df[df["mode"] == "chunked_mode"]["ram_mb"].values[0]
    assert chunked_ram < full_ram
