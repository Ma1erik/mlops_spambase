import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.model_selection import train_test_split

from src.config_schema import ValidatorConfig
from src.train_validator import generate_alien_data
from src.validator import DataValidator


@pytest.fixture(scope="module")
def valid_config():
    config_path = Path("configs/lab4_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return ValidatorConfig(**json.load(f))


@pytest.fixture(scope="module")
def prepared_data(valid_config):
    df = pd.read_csv(valid_config.data_path)
    X = (
        df.drop(columns=["target"]).values
        if "target" in df.columns
        else df.iloc[:, :-1].values
    )
    X_train, X_test = train_test_split(
        X,
        test_size=valid_config.test_size,
        random_state=valid_config.random_seed,
    )
    return X_train, X_test


# Контроль 1: отказ Pydantic на некорректных параметрах
def test_pydantic_rejects_invalid_validator_config():
    with pytest.raises(ValidationError):
        ValidatorConfig(
            data_path=Path("data/raw/spambase.csv"),
            chunk_size=-100,
        )

    with pytest.raises(ValidationError):
        ValidatorConfig(
            data_path=Path("data/raw/spambase.csv"),
            distance_quantile=1.5,
        )


# Контроль 2: совпадение статистик Full и Chunked режимов
def test_chunked_stats_match_full(valid_config, prepared_data):
    X_train, _ = prepared_data

    val_full = DataValidator(distance_quantile=valid_config.distance_quantile)
    val_full.fit_full(X_train)

    def chunk_gen():
        for i in range(0, len(X_train), valid_config.chunk_size):
            yield X_train[i : i + valid_config.chunk_size]

    val_chunked = DataValidator(distance_quantile=valid_config.distance_quantile)
    val_chunked.fit_chunked(chunk_gen)

    assert (
        np.max(np.abs(val_full.mean_ - val_chunked.mean_))
        <= valid_config.tolerance_stats
    )
    assert (
        np.max(np.abs(val_full.std_ - val_chunked.std_)) <= valid_config.tolerance_stats
    )
    assert (
        abs(val_full.distance_threshold_ - val_chunked.distance_threshold_)
        <= valid_config.tolerance_stats
    )


# Контроль 3: ограничение доли ложных отказов на своих данных
def test_false_alarm_rate_bounded(valid_config, prepared_data):
    X_train, X_test = prepared_data
    val = DataValidator(distance_quantile=valid_config.distance_quantile)
    val.fit_full(X_train)

    is_accepted, _, _ = val.validate(X_test)
    false_alarm_rate = np.mean(~is_accepted)

    assert false_alarm_rate <= valid_config.max_false_alarm_rate


# Контроль 4: полнота отсева чужеродных данных (OOD)
def test_alien_data_rejection_recall(valid_config, prepared_data):
    X_train, X_test = prepared_data
    val = DataValidator(distance_quantile=valid_config.distance_quantile)
    val.fit_full(X_train)

    X_alien, _ = generate_alien_data(X_test, seed=valid_config.random_seed)
    is_accepted, _, _ = val.validate(X_alien)
    recall = np.mean(~is_accepted)

    assert recall >= valid_config.min_rejection_recall
