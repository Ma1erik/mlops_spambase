import json
from pathlib import Path
import numpy as np
import pytest
from pydantic import ValidationError

from src.config_schema import GeneratorConfig
from src.generator import (
    SyntheticDataGenerator,
    calculate_ks,
    calculate_psi,
    compute_df_sha256,
    main,
)


@pytest.fixture(scope="module")
def base_config():
    config_path = Path("configs/lab5_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return GeneratorConfig(**json.load(f))


def test_pydantic_rejects_invalid_generator_config():
    with pytest.raises(ValidationError):
        GeneratorConfig(
            data_path=Path("data/raw/spambase.csv"),
            n_samples=-10,
        )

    with pytest.raises(ValidationError):
        GeneratorConfig(
            data_path=Path("data/raw/spambase.csv"),
            label_noise_rate=1.5,
        )


def test_generator_byte_exact_reproducibility(base_config):
    gen = SyntheticDataGenerator(base_config)
    df1 = gen.generate()
    df2 = gen.generate()

    hash1 = compute_df_sha256(df1)
    hash2 = compute_df_sha256(df2)

    assert hash1 == hash2


def test_mean_shift_knob(base_config):
    cfg_base = base_config.model_copy(update={"n_samples": 500, "mean_shift": 0.0})
    cfg_shifted = base_config.model_copy(update={"n_samples": 500, "mean_shift": 2.0})

    df_base = SyntheticDataGenerator(cfg_base).generate()
    df_shifted = SyntheticDataGenerator(cfg_shifted).generate()

    col = df_base.columns[0]
    assert np.mean(df_shifted[col]) > np.mean(df_base[col])


def test_class_ratio_knob(base_config):
    cfg_imbalance = base_config.model_copy(
        update={"n_samples": 500, "class_ratio": 0.8}
    )
    df = SyntheticDataGenerator(cfg_imbalance).generate()
    target_col = df.columns[-1]

    spam_rate = np.mean(df[target_col] == 1)
    assert abs(spam_rate - 0.8) < 0.05


def test_label_noise_knob(base_config):
    cfg_clean = base_config.model_copy(
        update={"n_samples": 500, "label_noise_rate": 0.0}
    )
    cfg_noisy = base_config.model_copy(
        update={"n_samples": 500, "label_noise_rate": 0.5}
    )

    df_clean = SyntheticDataGenerator(cfg_clean).generate()
    df_noisy = SyntheticDataGenerator(cfg_noisy).generate()
    target_col = df_clean.columns[-1]

    flipped_share = np.mean(df_clean[target_col] != df_noisy[target_col])
    assert flipped_share > 0.3


def test_modes_distribution_difference(base_config):
    cfg_sim = base_config.model_copy(update={"mode": "similar", "n_samples": 500})
    cfg_rnd = base_config.model_copy(update={"mode": "random", "n_samples": 500})

    gen_sim = SyntheticDataGenerator(cfg_sim)
    gen_rnd = SyntheticDataGenerator(cfg_rnd)

    df_sim = gen_sim.generate()
    df_rnd = gen_rnd.generate()

    _, pct_psi_sim, _ = gen_sim.compare_with_real(df_sim)
    metrics_rnd, _, _ = gen_rnd.compare_with_real(df_rnd)

    pct_rnd_high_psi = np.mean(metrics_rnd["psi"] >= 0.2) * 100

    assert pct_psi_sim >= 80.0
    assert pct_rnd_high_psi >= 50.0


def test_metric_helpers_and_edge_cases(base_config):
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert calculate_ks(a, b) == 0.0
    assert calculate_psi(a, b) == 0.0

    assert calculate_psi(np.zeros(10), np.zeros(10)) == 0.0

    gen = SyntheticDataGenerator(base_config)
    gen.config = base_config.model_copy(update={"mode": "random"})
    gen.config.__dict__["mode"] = "invalid_mode"
    with pytest.raises(ValueError):
        gen.generate()


def test_generator_main_execution():
    main()
    assert Path("reports/LAB5/lab5_report.md").exists()
    assert Path("reports/LAB5/synthetic_metrics.csv").exists()
    assert Path("reports/LAB5/generator_config.json").exists()
    assert Path("reports/LAB5/spambase_synthetic.csv").exists()
