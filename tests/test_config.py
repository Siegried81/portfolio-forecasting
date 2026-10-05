"""Unit tests for src/config.py: env-var wiring and static reference data consistency."""
from __future__ import annotations

import dataclasses

import pytest

import src.config as config
from src.config import (
    ALL_KNOWN_TICKERS,
    DEFAULT_EQUITY_TICKERS,
    DEFAULT_FORECAST_COV_METHOD,
    FORECAST_COV_METHOD_LABELS,
    FREQUENCY_TO_PERIODS_PER_YEAR,
    LLMSettings,
    MAX_PCA_FACTORS,
    MIN_PCA_FACTORS,
    QUICK_DATE_RANGES,
    SP500_SECTOR_UNIVERSE,
    _load_groq_keys,
)

_GROQ_ENV_VARS = ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4", "GROQ_API_KEY_5"]


@pytest.fixture(autouse=True)
def _clear_groq_env(monkeypatch):
    for name in _GROQ_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    yield


# ---------------------------------------------------------------------------
# _load_groq_keys
# ---------------------------------------------------------------------------

def test_load_groq_keys_empty_when_nothing_configured():
    assert _load_groq_keys() == []


def test_load_groq_keys_returns_only_primary_when_that_is_all_thats_set(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "primary-key")
    assert _load_groq_keys() == ["primary-key"]


def test_load_groq_keys_collects_all_five_in_order(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setenv("GROQ_API_KEY_2", "k2")
    monkeypatch.setenv("GROQ_API_KEY_3", "k3")
    monkeypatch.setenv("GROQ_API_KEY_4", "k4")
    monkeypatch.setenv("GROQ_API_KEY_5", "k5")
    assert _load_groq_keys() == ["k1", "k2", "k3", "k4", "k5"]


def test_load_groq_keys_skips_a_gap_instead_of_stopping_at_it(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setenv("GROQ_API_KEY_3", "k3")
    assert _load_groq_keys() == ["k1", "k3"]


def test_load_groq_keys_ignores_primary_gap_if_only_secondary_keys_set(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY_2", "k2")
    assert _load_groq_keys() == ["k2"]


# ---------------------------------------------------------------------------
# LLMSettings.groq_api_key
# ---------------------------------------------------------------------------

def test_groq_api_key_property_returns_first_key():
    settings = dataclasses.replace(config.LLM_SETTINGS, groq_api_keys=["a", "b"])
    assert settings.groq_api_key == "a"


def test_groq_api_key_property_none_when_no_keys():
    settings = dataclasses.replace(config.LLM_SETTINGS, groq_api_keys=[])
    assert settings.groq_api_key is None


# ---------------------------------------------------------------------------
# LLMSettings — frozen dataclass, env-var wiring
# ---------------------------------------------------------------------------

def test_llm_settings_is_frozen():
    settings = LLMSettings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.groq_model = "some-other-model"  # type: ignore[misc]


def test_llm_settings_groq_model_defaults_when_env_unset(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert LLMSettings().groq_model == "openai/gpt-oss-120b"


def test_llm_settings_groq_model_reads_env_override(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "some-custom-model")
    assert LLMSettings().groq_model == "some-custom-model"


def test_llm_settings_optional_key_is_none_not_empty_string_when_unset(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    assert LLMSettings().fred_api_key is None


def test_llm_settings_optional_key_empty_string_env_var_also_becomes_none(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "")
    assert LLMSettings().fred_api_key is None


# ---------------------------------------------------------------------------
# Static reference data
# ---------------------------------------------------------------------------

def test_all_known_tickers_is_deduplicated_and_sorted():
    assert list(ALL_KNOWN_TICKERS) == sorted(set(ALL_KNOWN_TICKERS))


def test_all_known_tickers_includes_every_sector_ticker():
    sector_tickers = {t for tickers in SP500_SECTOR_UNIVERSE.values() for t in tickers}
    assert sector_tickers.issubset(set(ALL_KNOWN_TICKERS))


def test_all_known_tickers_includes_default_equity_tickers():
    assert set(DEFAULT_EQUITY_TICKERS).issubset(set(ALL_KNOWN_TICKERS))


def test_frequency_to_periods_per_year_has_expected_values():
    assert FREQUENCY_TO_PERIODS_PER_YEAR == {"daily": 252, "weekly": 52, "monthly": 12, "yearly": 1}


def test_quick_date_ranges_are_strictly_increasing():
    values = list(QUICK_DATE_RANGES.values())
    assert values == sorted(values)


def test_pca_factor_bounds_are_sane():
    assert MIN_PCA_FACTORS < MAX_PCA_FACTORS


def test_sp500_sector_universe_has_no_duplicate_ticker_across_sectors():
    seen: set[str] = set()
    for tickers in SP500_SECTOR_UNIVERSE.values():
        for t in tickers:
            assert t not in seen, f"{t} appears in more than one GICS sector"
            seen.add(t)


def test_horizon_bounds_cover_every_frequency_with_a_consistent_range():
    assert set(config.HORIZON_BOUNDS_BY_FREQUENCY) == set(FREQUENCY_TO_PERIODS_PER_YEAR)
    for low, high, default, step in config.HORIZON_BOUNDS_BY_FREQUENCY.values():
        assert 1 <= low <= default <= high and step >= 1


def test_yearly_horizon_fits_in_the_longest_quick_range():
    """10-90 yearly periods could never be held out of at most 15 years of data."""
    low, _, default, _ = config.HORIZON_BOUNDS_BY_FREQUENCY["yearly"]
    longest_years = max(QUICK_DATE_RANGES.values()) / 365
    assert default < longest_years and low < longest_years


def test_quick_date_ranges_contain_the_app_default():
    assert "5 years" in QUICK_DATE_RANGES  # app.py's start-date default reads this key


# ---------------------------------------------------------------------------
# Forecast-based covariance toggle (sidebar -> run_walk_forward)
# ---------------------------------------------------------------------------

def test_forecast_cov_method_labels_map_to_values_run_walk_forward_accepts():
    # run_walk_forward only branches on COV_METHOD_GARCH; anything else means
    # "reuse the historical covariance", which it expresses as None.
    assert set(FORECAST_COV_METHOD_LABELS.values()) == {None, config.COV_METHOD_GARCH}


def test_forecast_cov_method_default_is_the_historical_behaviour():
    # The default must stay None so the three-portfolio comparison keeps
    # producing the same numbers it did before the toggle existed.
    assert DEFAULT_FORECAST_COV_METHOD is None
    assert DEFAULT_FORECAST_COV_METHOD in FORECAST_COV_METHOD_LABELS.values()


def test_forecast_cov_method_first_label_is_the_default():
    # app.py's selectbox passes index=0, so the first label must be the default.
    first_label = next(iter(FORECAST_COV_METHOD_LABELS))
    assert FORECAST_COV_METHOD_LABELS[first_label] == DEFAULT_FORECAST_COV_METHOD


def test_forecast_cov_method_is_not_confused_with_the_historical_cov_estimator():
    # The historical-covariance selector offers Ledoit-Wolf / PCA; GARCH belongs
    # only to the forecast-covariance switch, and mixing them would silently
    # change what the Historical-based portfolio is measuring.
    assert config.DEFAULT_COV_METHOD in {config.COV_METHOD_LEDOIT_WOLF, config.COV_METHOD_PCA}
    assert config.COV_METHOD_GARCH not in {config.COV_METHOD_LEDOIT_WOLF, config.COV_METHOD_PCA}
