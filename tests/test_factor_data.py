"""Unit tests for src/factor_data.py. Fully mocked — no real network call."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest
import requests
import streamlit as st

import src.factor_data as factor_data
from src.factor_data import (
    THREE_FACTOR_COLUMNS,
    compute_factor_exposures,
    fetch_fama_french_factors,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _three_factor_csv(rows: list[tuple[str, float, float, float, float]]) -> str:
    header = "This file was created by CMPT_ME_BEME_RETS using the 202608 CRSP database.\n"
    header += ",Mkt-RF,SMB,HML,RF\n"
    body = "\n".join(f"{date},{mkt},{smb},{hml},{rf}" for date, mkt, smb, hml, rf in rows)
    footer = "\n\nCopyright 2026 Kenneth R. French\n"
    return header + body + footer


def _zip_bytes(csv_text: str, filename: str = "F-F_Research_Data_Factors.CSV") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(filename, csv_text)
    return buffer.getvalue()


class _FakeZipResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


# ---------------------------------------------------------------------------
# fetch_fama_french_factors
# ---------------------------------------------------------------------------

def test_fetch_fama_french_factors_parses_a_well_formed_file(monkeypatch):
    csv_text = _three_factor_csv([
        ("20260901", 0.50, 0.10, -0.05, 0.02),
        ("20260902", -0.30, 0.05, 0.02, 0.02),
    ])
    monkeypatch.setattr(factor_data.requests, "get", lambda *a, **k: _FakeZipResponse(_zip_bytes(csv_text)))

    df = fetch_fama_french_factors("3-factor")

    assert df is not None
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 2
    assert df.iloc[0]["Mkt-RF"] == pytest.approx(0.0050)
    assert df.index[0] == pd.Timestamp("2026-09-01")


def test_fetch_fama_french_factors_strips_header_and_footer_junk(monkeypatch):
    csv_text = _three_factor_csv([("20260901", 0.10, 0.0, 0.0, 0.01)])
    monkeypatch.setattr(factor_data.requests, "get", lambda *a, **k: _FakeZipResponse(_zip_bytes(csv_text)))
    df = fetch_fama_french_factors("3-factor")
    assert len(df) == 1


def test_fetch_fama_french_factors_returns_none_for_unrecognised_model():
    assert fetch_fama_french_factors("2-factor") is None


def test_fetch_fama_french_factors_returns_none_on_request_failure(monkeypatch):
    def _raise(*a, **k):
        raise requests.RequestException("network down")
    monkeypatch.setattr(factor_data.requests, "get", _raise)
    assert fetch_fama_french_factors("3-factor") is None


def test_fetch_fama_french_factors_returns_none_on_bad_zip(monkeypatch):
    monkeypatch.setattr(factor_data.requests, "get", lambda *a, **k: _FakeZipResponse(b"not a real zip file"))
    assert fetch_fama_french_factors("3-factor") is None


def test_fetch_fama_french_factors_returns_none_when_zip_has_no_csv_member(monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "no csv here")
    monkeypatch.setattr(factor_data.requests, "get", lambda *a, **k: _FakeZipResponse(buffer.getvalue()))
    assert fetch_fama_french_factors("3-factor") is None


def test_fetch_fama_french_factors_returns_none_on_unparseable_data_block(monkeypatch):
    csv_text = ",Mkt-RF,SMB,HML,RF\n20260901,not,a,number,here\n"
    monkeypatch.setattr(factor_data.requests, "get", lambda *a, **k: _FakeZipResponse(_zip_bytes(csv_text)))
    df = fetch_fama_french_factors("3-factor")
    assert df is None or df.empty


# ---------------------------------------------------------------------------
# compute_factor_exposures
# ---------------------------------------------------------------------------

def _synthetic_factors(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({
        "Mkt-RF": rng.normal(0.0004, 0.01, n),
        "SMB": rng.normal(0.0001, 0.005, n),
        "HML": rng.normal(0.0001, 0.005, n),
        "RF": np.full(n, 0.00005),
    }, index=idx)


def test_compute_factor_exposures_recovers_a_known_single_factor_loading():
    factors = _synthetic_factors(300, seed=1)
    portfolio_returns = factors["RF"] + 1.5 * factors["Mkt-RF"]
    result = compute_factor_exposures(portfolio_returns, factors)
    assert result is not None
    assert result["loadings"]["Mkt-RF"] == pytest.approx(1.5, abs=0.05)
    assert result["loadings"]["SMB"] == pytest.approx(0.0, abs=0.05)
    assert result["r_squared"] > 0.95


def test_compute_factor_exposures_detects_five_factor_columns_when_present():
    factors = _synthetic_factors(300, seed=2)
    factors["RMW"] = np.random.default_rng(3).normal(0.0001, 0.005, len(factors))
    factors["CMA"] = np.random.default_rng(4).normal(0.0001, 0.005, len(factors))
    portfolio_returns = factors["RF"] + 1.0 * factors["Mkt-RF"]
    result = compute_factor_exposures(portfolio_returns, factors)
    assert set(result["loadings"].keys()) == {"Mkt-RF", "SMB", "HML", "RMW", "CMA"}


def test_compute_factor_exposures_none_with_too_few_overlapping_observations():
    factors = _synthetic_factors(10, seed=5)
    portfolio_returns = factors["RF"] + factors["Mkt-RF"]
    assert compute_factor_exposures(portfolio_returns, factors) is None


def test_compute_factor_exposures_none_when_portfolio_and_factors_share_no_dates():
    factors = _synthetic_factors(300, seed=6)
    non_overlapping_index = pd.bdate_range("1990-01-01", periods=300)
    portfolio_returns = pd.Series(np.random.default_rng(7).normal(0, 0.01, 300), index=non_overlapping_index)
    assert compute_factor_exposures(portfolio_returns, factors) is None


def test_compute_factor_exposures_none_when_factors_missing_expected_columns():
    malformed_factors = pd.DataFrame({"Mkt-RF": [0.01] * 50, "RF": [0.0001] * 50}, index=pd.bdate_range("2024-01-01", periods=50))
    portfolio_returns = pd.Series([0.01] * 50, index=malformed_factors.index)
    assert compute_factor_exposures(portfolio_returns, malformed_factors) is None


def test_compute_factor_exposures_reports_actual_overlapping_sample_size():
    factors = _synthetic_factors(300, seed=8)
    portfolio_returns = (factors["RF"] + factors["Mkt-RF"]).iloc[:100]
    result = compute_factor_exposures(portfolio_returns, factors)
    assert result["n_obs"] == 100


def test_compute_factor_exposures_alpha_near_zero_when_return_is_pure_factor_exposure():
    factors = _synthetic_factors(300, seed=9)
    portfolio_returns = factors["RF"] + 1.0 * factors["Mkt-RF"]
    result = compute_factor_exposures(portfolio_returns, factors)
    assert result["alpha_annualised"] == pytest.approx(0.0, abs=0.05)