"""
Unit tests for the macro indicators added to src/macro_data.py on 2026-09-04.

Priority: _fetch_fred_yoy_change's date-matching logic, and the divide_by
parameter — SAHMREALTIME is NOT expressed as a percentage-of-100 figure like
every other series here, so getting divide_by wrong silently wrecks its 0.50
recession threshold rather than raising.
"""
import dataclasses

import pytest

import src.macro_data as macro_data
from src.macro_data import _fetch_fred_series_latest, _fetch_fred_yoy_change


class _FakeResponse:
    def __init__(self, json_data: dict):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def _fred_observations(values_by_date: dict[str, str]) -> dict:
    """Build a fake FRED payload, most-recent-first (matches sort_order=desc)."""
    dates = sorted(values_by_date.keys(), reverse=True)
    return {"observations": [{"date": d, "value": values_by_date[d]} for d in dates]}


def _patch_fred_key(monkeypatch, value):
    """LLM_SETTINGS is a frozen dataclass (config.py, deliberately immutable) —
    monkeypatch can't mutate a single field on it directly (raises
    FrozenInstanceError). Replace the whole object with a copy instead, scoped
    to macro_data's own imported name so other modules are unaffected."""
    fake_settings = dataclasses.replace(macro_data.LLM_SETTINGS, fred_api_key=value)
    monkeypatch.setattr(macro_data, "LLM_SETTINGS", fake_settings)


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    _patch_fred_key(monkeypatch, "fake-test-key")


# ---------------------------------------------------------------------------
# _fetch_fred_series_latest — the divide_by parameter
# ---------------------------------------------------------------------------

def test_fetch_fred_series_latest_default_divides_by_100(monkeypatch):
    monkeypatch.setattr(
        macro_data.requests, "get",
        lambda *a, **k: _FakeResponse(_fred_observations({"2026-09-01": "4.12"})),
    )
    assert _fetch_fred_series_latest("DGS10") == pytest.approx(0.0412)


def test_fetch_fred_series_latest_sahm_rule_must_not_be_divided(monkeypatch):
    # SAHMREALTIME's raw value (e.g. 0.33) IS the indicator itself — dividing
    # by 100 here would silently move every reading below the 0.50 recession
    # threshold, turning a real signal into permanent "n/a" territory.
    monkeypatch.setattr(
        macro_data.requests, "get",
        lambda *a, **k: _FakeResponse(_fred_observations({"2026-09-01": "0.55"})),
    )
    assert _fetch_fred_series_latest("SAHMREALTIME", divide_by=1.0) == pytest.approx(0.55)


def test_fetch_fred_series_latest_skips_missing_values(monkeypatch):
    monkeypatch.setattr(
        macro_data.requests, "get",
        lambda *a, **k: _FakeResponse(_fred_observations({"2026-09-01": ".", "2026-08-01": "3.90"})),
    )
    assert _fetch_fred_series_latest("DGS10") == pytest.approx(0.0390)


def test_fetch_fred_series_latest_none_without_a_key(monkeypatch):
    _patch_fred_key(monkeypatch, None)
    assert _fetch_fred_series_latest("DGS10") is None


# ---------------------------------------------------------------------------
# _fetch_fred_yoy_change — CPI-style year-over-year calculation
# ---------------------------------------------------------------------------

def _month_before(date_str: str, n: int) -> str:
    """Subtract n whole months from a YYYY-MM-01 date string, with year rollover."""
    year, month, _ = map(int, date_str.split("-"))
    month -= n
    while month <= 0:
        month += 12
        year -= 1
    return f"{year:04d}-{month:02d}-01"


def test_fetch_fred_yoy_change_computes_expected_ratio(monkeypatch):
    # 14 consecutive months back from the latest: latest=310.0, exactly 12
    # months earlier=300.0, everything else a filler value -> +3.33% YoY.
    values = {}
    for i in range(14):
        date = _month_before("2026-08-01", i)
        if i == 0:
            values[date] = "310.0"
        elif i == 12:
            values[date] = "300.0"
        else:
            values[date] = "305.0"
    monkeypatch.setattr(macro_data.requests, "get", lambda *a, **k: _FakeResponse(_fred_observations(values)))
    result = _fetch_fred_yoy_change("CPIAUCSL")
    assert result == pytest.approx(310.0 / 300.0 - 1.0, abs=1e-6)


def test_fetch_fred_yoy_change_none_with_too_little_history(monkeypatch):
    values = {"2026-08-01": "310.0", "2026-07-01": "308.0"}
    monkeypatch.setattr(macro_data.requests, "get", lambda *a, **k: _FakeResponse(_fred_observations(values)))
    assert _fetch_fred_yoy_change("CPIAUCSL") is None


def test_fetch_fred_yoy_change_none_without_a_key(monkeypatch):
    _patch_fred_key(monkeypatch, None)
    assert _fetch_fred_yoy_change("CPIAUCSL") is None


# ---------------------------------------------------------------------------
# fetch_macro_snapshot — GDP growth and the industrial-production PMI proxy
# ---------------------------------------------------------------------------

def test_fetch_macro_snapshot_includes_gdp_and_industrial_production(monkeypatch):
    import streamlit as st
    st.cache_data.clear()  # fetch_macro_snapshot is @st.cache_data — avoid a stale
    # result from another test leaking in, since it takes no args to key on.

    # GDP growth uses _fetch_fred_series_latest (already a rate, like the
    # yield series) -> single observation is enough.
    # Industrial production uses _fetch_fred_yoy_change (an index level, like
    # CPI) -> needs 13+ months of history.
    call_log = []

    def fake_get(url, params=None, **kwargs):
        series_id = params["series_id"]
        call_log.append(series_id)
        if series_id == macro_data.GDP_GROWTH_SERIES_ID:
            return _FakeResponse(_fred_observations({"2026-06-01": "2.80"}))
        if series_id == macro_data.INDUSTRIAL_PRODUCTION_SERIES_ID:
            values = {}
            for i in range(14):
                year, month = 2026, 8 - i
                while month <= 0:
                    month += 12
                    year -= 1
                values[f"{year:04d}-{month:02d}-01"] = "103.0" if i == 0 else ("100.0" if i == 12 else "101.0")
            return _FakeResponse(_fred_observations(values))
        # every other series (yields, unemployment, etc.) not under test here
        return _FakeResponse(_fred_observations({"2026-08-01": "4.00"}))

    monkeypatch.setattr(macro_data.requests, "get", fake_get)
    snapshot = macro_data.fetch_macro_snapshot()

    assert snapshot["gdp_growth_qoq_annualized"] == pytest.approx(0.028, abs=1e-6)
    assert snapshot["industrial_production_yoy"] == pytest.approx(103.0 / 100.0 - 1.0, abs=1e-6)
    assert macro_data.GDP_GROWTH_SERIES_ID in call_log
    assert macro_data.INDUSTRIAL_PRODUCTION_SERIES_ID in call_log