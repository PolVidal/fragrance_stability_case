"""Unit tests for the cleaning rules. Each test states one behaviour a rule must have.
Run from the repository root with:  pytest -q"""
import numpy as np
import pandas as pd
import pytest

from fragrance import cleaning, config


# --- R1: timestamps ---------------------------------------------------------- #
def test_r1_parses_all_three_formats(raw):
    out, touched = cleaning.parse_event_time(raw)
    assert out["event_time"].notna().all()
    assert out.loc[1, "event_time"] == pd.Timestamp("2023-06-16")  # dd/mm, not mm/dd
    assert out.loc[2, "event_time"] == pd.Timestamp("2023-05-15")
    assert touched == 2


def test_r1_flags_date_only_rows_and_drops_raw_string(raw):
    out, _ = cleaning.parse_event_time(raw)
    assert out["has_time"].tolist() == [True, False, False, True, True]
    assert "event_timestamp" not in out.columns


# --- R3: sentinels ----------------------------------------------------------- #
def test_r3_sentinels_become_nan(raw):
    out, touched = cleaning.sentinels_to_nan(raw)
    assert out.loc[2, ["temp_value", "vessel_pressure_bar", "density_g_cm3", "ph_level"]].isna().all()
    assert touched == 1


def test_r3_keeps_plausible_out_of_range_values(raw):
    out, _ = cleaning.sentinels_to_nan(raw)
    assert out.loc[4, "ph_level"] == 6.1  # process signal, not an error (R6)


def test_r3_does_not_modify_input(raw):
    before = raw.copy()
    cleaning.sentinels_to_nan(raw)
    pd.testing.assert_frame_equal(raw, before)


# --- R2: units ---------------------------------------------------------------- #
def test_r2_converts_fahrenheit_to_celsius(raw):
    after_r3, _ = cleaning.sentinels_to_nan(raw)  # same order as `clean`
    out, touched = cleaning.fahrenheit_to_celsius(after_r3)
    temp = out[config.TEMP_CELSIUS]
    assert temp[1] == pytest.approx(85.0)   # 185 °F -> 85 °C
    assert temp[0] == pytest.approx(85.0)   # already °C, unchanged
    assert np.isnan(temp[2])                # 410 °F sentinel stays NaN, never 210 °C
    assert np.isnan(temp[3])                # missing stays missing
    assert {config.TEMP_VALUE, config.TEMP_UNIT}.isdisjoint(out.columns)
    assert touched == 1


# --- R4: reactor key --------------------------------------------------------- #
def test_r4_reactor_key_is_plant_plus_id(raw):
    out, touched = cleaning.add_reactor_key(raw)
    assert out["reactor"].tolist() == ["Plant-04/R07", "Plant-04/R03", "Plant-04/R01",
                                       "Plant-01/R10", "Plant-02/R05"]
    assert "plant_country" not in out.columns
    assert touched == len(raw)


# --- R5: empty records ------------------------------------------------------- #
def test_r5_flags_but_keeps_empty_records(raw):
    out, touched = cleaning.flag_empty_records(raw)
    assert out["all_sensors_missing"].tolist() == [False, False, False, True, False]
    assert len(out) == len(raw)
    assert touched == 1


# --- Full pipeline ----------------------------------------------------------- #
def test_clean_keeps_every_row_and_reports_each_rule(raw):
    out, report = cleaning.clean(raw)
    assert len(out) == len(raw)
    assert report["rule"].tolist() == ["R1", "R3", "R2", "R4", "R5"]
    assert out.loc[2, config.TEMP_CELSIUS] != pytest.approx(210.0)  # sentinel did not survive as 210 °C


def test_clean_keeps_leakage_columns_out_of_sensor_list(raw):
    out, _ = cleaning.clean(raw)
    assert set(config.LEAKAGE_COLS) <= set(out.columns)          # R8: kept in the file ...
    assert not set(config.LEAKAGE_COLS) & set(config.CLEAN_SENSOR_COLS)  # ... never among sensors
