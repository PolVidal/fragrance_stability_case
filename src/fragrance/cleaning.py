"""Cleaning rules (phase 2): turn the raw table into the clean table used by phases 3 and 4.

Principles
- Every rule comes from an audit finding (notebook 01) and reports how many rows it touched.
- Only **sensor errors** are fixed. Plausible out-of-range values are process signal and stay.
- Model-agnostic: unknown values become NaN; filling them in (imputation) is a phase 4 decision,
  fitted on training data only.
- Each rule takes a DataFrame and returns `(new_df, rows_touched)`. The input is never modified.

Rules and the order they run in (the order matters: see `clean`)
- R1  parse the three timestamp formats
- R3  sentinel codes -> NaN
- R2  temperature °F -> °C, one Celsius column
- R4  physical reactor key; drop the redundant country column
- R5  flag rows with every sensor missing (kept, not dropped)
- R6  no clipping of out-of-range values         - deliberate non-rule
- R7  no imputation of single missing values     - deliberate non-rule, phase 4
- R8  leakage columns kept but never features    - guarded by config.LEAKAGE_COLS
"""
from __future__ import annotations

import pandas as pd

from fragrance import config


def parse_event_time(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """R1. Replace the raw `event_timestamp` string by a datetime `event_time` and
    a boolean `has_time` (False for the date-only formats, which get midnight).

    WHY: three formats coexist. Parsing all of them keeps 100% of dates;
    `has_time` stops later analyses from trusting a fake 00:00.
    TOUCHED: rows that were not in the ISO format (they needed a second format).
    """
    raw_ts = df["event_timestamp"]
    event_time = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for fmt in config.TIMESTAMP_FORMATS.values():
        event_time = event_time.fillna(pd.to_datetime(raw_ts, format=fmt, errors="coerce"))
    has_time = pd.to_datetime(raw_ts, format=config.TIMESTAMP_FORMATS["iso"], errors="coerce").notna()
    out = df.drop(columns="event_timestamp").assign(event_time=event_time, has_time=has_time)
    return out, int((~has_time).sum())


def sentinels_to_nan(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """R3. Set the sentinel codes of `config.SENTINELS` to NaN, column by column.

    Works on the raw columns, so it must run before R2 renames and drops `temp_value`.
    WHY: a sentinel means "no valid reading". Kept, it poisons statistics
    (558 densities of 860 g/cm³ push the mean from 0.86 to 2.89); clipped, it
    would fake an anomaly; NaN is the honest representation.
    TOUCHED: rows with at least one sentinel.
    """
    out = df.copy()
    hit = pd.Series(False, index=df.index)
    for col, codes in config.SENTINELS.items():
        is_code = df[col].isin(codes)
        out[col] = df[col].mask(is_code)
        hit |= is_code
    return out, int(hit.sum())


def fahrenheit_to_celsius(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """R2. Create `config.TEMP_CELSIUS` with every temperature in °C, converting
    the rows where `temp_unit == "F"` with C = (F - 32) * 5 / 9, then drop
    `temp_value` and `temp_unit`.

    WHY: 1,487 readings are correct values in another unit. Unconverted,
    they look like 190 °C, the most extreme "anomalies" in the data.
    TOUCHED: rows actually converted (°F with a value; the 410 °F sentinel is
    already NaN because R3 runs first, and NaN stays NaN through the formula).
    """
    temp, unit = df[config.TEMP_VALUE], df[config.TEMP_UNIT]
    is_fahrenheit = unit.eq("F") & temp.notna()
    celsius = temp.where(~is_fahrenheit, (temp - 32) * 5 / 9)
    out = (df.drop(columns=[config.TEMP_VALUE, config.TEMP_UNIT])
             .assign(**{config.TEMP_CELSIUS: celsius}))
    return out, int(is_fahrenheit.sum())


def add_reactor_key(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """R4. Add `reactor` = plant + reactor ID (e.g. "Plant-04/R07") and drop
    `plant_country`.

    WHY: reactor IDs are reused across plants: "Reactor-07" alone mixes five
    machines. `plant_country` is one-to-one with `plant_location`.
    TOUCHED: all rows (a new column for every row).
    """
    reactor = df["plant_location"] + "/R" + df["production_line_id"].str[-2:]
    out = df.drop(columns="plant_country").assign(reactor=reactor)
    return out, len(out)


def flag_empty_records(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """R5. Add `all_sensors_missing` = True where every sensor is NaN. Rows are kept.

    WHY: 5,344 records lost all sensors at once. They are real batches with
    plant, reactor, time and stability, so they count for root cause. Whether a
    model uses them is a phase 4 decision; the flag makes it a one-line choice.
    Runs after R3, so a row whose only readings were sentinels is also flagged.
    TOUCHED: flagged rows.
    """
    sensors = config.CLEAN_SENSOR_COLS if config.TEMP_CELSIUS in df.columns else config.SENSOR_COLS
    empty = df[sensors].isna().all(axis=1)
    return df.assign(all_sensors_missing=empty), int(empty.sum())


# Order matters: sentinels are codes written in the raw columns, so they become NaN before the
# unit conversion (410 °F is a code, not a temperature; converting it is meaningless and only
# lands on another listed code by coincidence) and before empty records are flagged.
RULES = [
    ("R1", "parse three timestamp formats", parse_event_time),
    ("R3", "sentinel codes -> NaN", sentinels_to_nan),
    ("R2", "temperature °F -> °C", fahrenheit_to_celsius),
    ("R4", "physical reactor key, drop plant_country", add_reactor_key),
    ("R5", "flag rows with all sensors missing", flag_empty_records),
]


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every rule in order.

    WHAT: the clean table and a report with one row per rule: id, description,
    rows touched and their share of all rows.
    WHY: the report is what we show when someone asks "how much did you change?"
    """
    df, rows = raw, []
    for rule_id, description, rule in RULES:
        df, touched = rule(df)
        rows.append({"rule": rule_id, "description": description,
                     "rows_touched": touched, "share": touched / len(raw)})
    return df, pd.DataFrame(rows)


def save_clean(df: pd.DataFrame, path=config.CLEAN_PARQUET) -> None:
    """Write the clean table to `data/processed/` as Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
