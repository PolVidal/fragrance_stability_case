"""Single source of truth for column roles and the normal ranges given in the use-case statement."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_PARQUET = ROOT / "data" / "raw" / "fragrance_stability.parquet"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"
CLEAN_PARQUET = PROCESSED_DIR / "batches_clean.parquet"
MODEL_PATH = ROOT / "models" / "stability_predictor.joblib"

ID_COLS = ["batch_id", "event_timestamp"]
# The raw file has an extra `plant_location` column not described in the statement
CATEGORICAL_COLS = ["plant_location", "plant_country", "production_line_id"]
# Temperature arrives as value + unit (not `reaction_temperature_celsius` as in the statement):
# the normal range below only applies to readings in Celsius.
TEMP_VALUE, TEMP_UNIT = "temp_value", "temp_unit"
SENSOR_COLS = [
    "temp_value",
    "vessel_pressure_bar",
    "refractive_index",
    "density_g_cm3",
    "ph_level",
    "mixing_torque_nm",
    "mass_flow_rate_kg_h",
]
# Timestamp formats found in the raw file, tried in this order.
# The slash format is day/month (values above 12 appear in the first field, never in the second).
TIMESTAMP_FORMATS = {
    "iso": "%Y-%m-%d %H:%M:%S",
    "day_month_year": "%d/%m/%Y",
    "month_name": "%b %d, %Y",
}
TARGET = "stability_months"
# Derived from the target: never use as model inputs (target leakage)
LEAKAGE_COLS = ["purity_grade", "oxidation_risk_flag"]

# Normal operating ranges from the statement (min, max)
NORMAL_RANGES = {
    "temp_value": (75.0, 100.0),  # Celsius
    "vessel_pressure_bar": (2.0, 3.0),
    "refractive_index": (1.43, 1.49),
    "density_g_cm3": (0.80, 0.90),
    "ph_level": (5.0, 6.0),
    "mixing_torque_nm": (9.0, 12.5),
    "mass_flow_rate_kg_h": (380.0, 520.0),
}
# Sentinel codes found in the audit: one fixed value repeated hundreds of times, far outside
# any plausible reading, all in Plant-04 (temperature: 210 °C or its Fahrenheit twin 410 °F)
SENTINELS = {
    "temp_value": [210.0, 410.0],
    "vessel_pressure_bar": [-1.0],
    "density_g_cm3": [860.0],
    "ph_level": [14.0],
}
# Physically impossible values (sensor failure, not process anomaly)
NON_NEGATIVE = ["vessel_pressure_bar", "mixing_torque_nm", "mass_flow_rate_kg_h"]
TARGET_RANGE = (18.0, 30.0)

# --- Clean table (phase 2) -------------------------------------------------- #
# Temperature in °C after unit harmonisation (the name used in the statement)
TEMP_CELSIUS = "reaction_temperature_celsius"
CLEAN_SENSOR_COLS = [TEMP_CELSIUS] + [c for c in SENSOR_COLS if c != TEMP_VALUE]

