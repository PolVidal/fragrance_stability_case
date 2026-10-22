"""Shared test setup: make `fragrance` importable and provide a tiny raw table."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def raw() -> pd.DataFrame:
    """Five hand-made rows with the raw schema, one per situation the rules must handle:
    0 normal ISO row, 1 a °F reading in dd/mm/yyyy, 2 sentinels (410 °F, pH 14) in "Mon dd, yyyy",
    3 a row with every sensor missing, 4 a plausible out-of-range pH (6.1) that must survive."""
    nan = np.nan
    return pd.DataFrame({
        "batch_id": ["B-1", "B-2", "B-3", "B-4", "B-5"],
        "event_timestamp": ["2023-05-08 10:30:00", "16/06/2023", "May 15, 2023",
                            "2023-04-01 00:00:30", "2023-06-29 23:59:30"],
        "plant_location": ["Plant-04", "Plant-04", "Plant-04", "Plant-01", "Plant-02"],
        "plant_country": ["Country-04", "Country-04", "Country-04", "Country-01", "Country-02"],
        "production_line_id": ["Fragrance-Synthesis-Reactor-07", "Fragrance-Synthesis-Reactor-03",
                               "Fragrance-Synthesis-Reactor-01", "Fragrance-Synthesis-Reactor-10",
                               "Fragrance-Synthesis-Reactor-05"],
        "temp_value": [85.0, 185.0, 410.0, nan, 90.0],
        "temp_unit": ["C", "F", "F", nan, "C"],
        "vessel_pressure_bar": [2.3, 2.4, -1.0, nan, 2.2],
        "refractive_index": [1.46, 1.45, 1.47, nan, 1.46],
        "density_g_cm3": [0.86, 0.85, 860.0, nan, 0.87],
        "ph_level": [5.5, 5.4, 14.0, nan, 6.1],
        "mixing_torque_nm": [10.8, 12.0, 11.9, nan, 12.1],
        "mass_flow_rate_kg_h": [450.0, 460.0, 455.0, nan, 440.0],
        "stability_months": [18.6, 21.8, 21.0, 22.1, 19.6],
        "purity_grade": ["Technical Grade", "Industrial Grade", "Industrial Grade",
                         "Industrial Grade", "Technical Grade"],
        "oxidation_risk_flag": [True, False, False, False, True],
    })
