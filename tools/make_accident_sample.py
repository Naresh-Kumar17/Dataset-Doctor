"""
Generate realistic synthetic road accident sample dataset for Phase 0 and baseline testing.
Keeps an ~85% minor/slight (0) vs 15% severe (1) target distribution with realistic feature overlap.
"""
import numpy as np
import pandas as pd
from pathlib import Path

def generate_accident_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    n_severe = int(n_samples * 0.15)
    n_slight = n_samples - n_severe
    
    # Shared base distributions with subtle shifts for realism and non-trivial classification
    # Speed limit: slight bias toward higher limits for severe
    speed_slight = np.random.choice([30, 40, 50, 60, 70], size=n_slight, p=[0.50, 0.20, 0.15, 0.10, 0.05])
    speed_severe = np.random.choice([30, 40, 50, 60, 70], size=n_severe, p=[0.30, 0.15, 0.15, 0.20, 0.20])
    
    # Driver age: overlapping normal distributions
    age_slight = np.random.normal(loc=39, scale=14, size=n_slight).clip(17, 85).astype(int)
    age_severe = np.random.normal(loc=42, scale=18, size=n_severe).clip(17, 85).astype(int)
    
    # Vehicle age
    veh_age_slight = np.random.exponential(scale=6, size=n_slight).clip(1, 25).astype(int)
    veh_age_severe = np.random.exponential(scale=7.5, size=n_severe).clip(1, 25).astype(int)
    
    # Engine CC
    engine_slight = np.random.normal(loc=1500, scale=400, size=n_slight).clip(900, 3500).astype(int)
    engine_severe = np.random.normal(loc=1700, scale=450, size=n_severe).clip(900, 4000).astype(int)
    
    # Weather
    weather_slight = np.random.choice(["clear", "rain", "fog", "snow"], size=n_slight, p=[0.70, 0.22, 0.06, 0.02])
    weather_severe = np.random.choice(["clear", "rain", "fog", "snow"], size=n_severe, p=[0.58, 0.28, 0.10, 0.04])
    
    # Road surface
    road_slight = np.random.choice(["dry", "wet", "ice"], size=n_slight, p=[0.68, 0.28, 0.04])
    road_severe = np.random.choice(["dry", "wet", "ice"], size=n_severe, p=[0.55, 0.36, 0.09])
    
    # Light condition
    light_slight = np.random.choice(["daylight", "darkness_lit", "darkness_unlit"], size=n_slight, p=[0.70, 0.22, 0.08])
    light_severe = np.random.choice(["daylight", "darkness_lit", "darkness_unlit"], size=n_severe, p=[0.52, 0.28, 0.20])
    
    # Urban or rural
    urban_slight = np.random.choice(["urban", "rural"], size=n_slight, p=[0.72, 0.28])
    urban_severe = np.random.choice(["urban", "rural"], size=n_severe, p=[0.52, 0.48])

    df_slight = pd.DataFrame({
        "speed_limit": speed_slight,
        "driver_age": age_slight,
        "vehicle_age": veh_age_slight,
        "engine_cc": engine_slight,
        "weather": weather_slight,
        "road_surface": road_slight,
        "light_condition": light_slight,
        "urban_or_rural": urban_slight,
        "severe_accident": 0
    })

    df_severe = pd.DataFrame({
        "speed_limit": speed_severe,
        "driver_age": age_severe,
        "vehicle_age": veh_age_severe,
        "engine_cc": engine_severe,
        "weather": weather_severe,
        "road_surface": road_severe,
        "light_condition": light_severe,
        "urban_or_rural": urban_severe,
        "severe_accident": 1
    })

    df = pd.concat([df_slight, df_severe], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "data" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "accident.csv"
    df = generate_accident_data(n_samples=2000, seed=42)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows at {out_path}")
    print("Class distribution:")
    print(df["severe_accident"].value_counts(normalize=True))
