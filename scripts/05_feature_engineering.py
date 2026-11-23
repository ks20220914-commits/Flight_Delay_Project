import pandas as pd
import numpy as np
import gc

print("🚀 Starting Feature Engineering for the ENTIRE 12 Months dataset...")
print("⏳ This might take a few minutes and will use a lot of RAM...")

input_path = "data/interim/cleaned_flights_weather_12_months.csv"
output_path = "data/processed/final_features_12_months.csv"

# 1. Load the whole dataset
print("📂 Loading data...")
df = pd.read_csv(input_path)
print(f"✅ Data loaded. Shape: {df.shape}")

# 2. Sort data (Crucial for shift operations)
print("🔄 Sorting data by Plane and Time...")
# Sort to ensure previous flights are correctly identified
df = df.sort_values(by=['TAIL_NUM', 'FL_DATE', 'CRS_DEP_TIME']).reset_index(drop=True)

# 3. Previous Flight Delay
print("⚙️ Calculating previous flight delay...")
df['prev_arr_delay'] = df.groupby('TAIL_NUM')['ARR_DELAY'].shift(1)

# 4. Turnaround Time
print("⚙️ Calculating turnaround time...")
df['prev_crs_arr_time'] = df.groupby('TAIL_NUM')['CRS_ARR_TIME'].shift(1)
df['turnaround_time'] = df['CRS_DEP_TIME'] - df['prev_crs_arr_time']
# Fix negative times
df.loc[df['turnaround_time'] < 0, 'turnaround_time'] = np.nan

# 5. Basic Time Features
df['dep_hour'] = (df['CRS_DEP_TIME'] // 100).astype(int)

# 6. Clean up missing values
print("🧹 Dropping rows with no history and filling weather NaNs...")
# Drop rows where prev_arr_delay is missing (first flight of each plane)
df = df.dropna(subset=['prev_arr_delay', 'turnaround_time'])

# Fill missing weather with 0
weather_cols = ['temperature_2m', 'precipitation', 'rain', 'snowfall', 'wind_speed_10m', 'cloud_cover']
df[weather_cols] = df[weather_cols].fillna(0)

print(f"📊 Final shape after FE: {df.shape}")

# 7. Save
print("💾 Saving to CSV...")
df.to_csv(output_path, index=False)

print(f"\n🎉 Feature Engineering Complete! Saved to: {output_path}")