import pandas as pd
import glob
import os
import gc
import logging
import psutil

# ==========================================
# 1. Setup Logging & Memory Tracking
# ==========================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/merge_clean.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_memory_usage():
    """دالة بترجع استهلاك الرامات الحالي بالميجابايت"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

logging.info("🚀 Starting Master Merge Process...")
print(f"💾 Initial Memory Usage: {get_memory_usage():.2f} MB")

# ==========================================
# 2. Read & Validate BTS Files
# ==========================================
bts_files = glob.glob("data/raw/bts_2024_*.csv")
if not bts_files:
    logging.error("❌ No BTS files found in data/raw/")
    raise FileNotFoundError("No BTS files found!")

logging.info(f"📂 Found {len(bts_files)} BTS files to process.")
bts_list = []

for file in bts_files:
    try:
        print(f"Loading {os.path.basename(file)}...")
        # تحديد الأعمدة المطلوبة فقط لتوفير الذاكرة
        cols = ['FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'ORIGIN', 'DEST', 
                'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'TAXI_IN', 
                'CRS_ARR_TIME', 'ARR_DELAY', 'CANCELLED', 'DIVERTED', 'DISTANCE', 'DAY_OF_WEEK']
        
        # قراءة الملف
        df_temp = pd.read_csv(file, usecols=cols)
        bts_list.append(df_temp)
        logging.info(f"✅ Loaded {file} successfully.")
        
    except Exception as e:
        # لو ملف فيه مشكلة، بنسجلها وبنكمل
        logging.warning(f"⚠️ Failed to load {file}: {e}")
        continue

if not bts_list:
    logging.error("❌ No valid BTS data loaded. Exiting.")
    raise ValueError("No valid BTS data loaded.")

print("🔗 Combining all BTS months...")
bts_df = pd.concat(bts_list, ignore_index=True)
del bts_list # مسح القايمة من الذاكرة عشان نوفر
gc.collect() # تنظيف الذاكرة

logging.info(f"✅ Combined BTS data shape: {bts_df.shape}")
print(f"💾 Memory after BTS Merge: {get_memory_usage():.2f} MB")

# ==========================================
# 3. Load & Validate Weather Data
# ==========================================
weather_path = "data/raw/all_weather_2024_full.csv"
try:
    print("☁️ Loading weather data...")
    weather_df = pd.read_csv(weather_path)
    logging.info(f"✅ Weather data loaded. Shape: {weather_df.shape}")
except Exception as e:
    logging.error(f"❌ Failed to load weather data: {e}")
    raise

# ==========================================
# 4. Data Formatting & Merging
# ==========================================
print("⚙️ Formatting dates and times...")
bts_df['hour'] = (bts_df['CRS_DEP_TIME'] // 100).astype(int)
bts_df['FlightDate'] = pd.to_datetime(bts_df['FL_DATE'], format='mixed').dt.strftime('%Y-%m-%d')

weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
weather_df['date'] = weather_df['datetime'].dt.strftime('%Y-%m-%d')
weather_df['hour'] = weather_df['datetime'].dt.hour

print("🔗 Merging flights with weather...")
try:
    merged_df = pd.merge(
        bts_df, weather_df, 
        left_on=['ORIGIN', 'FlightDate', 'hour'], 
        right_on=['ORIGIN', 'date', 'hour'], 
        how='left'
    )
    merged_df.drop(columns=['date', 'datetime'], inplace=True, errors='ignore')
    logging.info("✅ Merge completed successfully.")
except Exception as e:
    logging.error(f"❌ Merge failed: {e}")
    raise

# مسح الداتا القديمة من الذاكرة
del bts_df, weather_df
gc.collect()
print(f"💾 Memory after Merge: {get_memory_usage():.2f} MB")

# ==========================================
# 5. Cleaning & Data Validation
# ==========================================
print("🧹 Starting Cleaning Phase...")

# أ. شيل الرحلات الملغية والمحولة
initial_len = len(merged_df)
merged_df = merged_df[(merged_df['CANCELLED'] == 0) & (merged_df['DIVERTED'] == 0)]
logging.info(f"Dropped {initial_len - len(merged_df)} cancelled/diverted flights.")
# ب. شيل الرحلات اللي مفيهاش ArrDelay
initial_len = len(merged_df)
merged_df = merged_df.dropna(subset=['ARR_DELAY'])
logging.info(f"Dropped {initial_len - len(merged_df)} rows with missing ARR_DELAY.")

# ج. إنشاء الـ Target
merged_df['is_delayed'] = (merged_df['ARR_DELAY'] >= 30).astype(int)

# د. التحقق من التكرار (Data Validation)
duplicates = merged_df.duplicated(subset=['FL_DATE', 'TAIL_NUM', 'CRS_DEP_TIME']).sum()
if duplicates > 0:
    logging.warning(f"⚠️ Found {duplicates} duplicate flights. Dropping...")
    merged_df.drop_duplicates(subset=['FL_DATE', 'TAIL_NUM', 'CRS_DEP_TIME'], inplace=True)
else:
    logging.info("✅ No duplicates found.")

# ==========================================
# 6. Save Final Data
# ==========================================
output_path = "data/interim/cleaned_flights_weather_12_months.csv"
merged_df.to_csv(output_path, index=False)

logging.info(f"🎉 Process Complete! Saved to {output_path}. Final shape: {merged_df.shape}")
print(f"\n🎉 Process Complete! Final shape: {merged_df.shape}")
print(f"💾 Final Memory Usage: {get_memory_usage():.2f} MB")