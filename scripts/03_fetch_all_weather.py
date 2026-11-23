import pandas as pd
import requests
import time
import os
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# ==========================================
# 1. Setup Professional Logging
# ==========================================
# إنشاء ملف log.txt لتسجيل كل خطوة بتحدث في السكريبت
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/weather_fetch.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("🚀 Starting Weather Data Collection Process...")

# ==========================================
# 2. Setup Robust API Session (Retry Logic)
# ==========================================
# لو النت فصل أو السيرفر رفض، الكود هيحاول 3 مرات تاني لوحده
session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# ==========================================
# 3. Data Loading & Validation
# ==========================================
try:
    airports_path = "data/raw/airports.csv"
    if not os.path.exists(airports_path):
        raise FileNotFoundError(f"File not found: {airports_path}")
        
    airports_df = pd.read_csv(airports_path)
    
    # Data Validation: التأكد إن الأعمدة المهمة موجودة
    required_cols = ['iata_code', 'latitude_deg', 'longitude_deg']
    if not all(col in airports_df.columns for col in required_cols):
        raise ValueError("Airports CSV is missing required columns.")
        
    logging.info("✅ Airports data loaded and validated successfully.")
    
except Exception as e:
    logging.error(f"❌ Error loading airports data: {e}")
    print(f"Error: {e}")
    exit()

# تجهيز الإحداثيات
airports_clean = airports_df[['iata_code', 'latitude_deg', 'longitude_deg']].dropna()
coords_dict = airports_clean.set_index('iata_code').to_dict('index')

# قراءة مطارات BTS
try:
    bts_sample = pd.read_csv("data/raw/bts_2024_01.csv", usecols=['ORIGIN'])
    unique_origins = bts_sample['ORIGIN'].unique()
    logging.info(f"Found {len(unique_origins)} unique origin airports.")
except Exception as e:
    logging.error(f"❌ Error loading BTS sample: {e}")
    exit()

# ==========================================
# 4. Fetching Data with Error Handling
# ==========================================
all_weather_data = []

for i, origin in enumerate(unique_origins):
    # تسجيل التقدم كل 10 مطارات عشان ميبطئش الملف
    if i % 10 == 0:
        logging.info(f"Progress: Fetching weather for {origin} ({i+1}/{len(unique_origins)})")
        print(f"[{i+1}/{len(unique_origins)}] Fetching {origin}...")
    
    if origin not in coords_dict:
        logging.warning(f"⚠️ No coordinates found for {origin}. Skipping.")
        continue
        
    lat = coords_dict[origin]['latitude_deg']
    lon = coords_dict[origin]['longitude_deg']
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "hourly": "temperature_2m,precipitation,rain,snowfall,wind_speed_10m,cloud_cover",
        "timezone": "auto"
    }
    
    try:
        # استخدام الـ session اللي فيه Retry
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status() # هيرمي Error لو الـ status مش 200
        
        data = response.json()
        weather_df = pd.DataFrame(data["hourly"])
        weather_df.rename(columns={"time": "datetime"}, inplace=True)
        weather_df['ORIGIN'] = origin
        all_weather_data.append(weather_df)
        
        time.sleep(0.3) # تأخير بسيط لاحترام الـ API
        
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Failed to fetch weather for {origin}: {e}")
        continue # الكود هيكمل للمطار اللي بعده ومش هيقف
    except KeyError as e:
        logging.error(f"❌ Data format error for {origin}: {e}")
        continue
# ==========================================
# 5. Final Save & Data Validation
# ==========================================
if all_weather_data:
    logging.info("💾 Combining all weather data...")
    final_weather_df = pd.concat(all_weather_data, ignore_index=True)
    
    # Data Validation: التأكد إن الملف النهائي مش فاضي
    if len(final_weather_df) > 0:
        output_path = "data/raw/all_weather_2024_full.csv"
        final_weather_df.to_csv(output_path, index=False)
        logging.info(f"✅ Successfully saved {len(final_weather_df)} rows to {output_path}")
        print("✅ Process Complete! Check logs/weather_fetch.log for details.")
    else:
        logging.error("❌ Final DataFrame is empty!")
else:
    logging.error("❌ No weather data was collected.")