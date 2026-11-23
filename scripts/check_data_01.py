import pandas as pd
import os

# مسار الملف
file_path = os.path.join("data", "raw", "bts_2024_01.csv")

# التأكد إن الملف موجود
if os.path.exists(file_path):
    print("✅ الملف موجود! جاري القراءة...")
    
    # قراءة أول 5 صفوف بس عشان نتأكد
    df = pd.read_csv(file_path, nrows=5)
    
    print(f"\n📊 عدد الأعمدة اللي نزلت: {len(df.columns)}")
    print("\n📋 أسماء الأعمدة:")
    for col in df.columns:
        print(f"- {col}")
        
    print("\n👀 أول 3 صفوف من البيانات:")
    print(df.head(3))
else:
    print("❌ الملف مش موجود في المسار ده! تأكد إنك حطيته في data/raw")