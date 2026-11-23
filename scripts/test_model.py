import unittest
import pandas as pd
import joblib
import os
import numpy as np

class TestFlightDelayModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # 1. تحميل الموديل
        model_path = "../models/xgb_flight_delay_model_12m.pkl"
        if not os.path.exists(model_path):
            # في حالة لو المسار مختلف
            model_path = "models/xgb_flight_delay_model_12m.pkl"
            
        cls.model = joblib.load(model_path)
        
        # 2. تجهيز بيانات وهمية للتجربة
        cls.dummy_data = pd.DataFrame([{
            'OP_UNIQUE_CARRIER': 'AA', 'ORIGIN': 'JFK', 'DEST': 'LAX',
            'CRS_DEP_TIME': 1200, 'CRS_ARR_TIME': 1500, 'DISTANCE': 2500,
            'DAY_OF_WEEK': 1, 'prev_arr_delay': 0, 'turnaround_time': 60,
            'temperature_2m': 20.0, 'precipitation': 0.0, 'wind_speed_10m': 10.0, 'cloud_cover': 10.0
        }])

    def test_model_output_shape(self):
        # الاختبار الأول: هل الموديل بيرجع توقع واحد لصف واحد؟
        preds = self.model.predict(self.dummy_data)
        self.assertEqual(len(preds), 1)
        
    def test_probability_range(self):
        # الاختبار التاني: هل الاحتمالية بين 0 و 1؟
        probs = self.model.predict_proba(self.dummy_data)[0]
        self.assertTrue(0.0 <= probs[0] <= 1.0)
        self.assertTrue(0.0 <= probs[1] <= 1.0)

    def test_high_previous_delay(self):
        # الاختبار التالت: لو الرحلة السابقة متأخرة جداً، هل الموديل هيرفع الخطر؟
        high_delay_data = self.dummy_data.copy()
        high_delay_data['prev_arr_delay'] = 180 # 3 ساعات تأخير
        
        prob_normal = self.model.predict_proba(self.dummy_data)[0][1]
        prob_high = self.model.predict_proba(high_delay_data)[0][1]
        
        # المفروض احتمالية التأخير تزيد لما التأخير السابق يزيد
        self.assertGreater(prob_high, prob_normal)

if __name__ == '__main__':
    unittest.main()