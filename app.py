import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, brier_score_loss, log_loss
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

class MobileGoalAlert:
    """
    Telefon için optimize edilmiş İlk Gol Alarm Sistemi
    """

    def __init__(self, data_path=None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.threshold = 0.65  # Alarm eşiği
        self.min_data_points = 50  # Minimum veri noktası
        self.alert_history = []
        self.model_version = "1.0"
        self.last_training_date = None

        # Özellik listesi (veri setindeki tüm önemli değişkenler)
        self.feature_names = [
            # Maç öncesi
            'Home_Pos', 'Away_Pos',
            # Alarm zamanı (Alert Time)
            'H_Score', 'A_Score',
            'H_Momentum', 'A_Momentum',
            'H_xG', 'A_xG',
            'H_SOT', 'A_SOT',
            'H_SOFF', 'A_SOFF',
            'H_Corners', 'A_Corners',
            'H_Attacks', 'A_Attacks',
            'H_Dn_Attacks', 'A_Dn_Attacks',
            'H_Poss', 'A_Poss',
            'H_Y_Cards', 'A_Y_Cards',
            # Oranlar (Live Odds)
            'Odds_Home', 'Odds_Draw', 'Odds_Away',
            'Odds_Over_0.5', 'Odds_Under_0.5',
            'Odds_Over_1.5', 'Odds_Under_1.5',
            # Pre-Match Odds
            'PM_Odds_Home', 'PM_Odds_Draw', 'PM_Odds_Away',
            'PM_Odds_Over_0.5', 'PM_Odds_Under_0.5',
        ]

        if data_path and os.path.exists(data_path):
            self.load_data(data_path)

    def load_data(self, file_path):
        """Veri setini yükle ve işle"""
        print(f"📊 Veri yükleniyor: {file_path}")
        self.raw_data = pd.read_csv(file_path, encoding='utf-8-sig')
        self.process_data()
        return self

    def process_data(self):
        """Veriyi model için hazırla"""
        df = self.raw_data.copy()

        # İlk gol dakikasını çıkar
        def extract_first_goal(row):
            if pd.isna(row.get('Goal Times', '')):
                return None
            goals = str(row['Goal Times']).split(',')
            if goals and goals[0].strip().isdigit():
                return int(goals[0].strip())
            return None

        df['First_Goal_Minute'] = df.apply(extract_first_goal, axis=1)

        # İlk gol var mı?
        df['Has_Goal'] = df['First_Goal_Minute'].notna().astype(int)

        # Alarm zamanından ilk gole kadar geçen süre
        def get_alert_time(row):
            try:
                if pd.isna(row.get('Date', '')):
                    return None
                date_str = str(row['Date']).strip()
                if ' ' in date_str:
                    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                return datetime.strptime(date_str, '%Y-%m-%d')
            except:
                return None

        df['Alert_Time'] = df.apply(get_alert_time, axis=1)

        # Hedef değişken: 15 dakika içinde gol olacak mı?
        def target_15min(row):
            if pd.isna(row.get('First_Goal_Minute')) or pd.isna(row.get('Alert_Time')):
                return 0
            # Alarm dakikasını hesapla
            alert_time = row['Alert_Time']
            # Maçın başlangıcından itibaren kaç dakika geçmiş?
            # Basitleştirilmiş: Alert_Time'ı dakika olarak kullan
            if isinstance(alert_time, datetime):
                # Alert_Time'dan dakika çıkar (örnek: 15:30 -> 15)
                alert_minute = alert_time.minute
            else:
                alert_minute = 0
            # İlk gol dakikası ile karşılaştır
            if row['First_Goal_Minute'] <= alert_minute + 15:
                return 1
            return 0

        df['Target_15min'] = df.apply(target_15min, axis=1)

        # Özellikleri hazırla
        feature_data = {}
        for feat in self.feature_names:
            if feat in df.columns:
                feature_data[feat] = df[feat].fillna(0).values
            else:
                # Kolon yoksa 0 ile doldur
                feature_data[feat] = np.zeros(len(df))

        self.X = pd.DataFrame(feature_data)
        self.y = df['Target_15min'].fillna(0).values
        self.first_goal_minutes = df['First_Goal_Minute'].fillna(0).values

        print(f"✅ Veri işlendi: {len(self.X)} maç, {len(self.feature_names)} özellik")
        return self

    def train_model(self):
        """Modeli eğit"""
        print("🧠 Model eğitiliyor...")

        # Eksik değerleri doldur
        X = self.X.fillna(0)
        y = self.y

        # Eğitim/test böl
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Ölçeklendir
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Model oluştur (XGBoost + LightGBM + Random Forest ensemble)
        print("  - XGBoost eğitiliyor...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_scaled, y_train)

        print("  - LightGBM eğitiliyor...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train_scaled, y_train)

        print("  - Random Forest eğitiliyor...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)

        # Ensemble model
        class EnsembleModel:
            def __init__(self, models):
                self.models = models

            def predict_proba(self, X):
                probs = []
                for model in self.models:
                    if hasattr(model, 'predict_proba'):
                        probs.append(model.predict_proba(X)[:, 1])
                return np.mean(probs, axis=0)

            def predict(self, X):
                return (self.predict_proba(X) > 0.5).astype(int)

        self.model = EnsembleModel([xgb_model, lgb_model, rf_model])
        self.feature_columns = X.columns.tolist()

        # Performans metrikleri
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0.5

        print(f"\n📊 Model Performansı:")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        print(f"  ROC-AUC:   {roc_auc:.3f}")

        # En iyi eşiği bul
        self.find_optimal_threshold(X_test_scaled, y_test)

        self.last_training_date = datetime.now()
        self.model_version = f"v{datetime.now().strftime('%Y%m%d')}"

        # Modeli kaydet
        self.save_model()

        return self

    def find_optimal_threshold(self, X, y):
        """En iyi alarm eşiğini bul"""
        thresholds = np.arange(0.3, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.65

        y_proba = self.model.predict_proba(X)

        for thresh in thresholds:
            y_pred = (y_proba > thresh).astype(int)
            try:
                f1 = f1_score(y, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = thresh
            except:
                continue

        self.threshold = best_threshold
        print(f"  ✅ Optimal eşik: {best_threshold:.2f} (F1: {best_f1:.3f})")
        return best_threshold

    def predict_match(self, match_data):
        """
        Canlı maç için tahmin yap

        Args:
            match_data: dict - Canlı istatistikler

        Returns:
            dict: Tahmin sonuçları
        """
        if self.model is None:
            print("⚠️ Model eğitilmemiş!")
            return None

        # Gelen veriyi DataFrame'e çevir
        df = pd.DataFrame([match_data])

        # Eksik özellikleri doldur
        for feat in self.feature_columns:
            if feat not in df.columns:
                df[feat] = 0

        df = df[self.feature_columns].fillna(0)

        # Ölçeklendir
        X_scaled = self.scaler.transform(df)

        # Tahmin
        prob = self.model.predict_proba(X_scaled)[0]

        # Alarm kararı
        alert = prob >= self.threshold

        # Güven skoru (0-100)
        confidence = min(100, int(prob * 100 * 1.2))

        # Risk skoru
        risk = 100 - confidence

        # Beklenen gol dakikası
        expected_minute = self.estimate_goal_time(df)

        # Benzer maçları bul
        similar_matches = self.find_similar_matches(df, n=10)

        return {
            'alert': alert,
            'probability': round(prob * 100, 1),
            'confidence': min(100, confidence),
            'risk': min(100, risk),
            'expected_goal_minute': expected_minute,
            'threshold': round(self.threshold * 100, 1),
            'similar_matches_count': len(similar_matches),
            'similar_success_rate': self.calculate_similar_success(similar_matches),
            'timestamp': datetime.now().isoformat()
        }

    def estimate_goal_time(self, match_features):
        """Beklenen gol dakikasını tahmin et"""
        # Basit tahmin: ortalama ilk gol dakikası + xG ve baskıya göre düzeltme
        base = 30  # Varsayılan ortalama

        # xG'ye göre düzeltme
        h_xg = match_features.get('H_xG', 0.5)
        a_xg = match_features.get('A_xG', 0.5)
        xg_factor = max(0.5, min(2.0, (h_xg + a_xg) / 0.5))

        # Momentum'a göre düzeltme
        h_mom = match_features.get('H_Momentum', 50)
        a_mom = match_features.get('A_Momentum', 50)
        momentum_factor = 1 + (h_mom - a_mom) / 100

        # Korner faktörü
        corners = match_features.get('H_Corners', 0) + match_features.get('A_Corners', 0)
        corner_factor = 1 - (corners / 20) * 0.1

        estimated = base / (xg_factor * momentum_factor * corner_factor)
        return max(5, min(90, int(estimated)))

    def find_similar_matches(self, match_features, n=10):
        """Benzer maçları bul (KNN)"""
        if len(self.X) < n:
            return []

        # Basit benzerlik: xG ve momentum
        h_xg = match_features.get('H_xG', 0.5)
        a_xg = match_features.get('A_xG', 0.5)

        similarities = []
        for idx, row in self.X.iterrows():
            sim = 1 - (abs(row.get('H_xG', 0.5) - h_xg) + abs(row.get('A_xG', 0.5) - a_xg))
            similarities.append((idx, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in similarities[:n]]

    def calculate_similar_success(self, similar_indices):
        """Benzer maçların başarı oranı"""
        if not similar_indices:
            return 0

        success = sum(self.y[idx] for idx in similar_indices if idx < len(self.y))
        return success / len(similar_indices)

    def save_model(self, path="goal_alert_model.pkl"):
        """Modeli kaydet"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'threshold': self.threshold,
            'model_version': self.model_version,
            'last_training_date': self.last_training_date
        }
        joblib.dump(model_data, path)
        print(f"💾 Model kaydedildi: {path}")

    def load_model(self, path="goal_alert_model.pkl"):
        """Modeli yükle"""
        if os.path.exists(path):
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.threshold = model_data['threshold']
            self.model_version = model_data.get('model_version', 'unknown')
            self.last_training_date = model_data.get('last_training_date', None)
            print(f"📂 Model yüklendi: {path} (v{self.model_version})")
            return True
        print("⚠️ Model dosyası bulunamadı!")
        return False


class MatchSimulator:
    """Canlı maç simülatörü (test için)"""

    def __init__(self, model):
        self.model = model
        self.current_match = None

    def simulate_match(self, match_id=1):
        """Test maçı simüle et"""
        print(f"\n⚽ Maç {match_id} simüle ediliyor...")

        # Örnek maç verisi
        match_data = {
            'Home_Pos': 3,
            'Away_Pos': 8,
            'H_Score': 0,
            'A_Score': 0,
            'H_Momentum': 65,
            'A_Momentum': 35,
            'H_xG': 0.8,
            'A_xG': 0.2,
            'H_SOT': 4,
            'A_SOT': 1,
            'H_Corners': 3,
            'A_Corners': 1,
            'H_Attacks': 28,
            'A_Attacks': 15,
            'H_Dn_Attacks': 12,
            'A_Dn_Attacks': 5,
            'H_Poss': 62,
            'A_Poss': 38,
            'H_Y_Cards': 0,
            'A_Y_Cards': 1,
            'Odds_Home': 1.8,
            'Odds_Draw': 3.5,
            'Odds_Away': 4.2,
            'Odds_Over_0.5': 1.25,
            'Odds_Under_0.5': 3.5,
            'PM_Odds_Home': 1.9,
            'PM_Odds_Draw': 3.4,
            'PM_Odds_Away': 4.0,
            'PM_Odds_Over_0.5': 1.22,
            'PM_Odds_Under_0.5': 3.8,
        }

        # 3 farklı senaryo
        scenarios = [
            ("Baskılı Atak", {
                'H_Momentum': 72, 'H_xG': 1.2, 'H_SOT': 6,
                'H_Corners': 5, 'H_Dn_Attacks': 18
            }),
            ("Dengeli Maç", {
                'H_Momentum': 52, 'A_Momentum': 48, 'H_xG': 0.5,
                'A_xG': 0.4, 'H_SOT': 2, 'A_SOT': 2
            }),
            ("Savunma Ağırlıklı", {
                'H_Momentum': 30, 'A_Momentum': 70, 'H_xG': 0.1,
                'A_xG': 0.8, 'H_SOT': 0, 'A_SOT': 5
            })
        ]

        results = []
        for name, updates in scenarios:
            data = match_data.copy()
            data.update(updates)

            result = self.model.predict_match(data)

            if result:
                result['scenario'] = name
                results.append(result)

                print(f"\n📊 {name}:")
                print(f"  İlk Gol Olasılığı: {result['probability']}%")
                print(f"  Alarm: {'🔔 VER' if result['alert'] else '❌ VERME'}")
                print(f"  Güven: {result['confidence']}/100")
                print(f"  Risk: {result['risk']}/100")
                print(f"  Beklenen Gol: {result['expected_goal_minute']}. dakika")

        return results


class MobileApp:
    """Telefon uygulaması arayüzü"""

    def __init__(self):
        self.model = None
        self.simulator = None

    def start(self):
        """Uygulamayı başlat"""
        print("\n" + "="*50)
        print("   📱 MOBİL İLK GOL ALARM SİSTEMİ")
        print("="*50)

        # Veri dosyasını kontrol et
        data_file = input("\n📂 Veri dosyası yolu (boş bırak=CSV varsayılan): ").strip()
        if not data_file:
            data_file = "InPlayGuru_Strategy_1050497_Picks_1784327580.csv"

        # Modeli yükle veya oluştur
        self.model = MobileGoalAlert()

        if os.path.exists(data_file):
            self.model.load_data(data_file)
            self.model.train_model()
        else:
            print(f"⚠️ Veri dosyası bulunamadı: {data_file}")
            print("   Örnek veri ile devam ediliyor...")
            self.create_sample_data()
            self.model.train_model()

        self.simulator = MatchSimulator(self.model)

        # Ana menü
        self.main_menu()

    def create_sample_data(self):
        """Örnek veri oluştur"""
        print("📊 Örnek veri oluşturuluyor...")

        # 1000 örnek maç verisi
        np.random.seed(42)
        n = 1000

        data = {
            'Home_Pos': np.random.randint(1, 20, n),
            'Away_Pos': np.random.randint(1, 20, n),
            'H_Score': np.random.randint(0, 3, n),
            'A_Score': np.random.randint(0, 3, n),
            'H_Momentum': np.random.randint(10, 90, n),
            'A_Momentum': np.random.randint(10, 90, n),
            'H_xG': np.random.uniform(0.1, 2.5, n),
            'A_xG': np.random.uniform(0.1, 2.5, n),
            'H_SOT': np.random.randint(0, 15, n),
            'A_SOT': np.random.randint(0, 15, n),
            'H_Corners': np.random.randint(0, 12, n),
            'A_Corners': np.random.randint(0, 12, n),
            'H_Attacks': np.random.randint(5, 80, n),
            'A_Attacks': np.random.randint(5, 80, n),
            'H_Dn_Attacks': np.random.randint(0, 40, n),
            'A_Dn_Attacks': np.random.randint(0, 40, n),
            'H_Poss': np.random.randint(20, 80, n),
            'A_Poss': np.random.randint(20, 80, n),
            'H_Y_Cards': np.random.randint(0, 4, n),
            'A_Y_Cards': np.random.randint(0, 4, n),
            'Goal Times': [f"{np.random.randint(5, 90)}" for _ in range(n)],
            'Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S') for _ in range(n)],
        }

        self.raw_data = pd.DataFrame(data)
        self.model.raw_data = self.raw_data
        self.model.process_data()

    def main_menu(self):
        """Ana menü"""
        while True:
            print("\n" + "="*50)
            print("   📱 ANA MENÜ")
            print("="*50)
            print("1️⃣  Canlı Maç Tahmini (Simülasyon)")
            print("2️⃣  Maç Verisi Gir (Manuel)")
            print("3️⃣  Model Performansı")
            print("4️⃣  Simülasyon Çalıştır")
            print("5️⃣  Çıkış")

            choice = input("\nSeçiminiz (1-5): ").strip()

            if choice == '1':
                self.live_match_simulation()
            elif choice == '2':
                self.manual_match_input()
            elif choice == '3':
                self.show_performance()
            elif choice == '4':
                self.run_batch_simulation()
            elif choice == '5':
                print("\n👋 Çıkış yapılıyor...")
                break
            else:
                print("❌ Geçersiz seçim!")

    def live_match_simulation(self):
        """Canlı maç simülasyonu"""
        print("\n" + "="*50)
        print("   🟢 CANLI MAÇ TAHMİNİ")
        print("="*50)

        match_id = input("\nMaç ID (boş bırak=1): ").strip()
        match_id = int(match_id) if match_id else 1

        results = self.simulator.simulate_match(match_id)

        if results:
            print("\n" + "="*50)
            print("   📊 TAHMİN ÖZETİ")
            print("="*50)

            for r in results:
                status = "🔔 ALARM" if r['alert'] else "⏸️ BEKLE"
                print(f"\n{r['scenario']}:")
                print(f"  {status} | Olasılık: {r['probability']}% | Güven: {r['confidence']}/100")
                print(f"  ⏱️ Beklenen Gol: {r['expected_goal_minute']}. dakika")

    def manual_match_input(self):
        """Manuel maç verisi gir"""
        print("\n" + "="*50)
        print("   ✏️ MANUEL MAÇ VERİSİ")
        print("="*50)

        print("\nLütfen maç istatistiklerini girin:")
        print("(Boş bırakılan alanlar varsayılan değer alır)")

        data = {
            'Home_Pos': int(input("Ev Takımı Sıralaması (1-20): ") or 10),
            'Away_Pos': int(input("Deplasman Sıralaması (1-20): ") or 10),
            'H_Score': int(input("Ev Gol: ") or 0),
            'A_Score': int(input("Deplasman Gol: ") or 0),
            'H_Momentum': int(input("Ev Takımı Momentum (0-100): ") or 50),
            'A_Momentum': int(input("Deplasman Momentum (0-100): ") or 50),
            'H_xG': float(input("Ev xG: ") or 0.5),
            'A_xG': float(input("Deplasman xG: ") or 0.5),
            'H_SOT': int(input("Ev İsabetli Şut: ") or 2),
            'A_SOT': int(input("Deplasman İsabetli Şut: ") or 2),
            'H_Corners': int(input("Ev Korner: ") or 3),
            'A_Corners': int(input("Deplasman Korner: ") or 3),
            'H_Attacks': int(input("Ev Atak: ") or 30),
            'A_Attacks': int(input("Deplasman Atak: ") or 30),
            'H_Dn_Attacks': int(input("Ev Tehlikeli Atak: ") or 10),
            'A_Dn_Attacks': int(input("Deplasman Tehlikeli Atak: ") or 10),
            'H_Poss': int(input("Ev Topa Sahip Olma (%): ") or 50),
            'A_Poss': int(input("Deplasman Topa Sahip Olma (%): ") or 50),
            'H_Y_Cards': int(input("Ev Sarı Kart: ") or 0),
            'A_Y_Cards': int(input("Deplasman Sarı Kart: ") or 0),
        }

        # Oranlar (varsayılan)
        data.update({
            'Odds_Home': 2.0,
            'Odds_Draw': 3.5,
            'Odds_Away': 3.5,
            'Odds_Over_0.5': 1.3,
            'Odds_Under_0.5': 3.5,
            'PM_Odds_Home': 2.1,
            'PM_Odds_Draw': 3.4,
            'PM_Odds_Away': 3.4,
            'PM_Odds_Over_0.5': 1.25,
            'PM_Odds_Under_0.5': 3.6,
        })

        result = self.model.predict_match(data)

        if result:
            self.display_result(result)

    def display_result(self, result):
        """Sonucu göster"""
        print("\n" + "="*50)
        print("   🎯 TAHMİN SONUCU")
        print("="*50)

        # Alarm durumu
        if result['alert']:
            print("\n🔔 **İLK GOL ALARMI VER**")
        else:
            print("\n⏸️ **ALARM VERME**")

        # Detaylar
        print(f"\n📊 İlk Gol Olasılığı: {result['probability']}%")
        print(f"🎯 Güven Skoru: {result['confidence']}/100")
        print(f"⚠️ Risk Skoru: {result['risk']}/100")

        # Risk seviyesi
        if result['risk'] < 33:
            risk_level = "🟢 Düşük"
        elif result['risk'] < 66:
            risk_level = "🟡 Orta"
        else:
            risk_level = "🔴 Yüksek"
        print(f"📈 Risk Seviyesi: {risk_level}")

        print(f"\n⏱️ Beklenen İlk Gol: {result['expected_goal_minute']}. dakika")
        print(f"📌 Alarm Eşiği: {result['threshold']}%")

        # Benzer maçlar
        print(f"\n📊 Benzer Maç Sayısı: {result['similar_matches_count']}")
        print(f"📈 Benzer Maç Başarısı: {result['similar_success_rate']*100:.1f}%")

        print("\n" + "-"*50)

    def show_performance(self):
        """Model performansını göster"""
        print("\n" + "="*50)
        print("   📊 MODEL PERFORMANSI")
        print("="*50)

        print(f"\n📌 Model Versiyonu: {self.model.model_version}")
        print(f"📅 Son Eğitim: {self.model.last_training_date}")
        print(f"🎯 Alarm Eşiği: {self.model.threshold*100:.1f}%")
        print(f"📊 Özellik Sayısı: {len(self.model.feature_columns)}")

        if hasattr(self.model, 'feature_importances'):
            print("\n🔝 En Önemli Özellikler:")
            importances = self.model.feature_importances
            for idx in np.argsort(importances)[-5:][::-1]:
                print(f"  - {self.model.feature_columns[idx]}: {importances[idx]:.3f}")

        print("\n💡 Öneri:")
        print("  - Yeni maç verileri geldikçe modeli yeniden eğitin")
        print("  - Veri arttıkça doğruluk artacaktır")

    def run_batch_simulation(self):
        """Toplu simülasyon çalıştır"""
        print("\n" + "="*50)
        print("   🔄 TOPLU SİMÜLASYON")
        print("="*50)

        n = input("\nKaç maç simüle edilsin? (10-100): ").strip()
        n = int(n) if n else 10
        n = max(10, min(100, n))

        print(f"\n🎲 {n} maç simüle ediliyor...")

        results = []
        for i in range(1, n+1):
            sim_results = self.simulator.simulate_match(i)
            if sim_results:
                # En yüksek olasılıklı senaryoyu al
                best = max(sim_results, key=lambda x: x['probability'])
                results.append(best)
                print(f"  Maç {i}: {best['probability']}% - {'🔔' if best['alert'] else '⏸️'}")

        # Özet
        if results:
            alert_count = sum(1 for r in results if r['alert'])
            avg_prob = np.mean([r['probability'] for r in results])
            avg_conf = np.mean([r['confidence'] for r in results])

            print(f"\n📊 ÖZET:")
            print(f"  Toplam Maç: {len(results)}")
            print(f"  Alarm Verilen: {alert_count} ({alert_count/len(results)*100:.1f}%)")
            print(f"  Ortalama Olasılık: {avg_prob:.1f}%")
            print(f"  Ortalama Güven: {avg_conf:.1f}/100")


# ==================== UYGULAMAYI BAŞLAT ====================

if __name__ == "__main__":
    app = MobileApp()
    app.start()
