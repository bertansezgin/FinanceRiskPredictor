"""
Sistem Test Script - Hızlı kontrol için
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

def test_imports():
    """Kütüphaneleri test et"""
    print("📚 Kütüphaneler kontrol ediliyor...")
    
    libraries = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'catboost': 'catboost',
        'optuna': 'optuna',
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'joblib': 'joblib',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing = []
    for name, module in libraries.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - Yüklenmesi gerekiyor")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️ Eksik kütüphaneler: {', '.join(missing)}")
        print("Yüklemek için: pip install -r requirements.txt")
        return False
    
    print("\n✅ Tüm kütüphaneler yüklü!")
    return True


def test_data_files():
    """Veri dosyalarını kontrol et"""
    print("\n📁 Veri dosyaları kontrol ediliyor...")
    
    data_files = [
        'data/birlesik_risk_verisi.csv',
        'data/yeni_musteri.csv'
    ]
    
    for file in data_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            print(f"  ✅ {file} - {len(df)} satır")
        else:
            print(f"  ❌ {file} - Bulunamadı")
            return False
    
    print("\n✅ Tüm veri dosyaları mevcut!")
    return True


def test_basic_pipeline():
    """Basit pipeline test"""
    print("\n🔧 Basit pipeline test ediliyor...")
    
    try:
        from src.loader import load_data
        from src.preprocessing import clean_data, generate_features
        
        # Küçük örnek veri oluştur
        sample_data = pd.DataFrame({
            'ProjectId': range(100),
            'InstallmentCount': np.random.randint(1, 36, 100),
            'RemainingPrincipalAmount': np.random.uniform(0, 10000, 100),
            'AmountTL': np.random.uniform(0, 5000, 100),
            'PrincipalAmount': np.random.uniform(1000, 20000, 100),
            'TranDate': pd.date_range('2024-01-01', periods=100, freq='D'),
            'MaturityDate': pd.date_range('2024-02-01', periods=100, freq='D')
        })
        
        # Preprocessing test
        df_clean = clean_data(sample_data)
        df_features = generate_features(df_clean)
        
        print(f"  ✅ Preprocessing çalışıyor")
        print(f"  ✅ {len(df_features.columns)} özellik oluşturuldu")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Hata: {str(e)}")
        return False


def test_advanced_models():
    """Gelişmiş modelleri test et"""
    print("\n🚀 Gelişmiş modeller test ediliyor...")
    
    try:
        from src.advanced_models import AdvancedRiskModels
        
        # Örnek veri
        X_train = pd.DataFrame(np.random.randn(100, 6))
        y_train = np.random.uniform(0, 100, 100)
        X_test = pd.DataFrame(np.random.randn(20, 6))
        y_test = np.random.uniform(0, 100, 20)
        
        # Model test
        models = AdvancedRiskModels()
        print("  ✅ AdvancedRiskModels yüklendi")
        print(f"  ✅ {len(models.models)} model hazır")
        
        # Hızlı bir model eğit
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        score = rf.score(X_test, y_test)
        print(f"  ✅ Test model R2: {score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Hata: {str(e)}")
        return False


def test_automl():
    """AutoML sistemi test et"""
    print("\n🤖 AutoML sistemi test ediliyor...")
    
    try:
        from src.automl_system import AutoMLPipeline
        
        # AutoML pipeline oluştur
        automl = AutoMLPipeline(optimize_hyperparams=False)
        print("  ✅ AutoML pipeline oluşturuldu")
        
        # Küçük test verisi
        test_df = pd.DataFrame({
            'ProjectId': range(50),
            'OverdueDays': np.random.randint(0, 30, 50),
            'EksikOdemeOrani': np.random.uniform(0, 1, 50),
            'KalanOran': np.random.uniform(0, 1, 50),
            'OdenmediMi': np.random.choice([0, 1], 50),
            'InstallmentCount': np.random.randint(1, 36, 50),
            'OrtalamaOdeme': np.random.uniform(100, 5000, 50)
        })
        
        X, y = automl.prepare_data(test_df)
        print(f"  ✅ Veri hazırlandı: {X.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Hata: {str(e)}")
        return False


def run_all_tests():
    """Tüm testleri çalıştır"""
    print("="*60)
    print("🔍 SİSTEM TEST")
    print("="*60)
    print(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Testleri çalıştır
    results.append(("Kütüphaneler", test_imports()))
    results.append(("Veri Dosyaları", test_data_files()))
    results.append(("Basit Pipeline", test_basic_pipeline()))
    results.append(("Gelişmiş Modeller", test_advanced_models()))
    results.append(("AutoML Sistemi", test_automl()))
    
    # Sonuçları özetle
    print("\n" + "="*60)
    print("📊 TEST SONUÇLARI")
    print("="*60)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        print(f"{test_name:<20} {status}")
        if not result:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 TÜM TESTLER BAŞARILI!")
        print("\nSistem kullanıma hazır. Çalıştırmak için:")
        print("  - Basit: python main.py")
        print("  - Gelişmiş: python main_advanced.py")
        print("  - Web UI: streamlit run streamlit_app.py")
    else:
        print("\n⚠️ Bazı testler başarısız oldu.")
        print("Lütfen eksik bileşenleri yükleyin ve tekrar deneyin.")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)