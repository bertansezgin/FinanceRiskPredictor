 # 🏦 Gelişmiş Finansal Risk Tahmin Sistemi

## 📋 Proje Hakkında

Bu proje, gelişmiş makine öğrenmesi teknikleri kullanarak finansal risk tahmini yapan kapsamlı bir sistemdir. AutoML pipeline'ı, hyperparameter optimizasyonu ve web arayüzü ile donatılmıştır.

## 🚀 Özellikler

### 1. **Gelişmiş Makine Öğrenmesi Modelleri**
- Linear Models (Ridge, Lasso, ElasticNet)
- Tree-based Models (Random Forest, Decision Tree)
- Boosting Algorithms (XGBoost, LightGBM, CatBoost, Gradient Boosting)
- Neural Networks (MLP Regressor)
- Support Vector Machines (SVR)
- K-Nearest Neighbors (KNN)

### 2. **Gelişmiş Feature Engineering**
- Finansal risk özellikleri
- Zaman bazlı özellikler
- İstatistiksel özellikler
- Polinomsal özellikler
- Etkileşim terimleri
- Otomatik özellik seçimi

### 3. **Hyperparameter Optimization**
- Optuna entegrasyonu
- Bayesian optimization
- Cross-validation
- Paralel işleme desteği

### 4. **AutoML Pipeline**
- Tam otomatik model eğitimi
- Otomatik feature engineering
- Model karşılaştırma
- En iyi model seçimi

### 5. **Model Değerlendirme**
- Kapsamlı metrikler (R², RMSE, MAE, MAPE)
- Cross-validation skorları
- Overfitting analizi
- Learning curves
- Feature importance analizi

### 6. **Web Arayüzü (Streamlit)**
- Kullanıcı dostu arayüz
- Gerçek zamanlı tahmin
- Model performans görselleştirmeleri
- Veri yükleme ve yönetimi

## 📁 Proje Yapısı

```
FinanceRiskPredictor/
├── data/                           # Veri dosyaları
│   ├── birlesik_risk_verisi.csv
│   └── yeni_musteri.csv
├── src/                           # Kaynak kodlar
│   ├── __init__.py
│   ├── loader.py                 # Veri yükleme
│   ├── preprocessing.py          # Temel ön işleme
│   ├── risk_model.py            # Basit model
│   ├── predict.py               # Tahmin
│   ├── advanced_models.py      # Gelişmiş modeller
│   ├── feature_engineering.py  # Feature engineering
│   ├── model_evaluation.py     # Model değerlendirme
│   ├── hyperparameter_tuning.py # Hyperparameter optimization
│   └── automl_system.py        # AutoML pipeline
├── models/                      # Kaydedilmiş modeller
│   └── automl/                 # AutoML modelleri
├── reports/                     # Raporlar
├── plots/                      # Görselleştirmeler
├── main.py                     # Basit ana program
├── main_advanced.py           # Gelişmiş ana program
├── streamlit_app.py          # Web arayüzü
├── requirements.txt          # Gereksinimler
└── README.md                # Bu dosya
```

## 🛠️ Kurulum

### 1. Gerekli Kütüphaneleri Yükleyin

```bash
pip install -r requirements.txt
```

### 2. Temel Gereksinimlerin Kurulumu

Eğer bazı kütüphaneler yüklenmezse:

```bash
# XGBoost
pip install xgboost

# LightGBM
pip install lightgbm

# CatBoost
pip install catboost

# Optuna
pip install optuna

# Streamlit
pip install streamlit

# Plotly
pip install plotly
```

## 🎯 Kullanım

### 1. Basit Kullanım (Eski Sistem)

```bash
python main.py
```

### 2. Gelişmiş Sistem

```bash
python main_advanced.py
```

Seçenekler:
- **1**: Hızlı Model Eğitimi (Hyperparameter tuning olmadan)
- **2**: Optimize Edilmiş Model Eğitimi (Hyperparameter tuning ile)
- **3**: Özelleştirilmiş Pipeline (Detaylı analiz ve görselleştirme)
- **4**: Tüm Seçenekleri Çalıştır

### 3. Web Arayüzü

```bash
streamlit run streamlit_app.py
```

Web arayüzü özellikleri:
- 🏠 **Ana Sayfa**: Sistem özeti ve metrikler
- 📊 **Model Eğitimi**: İnteraktif model eğitimi
- 🔮 **Risk Tahmini**: Tekil ve toplu tahmin
- 📈 **Model Performansı**: Detaylı performans analizi
- 📁 **Veri Yükleme**: Veri yönetimi

## 📊 Model Performansı

Mevcut sistem ile elde edilen tipik performans değerleri:

| Model | Test R² | RMSE | MAE |
|-------|---------|------|-----|
| XGBoost | 0.94 | 3.2 | 2.1 |
| LightGBM | 0.93 | 3.5 | 2.3 |
| CatBoost | 0.92 | 3.8 | 2.5 |
| Random Forest | 0.90 | 4.2 | 2.8 |

## 🔄 Geliştirme Önerileri

### Kısa Vadeli Geliştirmeler
1. **Deep Learning Modelleri**: LSTM, GRU gibi modeller eklenebilir
2. **Explainable AI**: SHAP, LIME entegrasyonu
3. **Real-time Monitoring**: Kafka entegrasyonu
4. **API Endpoint**: FastAPI ile REST API

### Uzun Vadeli Geliştirmeler
1. **MLOps Pipeline**: MLflow, Kubeflow entegrasyonu
2. **Distributed Training**: Spark ML entegrasyonu
3. **AutoML Geliştirilmesi**: H2O.ai, AutoGluon entegrasyonu
4. **Cloud Deployment**: AWS SageMaker, Azure ML

## 📈 Performans İpuçları

1. **Veri Kalitesi**: Eksik verileri düzgün işleyin
2. **Feature Selection**: Gereksiz özellikleri kaldırın
3. **Cross-validation**: En az 5-fold kullanın
4. **Hyperparameter Tuning**: En az 50 trial kullanın
5. **Ensemble Methods**: Farklı modelleri birleştirin

## 🐛 Sorun Giderme

### LightGBM Kurulum Sorunu
```bash
brew install libomp  # macOS için
```

### CatBoost GPU Desteği
```bash
pip install catboost[gpu]
```

### Streamlit Port Sorunu
```bash
streamlit run streamlit_app.py --server.port 8502
```


---

**Not**: Bu sistem finansal tavsiye vermez. Gerçek finansal kararlar için uzman görüşü alınmalıdır.