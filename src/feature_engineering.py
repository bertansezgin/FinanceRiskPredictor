"""
Gelişmiş Feature Engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineering:
    """Gelişmiş özellik mühendisliği"""
    
    def __init__(self):
        self.scaler = None
        self.poly_features = None
        self.pca = None
        self.feature_selector = None
        self.feature_names = []
        
    def create_advanced_features(self, df):
        """Gelişmiş özellikler oluştur"""
        
        df = df.copy()
        
        # Mevcut temel özellikleri koru
        df = self._create_basic_features(df)
        
        # Finansal risk özellikleri
        df = self._create_financial_risk_features(df)
        
        # Zaman bazlı özellikler
        df = self._create_temporal_features(df)
        
        # İstatistiksel özellikler
        df = self._create_statistical_features(df)
        
        # Kategorik değişken encoding
        df = self._encode_categorical_features(df)
        
        return df
    
    def _create_basic_features(self, df):
        """Temel özellikler"""
        
        # Güvenli doldurma
        df['InstallmentCount'] = df['InstallmentCount'].fillna(1)
        df['RemainingPrincipalAmount'] = df['RemainingPrincipalAmount'].fillna(0)
        df['AmountTL'] = df['AmountTL'].fillna(0)
        df['PrincipalAmount'] = df['PrincipalAmount'].fillna(1)
        df['FundingAmount'] = df['FundingAmount'].fillna(0)
        
        # Temel oranlar
        df['OrtalamaOdeme'] = df['AmountTL'] / df['InstallmentCount'].replace(0, 1)
        df['KalanOran'] = df['RemainingPrincipalAmount'] / df['PrincipalAmount'].replace(0, 1)
        df['KalanOran'] = df['KalanOran'].clip(0, 1)
        
        # Eksik ödeme oranı
        df['EksikOdemeOrani'] = 1 - (df['AmountTL'] / df['PrincipalAmount'].replace(0, 1))
        df['EksikOdemeOrani'] = df['EksikOdemeOrani'].clip(0, 1)
        
        # Ödeme durumu
        df['OdenmediMi'] = (df['AmountTL'] == 0).astype(float)
        
        # Gecikme günleri
        if 'TranDate' in df.columns and 'MaturityDate' in df.columns:
            df['TranDate'] = pd.to_datetime(df['TranDate'], errors='coerce')
            df['MaturityDate'] = pd.to_datetime(df['MaturityDate'], errors='coerce')
            df['OverdueDays'] = (df['TranDate'] - df['MaturityDate']).dt.days
            df['OverdueDays'] = df['OverdueDays'].apply(lambda x: max(x, 0) if pd.notnull(x) else 0)
            df['VaktindenOnceOdeme'] = (df['TranDate'] < df['MaturityDate']).astype(int)
        else:
            df['OverdueDays'] = 0
            df['VaktindenOnceOdeme'] = 0
        
        return df
    
    def _create_financial_risk_features(self, df):
        """Finansal risk özellikleri"""
        
        # Kredi kullanım oranı
        df['KrediKullanimOrani'] = df['AmountTL'] / df['FundingAmount'].replace(0, 1)
        df['KrediKullanimOrani'] = df['KrediKullanimOrani'].clip(0, 2)
        
        # Taksit başına düşen ana para
        df['TaksitBasinaAnapara'] = df['PrincipalAmount'] / df['InstallmentCount'].replace(0, 1)
        
        # Ödeme düzenliliği skoru
        df['OdemeDuzenlilik'] = np.where(df['OverdueDays'] == 0, 1,
                                          np.where(df['OverdueDays'] <= 7, 0.8,
                                                   np.where(df['OverdueDays'] <= 30, 0.5,
                                                            np.where(df['OverdueDays'] <= 90, 0.2, 0))))
        
        # Risk kategorisi
        df['RiskKategorisi'] = np.where(df['OverdueDays'] == 0, 0,  # Risksiz
                                         np.where(df['OverdueDays'] <= 30, 1,  # Düşük risk
                                                  np.where(df['OverdueDays'] <= 90, 2,  # Orta risk
                                                           3)))  # Yüksek risk
        
        # Finansal yük indeksi
        df['FinansalYukIndeksi'] = (df['RemainingPrincipalAmount'] * df['InstallmentCount']) / \
                                    df['FundingAmount'].replace(0, 1)
        
        # Erken ödeme eğilimi
        df['ErkenOdemeEgilimi'] = df.get('VaktindenOnceOdeme', 0) * \
                                   (1 - df['EksikOdemeOrani'])
        
        # Toplam borç yükü
        df['ToplamBorcYuku'] = df['RemainingPrincipalAmount'] + \
                                (df['RemainingPrincipalAmount'] * 0.1)  # Tahmini faiz
        
        # Ödeme gücü skoru
        df['OdemeGucuSkoru'] = (df['AmountTL'] / df['TaksitBasinaAnapara'].replace(0, 1)).clip(0, 2)
        
        # Temerrüt riski skoru
        df['TemerrütRiskSkoru'] = (df['OverdueDays'] * 0.3 + 
                                    df['EksikOdemeOrani'] * 100 * 0.4 +
                                    df['OdenmediMi'] * 100 * 0.3)
        
        return df
    
    def _create_temporal_features(self, df):
        """Zaman bazlı özellikler"""
        
        if 'TranDate' in df.columns:
            df['TranDate'] = pd.to_datetime(df['TranDate'], errors='coerce')
            
            # Ay, çeyrek yıl, yıl
            df['OdemeAyi'] = df['TranDate'].dt.month
            df['OdemeCeyregi'] = df['TranDate'].dt.quarter
            df['OdemeYili'] = df['TranDate'].dt.year
            df['OdemeGunu'] = df['TranDate'].dt.day
            df['HaftaninGunu'] = df['TranDate'].dt.dayofweek
            
            # Ay sonu mu?
            df['AySonuMu'] = (df['TranDate'].dt.day >= 25).astype(int)
            
            # Hafta sonu mu?
            df['HaftaSonuMu'] = df['HaftaninGunu'].isin([5, 6]).astype(int)
            
            # Mevsim
            df['Mevsim'] = df['OdemeAyi'].apply(lambda x: 
                                                  1 if x in [12, 1, 2] else  # Kış
                                                  2 if x in [3, 4, 5] else    # İlkbahar
                                                  3 if x in [6, 7, 8] else    # Yaz
                                                  4)                          # Sonbahar
        
        if 'MaturityDate' in df.columns:
            df['MaturityDate'] = pd.to_datetime(df['MaturityDate'], errors='coerce')
            
            # Vade ayı özellikleri
            df['VadeAyi'] = df['MaturityDate'].dt.month
            df['VadeCeyregi'] = df['MaturityDate'].dt.quarter
            
        # Ödeme ve vade arasındaki gün farkı
        if 'TranDate' in df.columns and 'MaturityDate' in df.columns:
            df['OdemeVadeFarki'] = (df['TranDate'] - df['MaturityDate']).dt.days
            
        return df
    
    def _create_statistical_features(self, df):
        """İstatistiksel özellikler"""
        
        # Log dönüşümleri (sıfırdan büyük değerler için)
        for col in ['AmountTL', 'PrincipalAmount', 'FundingAmount', 'RemainingPrincipalAmount']:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
        
        # Kare ve küp özellikler
        df['OverdueDays_squared'] = df['OverdueDays'] ** 2
        df['OverdueDays_cubed'] = df['OverdueDays'] ** 3
        df['EksikOdemeOrani_squared'] = df['EksikOdemeOrani'] ** 2
        
        # Etkileşim terimleri
        df['OverdueDays_x_EksikOdeme'] = df['OverdueDays'] * df['EksikOdemeOrani']
        df['KalanOran_x_InstallmentCount'] = df['KalanOran'] * df['InstallmentCount']
        df['OdenmediMi_x_OverdueDays'] = df['OdenmediMi'] * df['OverdueDays']
        
        # Bölme özellikleri
        df['Amount_per_Installment'] = df['AmountTL'] / df['InstallmentCount'].replace(0, 1)
        df['Remaining_per_Total'] = df['RemainingPrincipalAmount'] / df['PrincipalAmount'].replace(0, 1)
        
        # Çarpım özellikleri
        df['Total_Risk_Score'] = df['TemerrütRiskSkoru'] * df['RiskKategorisi']
        
        return df
    
    def _encode_categorical_features(self, df):
        """Kategorik değişkenleri encode et"""
        
        categorical_columns = ['ProductCode', 'PersonType', 'PortfolioClass', 'AgreementType']
        
        for col in categorical_columns:
            if col in df.columns:
                # Frequency encoding
                freq_encoding = df[col].value_counts(normalize=True)
                df[f'{col}_freq'] = df[col].map(freq_encoding).fillna(0)
                
                # Target encoding için hazırlık (gerçek uygulamada target ile yapılmalı)
                # df[f'{col}_target_enc'] = ...
        
        return df
    
    def create_polynomial_features(self, X, degree=2):
        """Polinomsal özellikler oluştur"""
        
        # Sadece sayısal özellikleri seç
        numeric_features = ['OverdueDays', 'EksikOdemeOrani', 'KalanOran', 
                            'OdenmediMi', 'InstallmentCount', 'OrtalamaOdeme']
        
        X_numeric = X[numeric_features]
        
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = self.poly_features.fit_transform(X_numeric)
        
        # Yeni özellik isimleri
        poly_feature_names = self.poly_features.get_feature_names_out(numeric_features)
        
        # DataFrame'e çevir
        X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
        
        # Orijinal özelliklerle birleştir
        X_combined = pd.concat([X, X_poly_df], axis=1)
        
        # Duplicate sütunları kaldır
        X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]
        
        return X_combined
    
    def scale_features(self, X, method='robust'):
        """Özellikleri ölçeklendir"""
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Bilinmeyen ölçeklendirme metodu: {method}")
        
        X_scaled = self.scaler.fit_transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def select_best_features(self, X, y, method='mutual_info', k=20):
        """En iyi özellikleri seç"""
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        elif method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=k)
        else:
            raise ValueError(f"Bilinmeyen özellik seçim metodu: {method}")
        
        X_selected = selector.fit_transform(X, y)
        
        # Seçilen özellik isimleri
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.feature_selector = selector
        self.feature_names = selected_features
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def apply_pca(self, X, n_components=0.95):
        """PCA uygula"""
        
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        # PCA bileşen isimleri
        pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        
        return pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
    
    def get_feature_importance_from_model(self, model, feature_names):
        """Model bazlı özellik önem analizi"""
        
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance
        else:
            return None