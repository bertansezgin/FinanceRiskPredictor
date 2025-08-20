"""
DATA LEAKAGE TEMİZLENMİŞ Feature Engineering
Sadece kredi verilirken bilinen değişkenler kullanılır
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineering:
    """Data leakage temizlenmiş özellik mühendisliği

    PRENSIP: Sadece kredi verilirken bilinen bilgiler kullanılır
    YASAK: Ödeme performansı, kalan borç, gerçek ödeme tarihleri
    """

    def __init__(self):
        self.scaler = None
        self.feature_names = []
        self.poly_transformer = None

    def create_advanced_features(self, df):
        """Temiz özellikler oluştur - Data leakage önlenmiş"""

        df = df.copy()

        # SADECE BAŞLANGIÇTA BİLİNEN ÖZELLIKLER
        df = self._create_clean_basic_features(df)

        # Türetilmiş özellikler (Temiz kaynaklardan)
        df = self._create_clean_derived_features(df)

        # Zaman özellikleri (Sadece planlanan tarihler)
        df = self._create_clean_temporal_features(df)

        return df

    def _create_clean_basic_features(self, df):
        """Temel özellikler - SADECE KREDİ BAŞLANGICINDA BİLİNEN"""

        # YASAK: Bu sütunları ASLA kullanma!
        from src.config import config
        forbidden_columns = config.LEAKAGE_COLUMNS

        # LEAKAGE SÜTUNLARINI SİL!
        leakage_found = []
        for col in forbidden_columns:
            if col in df.columns:
                leakage_found.append(col)
                print(f"🚨 SİLİNİYOR: {col} sütunu LEAKAGE riski!")

        # Leakage sütunlarını DataFrame'den çıkar
        if leakage_found:
            df = df.drop(columns=leakage_found)
            print(f"✅ {len(leakage_found)} leakage sütunu silindi")

        # SAFE: Kredi başvuru bilgileri
        df['InstallmentCount'] = df['InstallmentCount'].fillna(12).astype(float)
        df['PrincipalAmount'] = df['PrincipalAmount'].fillna(1000).astype(float)
        df['FundingAmount'] = df['FundingAmount'].fillna(1000).astype(float)
        df['MonthlyProfitRate'] = df['MonthlyProfitRate'].fillna(1.0).astype(float)

        # SAFE: Planlanan tarihler
        if 'ProjectDate' in df.columns:
            df['ProjectDate'] = pd.to_datetime(df['ProjectDate'], errors='coerce')
            df['KrediAyi'] = df['ProjectDate'].dt.month
            df['KrediCeyregi'] = df['ProjectDate'].dt.quarter
            df['KrediYili'] = df['ProjectDate'].dt.year

        # SAFE: Hesaplanabilir oranlar
        df['TaksitBasinaAnapara'] = df['PrincipalAmount'] / df['InstallmentCount'].replace(0, 1)
        df['FonlamaOrani'] = df['FundingAmount'] / df['PrincipalAmount'].replace(0, 1)
        df['FonlamaOrani'] = df['FonlamaOrani'].clip(0.5, 1.5)

        # SAFE: Tahmini aylık ödeme
        df['TahminiAylikOdeme'] = df['TaksitBasinaAnapara'] * (1 + df['MonthlyProfitRate']/100)

        # SAFE: Kategorik özellikler
        df['KrediTutarKategorisi'] = pd.cut(
            df['PrincipalAmount'],
            bins=[0, 10000, 30000, 100000, float('inf')],
            labels=[0, 1, 2, 3]
        ).astype(float).fillna(1)

        df['TaksitSayisiKategorisi'] = pd.cut(
            df['InstallmentCount'],
            bins=[0, 6, 12, 24, 36, float('inf')],
            labels=[0, 1, 2, 3, 4]
        ).astype(float).fillna(2)

        df['FaizOraniKategorisi'] = pd.cut(
            df['MonthlyProfitRate'],
            bins=[0, 1, 2, 3, float('inf')],
            labels=[0, 1, 2, 3]  # 0: Düşük faiz, 3: Yüksek faiz
        ).astype(float).fillna(1)

        return df

    def _create_clean_derived_features(self, df):
        """Türetilmiş özellikler - Sadece temiz değişkenlerden"""

        # Log dönüşümleri - Sadece temiz değişkenler
        for col in ['PrincipalAmount', 'FundingAmount', 'TahminiAylikOdeme']:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col].clip(lower=1))

        # Kredi risk göstergeleri (Sadece başlangıç bilgileri)
        df['KrediTutar_Taksit_Interaksiyon'] = df['PrincipalAmount'] * df['InstallmentCount']
        df['Faiz_Vade_Etkisi'] = df['MonthlyProfitRate'] * df['InstallmentCount']
        df['OdemeYuku_Oran'] = df['TahminiAylikOdeme'] / df['PrincipalAmount'].replace(0, 1)

        # Üst seviye risk skorları (Başlangıç değerleri)
        # Yüksek taksit sayısı + yüksek faiz = risk
        df['VadeRiskSkoru'] = df['TaksitSayisiKategorisi'] * df['FaizOraniKategorisi']

        # Büyük kredi + uzun vade = risk
        df['KrediRiskSkoru'] = df['KrediTutarKategorisi'] * df['TaksitSayisiKategorisi']

        # Matematiksel dönüşümler
        df['PrincipalAmount_sqrt'] = np.sqrt(df['PrincipalAmount'].clip(lower=0))
        df['InstallmentCount_square'] = df['InstallmentCount'] ** 2

        # Ürün türü kodlaması (Kategorik değişken)
        if 'ProductCode' in df.columns:
            product_dummies = pd.get_dummies(df['ProductCode'], prefix='Product')
            df = pd.concat([df, product_dummies], axis=1)

        # Şube bilgisi (Coğrafi risk)
        if 'BranchId' in df.columns:
            # Basit şube kategorisi (daha güvenli)
            branch_values = df['BranchId'].fillna(1)
            df['BranchCategory'] = (branch_values % 5).astype(float)  # 0-4 arası kategoriler

        return df

    def _create_clean_temporal_features(self, df):
        """Zaman özellikleri - Sadece planlanan tarihler"""

        # Kredi başlangıç tarihi özellikleri
        if 'ProjectDate' in df.columns:
            df['ProjectDate'] = pd.to_datetime(df['ProjectDate'], errors='coerce')

            # Kredi başlangıç ayı (Mevsimsel risk)
            df['KrediAyi'] = df['ProjectDate'].dt.month
            df['KrediCeyregi'] = df['ProjectDate'].dt.quarter
            df['KrediYili'] = df['ProjectDate'].dt.year

            # Ay sonu etkisi (Ödeme zorlukları)
            df['AySonuKredi'] = (df['ProjectDate'].dt.day >= 25).astype(int)

            # Hafta sonu kredisi (Risk göstergesi olabilir)
            df['HaftaSonuKredi'] = df['ProjectDate'].dt.dayofweek.isin([5, 6]).astype(int)

            # Mevsimsel özellikler
            df['YazKredisi'] = df['KrediAyi'].isin([6, 7, 8]).astype(int)  # Yaz ayları
            df['KisKredisi'] = df['KrediAyi'].isin([12, 1, 2]).astype(int)  # Kış ayları

        # İlk taksit tarihi özellikleri (Sadece planlanan)
        if 'FirstInstallmentDate' in df.columns:
            df['FirstInstallmentDate'] = pd.to_datetime(df['FirstInstallmentDate'], errors='coerce')

            # İlk ödeme ayı
            df['IlkOdemeAyi'] = df['FirstInstallmentDate'].dt.month
            df['IlkOdemeAySonu'] = (df['FirstInstallmentDate'].dt.day >= 25).astype(int)

        return df



    def create_polynomial_features(self, X, degree=2):
        """Polynomial özellikler oluştur - EKSİK OLAN FONKSİYON"""

        if self.poly_transformer is None:
            # Sadece en önemli sütunlarla polynomial features oluştur
            # Çok fazla feature oluşmasını önle
            important_cols = []

            # Sayısal sütunları seç
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

            # En önemlileri seç (max 10 sütun)
            priority_keywords = ['Amount', 'Rate', 'Count', 'Kategori', 'Skor', 'log']

            for col in numeric_cols:
                if any(keyword in col for keyword in priority_keywords):
                    important_cols.append(col)

            # En fazla 10 sütun seç
            important_cols = important_cols[:10] if len(important_cols) > 10 else important_cols

            # Hiç uygun sütun yoksa, ilk 5 sayısal sütunu al
            if not important_cols and numeric_cols:
                important_cols = numeric_cols[:5]

            if not important_cols:
                # Fallback: Orijinal X'i döndür
                return X

            # PolynomialFeatures oluştur
            self.poly_transformer = PolynomialFeatures(
                degree=degree,
                interaction_only=False,
                include_bias=False
            )

            # Sadece seçili sütunlarla fit et
            X_subset = X[important_cols]
            X_poly_subset = self.poly_transformer.fit_transform(X_subset)

            # Sütun adlarını oluştur
            poly_feature_names = self.poly_transformer.get_feature_names_out(important_cols)

            # Polynomial DataFrame oluştur
            X_poly_df = pd.DataFrame(
                X_poly_subset,
                columns=poly_feature_names,
                index=X.index
            )

            # Orijinal diğer sütunlarla birleştir
            other_cols = [col for col in X.columns if col not in important_cols]
            if other_cols:
                X_result = pd.concat([X[other_cols], X_poly_df], axis=1)
            else:
                X_result = X_poly_df

            return X_result

        else:
            # Daha önce fit edilmiş transformer kullan
            important_cols = list(self.poly_transformer.feature_names_in_)
            X_subset = X[important_cols]
            X_poly_subset = self.poly_transformer.transform(X_subset)

            poly_feature_names = self.poly_transformer.get_feature_names_out(important_cols)

            X_poly_df = pd.DataFrame(
                X_poly_subset,
                columns=poly_feature_names,
                index=X.index
            )

            other_cols = [col for col in X.columns if col not in important_cols]
            if other_cols:
                X_result = pd.concat([X[other_cols], X_poly_df], axis=1)
            else:
                X_result = X_poly_df

            return X_result

