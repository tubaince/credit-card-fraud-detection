# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 20:01:31 2025

@author: anily
"""

# Veri işleme
import pandas as pd

# Dengesiz veri teknikleri
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek


class Resampler:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def apply_smote(self):
        # SMOTE (Synthetic Minority Oversampling Technique)
        # → Azınlık sınıf için sentetik (yeni) örnekler üretir
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(self.X, self.y)
        return X_res, y_res

    def apply_enn(self):
        # SMOTE + ENN (Edited Nearest Neighbors)
        # → Önce SMOTE ile oversampling yapar,
        # → Sonra gürültülü/yanlış sınıflandırılmış örnekleri (komşuluk bazlı) temizler
        smote_enn = SMOTEENN(random_state=42)
        X_res, y_res = smote_enn.fit_resample(self.X, self.y)
        return X_res, y_res

    def apply_tomek(self):
        # SMOTE + Tomek Links
        # → Önce SMOTE ile oversampling yapar,
        # → Sonra sınıflar arasında sınırda kalan örnekleri (Tomek links) kaldırır
        smote_tomek = SMOTETomek(random_state=42)
        X_res, y_res = smote_tomek.fit_resample(self.X, self.y)
        return X_res, y_res

    def apply_undersampling(self):
        # Random UnderSampling
        # → Çoğunluk sınıfından rastgele örnekler atarak denge kurar
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(self.X, self.y)
        return X_res, y_res
