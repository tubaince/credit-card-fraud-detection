# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 19:50:27 2025

@author: anily
"""

# Veri işleme
import pandas as pd
import numpy as np

# Veri görselleştirme
import matplotlib.pyplot as plt
import seaborn as sns

# Makine öğrenmesi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dengesiz veri teknikleri
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

# Modeller
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# Veri setini yükle
df = pd.read_csv("creditcard.csv")

# İlk 5 satırı göster
print(df.head())

# Veri seti hakkında genel bilgi
print(df.info())

# Sınıf dağılımı (fraud ve non-fraud sayısı)
print(df['Class'].value_counts())


X = df.drop('Class', axis=1)
y = df['Class']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


"""
from resampler_class import Resampler

# Örnek kullanım
resampler = Resampler(X_train, y_train)

X_smote, y_smote = resampler.apply_smote()
print(pd.Series(y_smote).value_counts())

X_enn, y_enn = resampler.apply_enn()
print(pd.Series(y_enn).value_counts())

X_tomek, y_tomek = resampler.apply_tomek()
print(pd.Series(y_tomek).value_counts())


"""


from balanced_models_class import BalancedModels

# Balanced modeller denemesi
bal_models = BalancedModels(X_train, y_train, X_test, y_test)

log_model, log_score = bal_models.logistic_regression()
print("Logistic Regression Balanced Score:", log_score)

rf_model, rf_score = bal_models.random_forest()
print("Random Forest Balanced Score:", rf_score)

xgb_model, xgb_score = bal_models.xgboost()
print("XGBoost Balanced Score:", xgb_score)

