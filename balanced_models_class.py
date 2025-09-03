# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 21:30:00 2025

@author: anily

Bu dosyada dengesiz veriler üzerinde çalışan farklı modelleri denemek için bir sınıf oluşturuyoruz.
Her modelin mantığı, avantajları ve kullanım amacı yorum satırlarında açıklanmıştır.
Her fonksiyon artık:
- Eğitilmiş modeli
- ROC-AUC skorunu
döndürür, böylece ana kod dosyasında istediğin gibi hem modeli hem de performansı alabilirsin.
"""

# Veri işleme
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Makine öğrenmesi modelleri
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class BalancedModels:
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Constructor (kurucu fonksiyon):
        Eğitim ve test verilerini sınıfa alıyoruz.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_model(self, model):
        """
        Tüm modeller için ortak değerlendirme işlemleri:
        1. Modeli eğitir
        2. Test setinde tahmin yapar
        3. Confusion Matrix, Classification Report ve ROC-AUC skorunu ekrana yazdırır
        4. Eğitilmiş modeli ve ROC-AUC skorunu döndürür
        """
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        print("\n==============================")
        print(f"Model: {model.__class__.__name__}")
        print("==============================")
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("\nClassification Report:\n", classification_report(self.y_test, y_pred))
        roc = roc_auc_score(self.y_test, y_pred)
        print("ROC-AUC:", roc)

        return model, roc   # <-- Model ve skor döndürülüyor

    def logistic_regression(self):
        """
        Logistic Regression:
        - Basit, hızlı ve yorumlanabilir bir modeldir.
        - 'class_weight="balanced"' parametresi ile azınlık sınıfa daha fazla önem verilir.
        - Dengesiz veri setlerinde baseline (referans) model olarak sıkça tercih edilir.
        """
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        return self.evaluate_model(model)

    def random_forest(self):
        """
        Random Forest:
        - Karar ağaçlarının birleşimi (bagging yöntemi).
        - class_weight="balanced" parametresi ile dengesiz veri daha adil işlenir.
        - Özellikle büyük veri setlerinde güçlüdür.
        - Özellik önemlerini (feature importance) inceleme avantajı sağlar.
        """
        model = RandomForestClassifier(class_weight="balanced", random_state=42)
        return self.evaluate_model(model)

    def xgboost(self):
        """
        XGBoost (Extreme Gradient Boosting):
        - Boosting tabanlı, çok güçlü bir algoritmadır.
        - scale_pos_weight → azınlık sınıfın ağırlığını artırır (fraud detection gibi durumlarda kritik).
        - Dengesiz veri setlerinde ROC-AUC performansı genelde çok yüksektir.
        - Daha fazla hiperparametre ayarı ile optimize edilebilir.
        """
        scale = len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])
        # çoğunluk/azınlık oranına göre scale_pos_weight hesaplanıyor
        model = XGBClassifier(
            scale_pos_weight=scale,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
        return self.evaluate_model(model)
