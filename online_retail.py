
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from keras import layers


def load_retail_data():
    try:
        df = pd.read_csv('./kaggle/online_retail.csv', encoding='unicode_escape')
    except:
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
            df = pd.read_excel(url)
        except:
            return None
    return df

df = load_retail_data()

if df is not None:
    initial_size = df.shape[0]
    df = df.dropna(subset=['CustomerID', 'Description'])

    df['CustomerID'] = df['CustomerID'].astype(int)
    df['UnitPrice'] = df['UnitPrice'].astype(str).str.replace(',', '.').astype(float)
    try:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True, errors='coerce')
    except Exception as e:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d.%m.%Y %H:%M', errors='coerce')

    invalid_dates = df[df['InvoiceDate'].isnull()]
    if len(invalid_dates) > 0:
        df = df.dropna(subset=['InvoiceDate'])

    initial_size_filter = df.shape[0]
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]

    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    
    analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (analysis_date - x.max()).days, 
        'InvoiceNo': 'nunique',                                  
        'TotalPrice': 'sum'                                      
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    def create_rfm_segments(rfm_df):
        rfm_df['R_Score'] = pd.cut(rfm_df['Recency'],
                                bins=[0, 30, 90, 180, 365, float('inf')],
                                labels=[5, 4, 3, 2, 1],
                                include_lowest=True)
        
        freq_bins = [0, 1, 2, 5, 10, float('inf')]
        if rfm_df['Frequency'].nunique() < 5:
            freq_bins = [0, 1, 2, 3, float('inf')]
            labels = [1, 2, 3, 4]
        else:
            labels = [1, 2, 3, 4, 5]
        
        rfm_df['F_Score'] = pd.cut(rfm_df['Frequency'],
                                bins=freq_bins,
                                labels=labels,
                                include_lowest=True)
        
        monetary_values = rfm_df['Monetary']
        monetary_quartiles = monetary_values.quantile([0.2, 0.4, 0.6, 0.8])
        
        if len(monetary_quartiles.unique()) < 4:
            monetary_bins = [0, monetary_values.median(), monetary_values.quantile(0.75), monetary_values.max()]
            monetary_labels = [1, 2, 3]
        else:
            monetary_bins = [0] + monetary_quartiles.tolist() + [float('inf')]
            monetary_labels = [1, 2, 3, 4, 5]
        
        rfm_df['M_Score'] = pd.cut(rfm_df['Monetary'],
                                bins=monetary_bins,
                                labels=monetary_labels,
                                include_lowest=True)
        
        return rfm_df

    rfm = create_rfm_segments(rfm)

    rfm['R_Score'] = rfm['R_Score'].astype(int)
    rfm['F_Score'] = rfm['F_Score'].astype(int)
    rfm['M_Score'] = rfm['M_Score'].astype(int)

    rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

    def assign_customer_segment(row):
        r_score, f_score, m_score = row['R_Score'], row['F_Score'], row['M_Score']
        total_score = row['RFM_Score']
        
        if total_score >= 12:
            return 'Champions'
        elif r_score >= 4 and f_score >= 4:
            return 'Loyal Customers'
        elif r_score >= 4 and m_score >= 4:
            return 'Big Spenders'
        elif f_score >= 4:
            return 'Regulars'
        elif r_score <= 2 and f_score <= 2:
            return 'Lost Customers'
        elif r_score <= 2:
            return 'At Risk'
        else:
            return 'Potential'

    rfm['Customer_Segment'] = rfm.apply(assign_customer_segment, axis=1)

    segment_counts = rfm['Customer_Segment'].value_counts()


    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    segment_counts.plot(kind='bar', color='lightcoral')
    plt.title('Распределение клиентов по сегментам')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 2)
    rfm_boxplot = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    rfm_boxplot['Log_Frequency'] = np.log1p(rfm_boxplot['Frequency'])
    rfm_boxplot['Log_Monetary'] = np.log1p(rfm_boxplot['Monetary'])
    rfm_boxplot[['Recency', 'Log_Frequency', 'Log_Monetary']].boxplot()
    plt.title('Распределение RFM-метрик (логарифмированные)')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    scatter = plt.scatter(rfm['Recency'], np.log1p(rfm['Frequency']), 
                        c=rfm['RFM_Score'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='RFM Score')
    plt.xlabel('Recency (дни)')
    plt.ylabel('Log(Frequency)')
    plt.title('RFM Analysis: Recency vs Frequency')

    plt.tight_layout()
    plt.show()

    for segment in rfm['Customer_Segment'].unique():
        count = (rfm['Customer_Segment'] == segment).sum()
    
    
    features = rfm[['Recency', 'Frequency', 'Monetary']]
    
    features['Log_Frequency'] = np.log1p(features['Frequency'])
    features['Log_Monetary'] = np.log1p(features['Monetary'])
    features = features[['Recency', 'Log_Frequency', 'Log_Monetary']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(rfm['Customer_Segment'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Матрица ошибок')
    plt.ylabel('Истинные значения')
    plt.xlabel('Предсказанные значения')
    plt.show()
    
    print(classification_report(y_test, y_pred_classes, 
                              target_names=label_encoder.classes_))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Функция потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Точность')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    



    new_customer = np.array([[30, 2.5, 4.0]])
    new_customer_scaled = scaler.transform(new_customer)
    
    prediction = model.predict(new_customer_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_segment = label_encoder.inverse_transform([predicted_class])[0]
    
    print(f"Предсказание для нового клиента:")
    print(f"Recency: 30 дней, Frequency: {np.expm1(2.5):.1f}, Monetary: ${np.expm1(4.0):.2f}")
    print(f"Предсказанный сегмент: {predicted_segment}")
    print(f"Вероятности по сегментам:")
    for i, segment in enumerate(label_encoder.classes_):
        print(f"  {segment}: {prediction[0][i]:.3f}")

