

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# url = "https://raw.githubusercontent.com/srivatsan88/YouTubeLI/master/dataset/Mall_Customers.csv"
df = pd.read_csv("./kaggle/Mall_Customers.csv")

print(df.shape)
print(df.head())
print(df.info())





# print(df.describe())

# print(df.isnull().sum())

# plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
# sns.histplot(df['Age'], bins=20, kde=True)
# plt.title('Распределение возраста')

# plt.subplot(1, 3, 2)
# sns.histplot(df['Annual Income (k$)'], bins=20, kde=True)
# plt.title('Распределение годового дохода')

# plt.subplot(1, 3, 3)
# sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True)
# plt.title('Распределение оценки расходов')

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 5))
# sns.countplot(data=df, x='Gender')
# plt.title('Распределение по полу')
# plt.show()






label_encoder = LabelEncoder()
df['Gender_encoded'] = label_encoder.fit_transform(df['Gender'])

df_processed = df.drop(['CustomerID', 'Gender'], axis=1)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_processed)

df_scaled = pd.DataFrame(scaled_features, 
                        columns=df_processed.columns,
                        index=df_processed.index)







def create_segments(score):
    if score <= 40:
        return 'Low Spender'
    elif score <= 60:
        return 'Medium Spender'
    else:
        return 'High Spender'

df_scaled['Segment'] = df['Spending Score (1-100)'].apply(create_segments)

segment_encoder = LabelEncoder()
df_scaled['Segment_encoded'] = segment_encoder.fit_transform(df_scaled['Segment'])


X = df_scaled.drop(['Spending Score (1-100)', 'Segment', 'Segment_encoded'], axis=1)
y = df_scaled['Segment_encoded']









X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)






import tensorflow as tf
from tensorflow import keras
from keras import layers

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Архитектура модели:")
model.summary()







early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)







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





test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=segment_encoder.classes_,
            yticklabels=segment_encoder.classes_)
plt.title('Матрица ошибок')
plt.ylabel('Истинные значения')
plt.xlabel('Предсказанные значения')
plt.show()

print(classification_report(y_test, y_pred_classes, 
                          target_names=segment_encoder.classes_))






