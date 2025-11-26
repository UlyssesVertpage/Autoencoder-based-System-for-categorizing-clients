
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import json





# pattern --- f'./_saves_instacart/_{numberFilesSavesDir}_'
savesNum = len(os.listdir('./_saves_instacart'))
if savesNum % 13 != 0: print('savesNum incorrect'); exit()
numberFilesSavesDir = savesNum // 13 + 1 + 11
numOfTry = 8


actFunc = 'elu'
valSplitNumber = 0.2
learningRateNumber = 0.01
epochsTrainNumber = 100
batchSizeNumber = 512
patienceESNumber = 30
factorNumber = 0.4
patienceRPNumber = 5


need_load_features_flag = True
need_load_not_train = False
need_save = True



print("import")


def load_instacart_data():
    try:
        orders = pd.read_csv('./kaggle/instacart/orders.csv')
        order_products_prior = pd.read_csv('./kaggle/instacart/order_products__prior.csv')
        order_products_train = pd.read_csv('./kaggle/instacart/order_products__train.csv')
        products = pd.read_csv('./kaggle/instacart/products.csv')
        order_products = pd.concat([order_products_prior, order_products_train], ignore_index=True)
        return orders, order_products, products   
    except FileNotFoundError as e:
        return None, None, None

def create_instacart_features(orders, order_products, products):
    data = orders.merge(order_products, on='order_id', how='inner')
    data = data.merge(products, on='product_id', how='left')
    data['order_dow'] = data['order_dow'].astype(int)
    data['order_hour_of_day'] = data['order_hour_of_day'].astype(int)
    data['days_since_prior_order'] = data['days_since_prior_order'].fillna(0)
    last_order_number_per_user = orders.groupby('user_id')['order_number'].max().reset_index()
    last_order_info = orders.merge(last_order_number_per_user, on=['user_id', 'order_number'])
    rfm_analysis = data.groupby('user_id').agg({
        'order_id': 'nunique',                         
        'days_since_prior_order': 'mean',              
        'product_id': 'count',                         
        'add_to_cart_order': 'mean',                   
        'reordered': 'mean'                            
    }).reset_index()
    user_last_order = last_order_info[['user_id', 'order_dow', 'order_hour_of_day']]
    rfm_analysis = rfm_analysis.merge(user_last_order, on='user_id', how='left')
    rfm_analysis.columns = [
        'user_id', 'order_frequency', 'avg_days_between_orders', 
        'total_products', 'avg_cart_position', 'reorder_ratio',
        'last_order_dow', 'last_order_hour'
    ]
    rfm_analysis['time_based_recency'] = (rfm_analysis['last_order_dow'] * 24 + rfm_analysis['last_order_hour'])
    user_behavior = data.groupby('user_id').agg({
        'product_id': 'nunique',                        
        'order_dow': 'nunique',                         
        'order_hour_of_day': 'nunique',                 
        'days_since_prior_order': lambda x: x[x > 0].std() if len(x[x > 0]) > 1 else 0
    }).reset_index()
    user_behavior.columns = [
        'user_id', 'unique_products', 'unique_order_days', 
        'unique_order_hours', 'order_interval_std'
    ]
    rfm = rfm_analysis.merge(user_behavior, on='user_id', how='left')
    rfm = rfm.fillna(0)
    return rfm

"""def reconstruction_accuracy(y_true, y_pred):
    error = tf.abs(y_true - y_pred) / (tf.abs(y_true) + 1e-8)
    accuracy = tf.reduce_mean(tf.cast(error < 0.1, tf.float32))
    return accuracy"""

"""def reconstruction_accuracy(y_true, y_pred):
    relative_error = tf.abs(y_true - y_pred) / (tf.abs(y_true) + 1e-8)
    accuracy_5p = tf.reduce_mean(tf.cast(relative_error < 0.05, tf.float32))
    accuracy_10p = tf.reduce_mean(tf.cast(relative_error < 0.1, tf.float32))
    accuracy_20p = tf.reduce_mean(tf.cast(relative_error < 0.2, tf.float32))
    return (accuracy_5p + accuracy_10p + accuracy_20p) / 3.0"""

def reconstruction_accuracy(y_true, y_pred):
    relative_error = tf.abs(y_true - y_pred) / (tf.abs(y_true) + 1e-8)
    accuracy_5p = tf.reduce_mean(tf.cast(relative_error < 0.05, tf.float32))
    accuracy_10p = tf.reduce_mean(tf.cast(relative_error < 0.1, tf.float32))
    accuracy_15p = tf.reduce_mean(tf.cast(relative_error < 0.15, tf.float32))
    return 0.6 * accuracy_10p + 0.3 * accuracy_5p + 0.1 * accuracy_15p

def combined_loss(y_true, y_pred):
    mse_loss = keras.losses.mse(y_true, y_pred)
    mae_loss = keras.losses.mae(y_true, y_pred)
    return 0.7 * mse_loss + 0.3 * mae_loss

def plot_training_validation_accuracy(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['reconstruction_accuracy'], 
             label='Training Accuracy', linewidth=2, color='blue')
    ax1.plot(history.history['val_reconstruction_accuracy'], 
             label='Validation Accuracy', linewidth=2, color='red')
    ax1.set_title('Reconstruction Accuracy\n(Training vs Validation)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    final_train_acc = history.history['reconstruction_accuracy'][-1]
    final_val_acc = history.history['val_reconstruction_accuracy'][-1]
    ax1.annotate(f'Final Train: {final_train_acc:.3f}\nFinal Val: {final_val_acc:.3f}', 
                xy=(0.7, 0.1), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax2.plot(history.history['loss'], 
             label='Training Loss', linewidth=2, color='blue', alpha=0.7)
    ax2.plot(history.history['val_loss'], 
             label='Validation Loss', linewidth=2, color='red', alpha=0.7)
    ax2.set_title('Learning Curves\n(Loss)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MSE)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    ax2.annotate(f'Final Train: {final_train_loss:.4f}\nFinal Val: {final_val_loss:.4f}', 
                xy=(0.7, 0.1), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    plt.tight_layout()
    plt.show()

def predict_instacart_cluster(new_user_data):
    try:
        loaded_scaler = joblib.load(f'./_saves_instacart/_{numOfTry}_instacart_scaler.pkl')
        loaded_encoder = keras.models.load_model(f'./_saves_instacart/_{numOfTry}_instacart_encoder_model.h5')
        loaded_kmeans = joblib.load(f'./_saves_instacart/_{numOfTry}_instacart_kmeans_model.pkl')
        with open(f'./_saves_instacart/_{numOfTry}_instacart_feature_names.json', 'r') as f:
            feature_names = json.load(f)
        new_data_df = pd.DataFrame([new_user_data])
        for feature in ['order_frequency', 'avg_days_between_orders', 'total_products', 
                    'unique_products', 'order_interval_std']:
            if feature in new_data_df.columns:
                new_data_df[f'{feature}_log'] = np.log1p(new_data_df[feature])
        new_data_df = new_data_df.drop(columns=['order_frequency', 'avg_days_between_orders', 
                                            'total_products', 'unique_products', 
                                            'order_interval_std'], errors='ignore')
        new_data_df = new_data_df.reindex(columns=feature_names, fill_value=0)
        new_data_scaled = loaded_scaler.transform(new_data_df)
        encoded_new = loaded_encoder.predict(new_data_scaled, verbose=0)
        cluster_prediction = loaded_kmeans.predict(encoded_new)[0]
        return cluster_prediction
    except Exception as e:
        return None

def predict_instacart_cluster_demo():
    try:
        with open(f'./_saves_instacart/_{numOfTry}_instacart_feature_names.json', 'r') as f:
            feature_names = json.load(f)
        with open(f'./_saves_instacart/_{numOfTry}_instacart_pipeline_metadata.json', 'r') as f:
            metadata = json.load(f)
        example_user = {
            'order_frequency': 8,
            'avg_days_between_orders': 10.5,
            'total_products': 45,
            'avg_cart_position': 3.2,
            'reorder_ratio': 0.65,
            'time_based_recency': 150,
            'unique_products': 25,
            'unique_order_days': 4,
            'unique_order_hours': 3,
            'order_interval_std': 2.1
        }
        predicted_cluster = predict_instacart_cluster(example_user)
        print(f"Данные клиента: {example_user}")
        print(f"Кластер: {predicted_cluster}")
        cluster_distribution = metadata.get('cluster_distribution', {})
        total_users = metadata.get('n_customers', 0)
        if str(predicted_cluster) in cluster_distribution:
            cluster_size = cluster_distribution[str(predicted_cluster)]
            percentage = (cluster_size / total_users) * 100
            print(f"Размер кластера: {cluster_size} пользователей ({percentage:.1f}%)")
        print(f"Всего кластеров: {metadata.get('n_clusters', 'N/A')}")
        print(f"Использовано признаков: {len(feature_names)}")
        return predicted_cluster
        
    except Exception as e:
        print(f"Ошибка при демонстрации предсказания: {e}")
        return None

"""
class LearningRateLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = self.model.optimizer.lr.numpy()
"""

def load_saved_features(save_dir='./instacart_saved_features'):
    try:
        if not os.path.exists(save_dir):
            return None, None, None, None
        original_features = pd.read_csv(f'{save_dir}/original_features.csv')
        engineered_features = pd.read_csv(f'{save_dir}/engineered_features.csv')
        scaled_features = np.load(f'{save_dir}/scaled_features.npy')
        # users_info = pd.read_csv(f'{save_dir}/users_info.csv')
        # with open(f'{save_dir}/transformation_stats.json', 'r') as f:
        #     stats = json.load(f)
        return original_features, engineered_features, scaled_features  #, users_info, stats
    except Exception as e:
        return None, None, None, None, None


orders, order_products, products = load_instacart_data()


# if False:
if orders is None: print("Некорректные данные"); exit()


features_for_model = [
    'order_frequency', 'avg_days_between_orders', 'total_products',
    'avg_cart_position', 'reorder_ratio', 'time_based_recency',
    'unique_products', 'unique_order_days', 'unique_order_hours', 
    'order_interval_std'
]



if need_load_features_flag:
    rfm, X, X_scaled = load_saved_features()
else:
    rfm = create_instacart_features(orders, order_products, products)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0,0].hist(rfm['order_frequency'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Order Frequency\n(Количество заказов)', fontweight='bold')
    axes[0,0].set_xlabel('Заказы')

    axes[0,1].hist(rfm['avg_days_between_orders'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0,1].set_title('Avg Days Between Orders\n(Среднее время между заказами)', fontweight='bold')
    axes[0,1].set_xlabel('Дни')

    axes[0,2].hist(rfm['total_products'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,2].set_title('Total Products\n(Общее количество товаров)', fontweight='bold')
    axes[0,2].set_xlabel('Товары')

    axes[1,0].hist(rfm['reorder_ratio'], bins=30, alpha=0.7, color='gold', edgecolor='black')
    axes[1,0].set_title('Reorder Ratio\n(Перезаказы)', fontweight='bold')
    axes[1,0].set_xlabel('Доля')

    axes[1,1].hist(rfm['unique_products'], bins=30, alpha=0.7, color='violet', edgecolor='black')
    axes[1,1].set_title('Unique Products\n(Уникальные товары)', fontweight='bold')
    axes[1,1].set_xlabel('Товары')

    axes[1,2].hist(rfm['time_based_recency'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1,2].set_title('Time-based Recency\n(Давность)', fontweight='bold')
    axes[1,2].set_xlabel('Время')

    axes[0,2].set_xlim(0, 1500)
    axes[0,2].set_xticks(np.arange(0, 1501, 300))

    axes[1,1].set_xlim(0, 400)   
    axes[1,1].set_xticks(np.arange(0, 401, 80))   

    plt.tight_layout()
    # plt.show()


    X = rfm[features_for_model].copy()

    print(X.describe())
    skew_features = ['order_frequency', 'avg_days_between_orders', 'total_products', 
                    'unique_products', 'order_interval_std']
    for feature in skew_features:
        if X[feature].std() > 0:
            X[f'{feature}_log'] = np.log1p(X[feature])
        else:
            X[f'{feature}_log'] = 0

    X = X.drop(columns=skew_features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, f'./_saves_instacart/_{numberFilesSavesDir}_instacart_scaler.pkl')
    with open(f'./_saves_instacart/_{numberFilesSavesDir}_instacart_feature_names.json', 'w') as f:
        json.dump(X.columns.tolist(), f)


    original_features_path = f'./_saves_instacart/_{numberFilesSavesDir}_original_features.csv'
    rfm.to_csv(original_features_path, index=False)

    engineered_features_path = f'./_saves_instacart/_{numberFilesSavesDir}_engineered_features.csv'
    X.to_csv(engineered_features_path, index=False)

    scaled_features_path = f'./_saves_instacart/_{numberFilesSavesDir}_scaled_features.npy'
    np.save(scaled_features_path, X_scaled)

    users_info_path = f'./_saves_instacart/_{numberFilesSavesDir}_users_info.csv'
    rfm[['user_id']].to_csv(users_info_path, index=False)

    transformation_stats = {
        'original_features': features_for_model,
        'engineered_features': X.columns.tolist(),
        'skew_features_transformed': skew_features,
        'data_shape': {
            'original': rfm[features_for_model].shape,
            'engineered': X.shape,
            'scaled': X_scaled.shape
        },
        'timestamp': datetime.now().isoformat()
    }
    
    stats_path = f'./_saves_instacart/_{numberFilesSavesDir}_transformation_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(transformation_stats, f, indent=2)


# numeric_columns = rfm.select_dtypes(include=[np.number]).columns
# numeric_columns = [col for col in numeric_columns if col != 'user_id']

# corr_matrix = rfm[numeric_columns].corr()

# plt.figure(figsize=(12, 10))
# mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
#             square=True, linewidths=0.5, fmt='.2f')
# plt.title('Матрица корреляций Instacart метрик', fontsize=16, fontweight='bold')
# plt.tight_layout()
# # plt.show()


input_dim = X_scaled.shape[1]
encoding_dim = 6
if need_load_not_train:
    encoder = keras.models.load_model(f'./_saves_instacart/_{numOfTry}_instacart_encoder_model.h5')
    encoder.summary()
    print('\n'*5)
    autoencoder = keras.models.load_model(
        f'./_saves_instacart/_{numOfTry}_instacart_autoencoder_model.h5',
        custom_objects={
            'reconstruction_accuracy': reconstruction_accuracy,
            'mse': 'mse'
        }
    )
    autoencoder.summary()
else:
    # Вариант 1 (изначальный)
    """
    input_layer = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation='relu')(input_layer)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.3)(encoded)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(32, activation='relu')(encoded)

    encoded = layers.Dense(encoding_dim, activation='relu', name='bottleneck')(encoded)

    decoded = layers.Dense(32, activation='relu')(encoded)
    decoded = layers.Dense(64, activation='relu')(decoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)
    """

    # Вариант 2 (уменьшённый)
    """
    input_layer = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation='relu')(input_layer)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(32, activation='relu')(encoded)

    encoded = layers.Dense(8, activation='relu', name='bottleneck')(encoded)

    decoded = layers.Dense(32, activation='relu')(encoded)
    decoded = layers.Dense(64, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)
    autoencoder = keras.Model(input_layer, decoded)
    """

    # Вариант 3 (увеличенный)
    """
    input_layer = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(256, activation='relu')(input_layer)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(128, activation='relu')(encoded)
    encoded = layers.Dropout(0.15)(encoded)
    encoded = layers.Dense(64, activation='relu')(encoded)

    encoded = layers.Dense(12, activation='relu', name='bottleneck')(encoded)

    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(256, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)

    autoencoder = keras.Model(input_layer, decoded)
    """

        # Вариант 3 (увеличенный)
    
    # Вариант 4 (увеличенный x2)
    
    input_layer = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(512, activation=actFunc)(input_layer)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(256, activation=actFunc)(encoded)
    encoded = layers.Dropout(0.15)(encoded)
    encoded = layers.Dense(128, activation=actFunc)(encoded)

    encoded = layers.Dense(20, activation=actFunc, name='bottleneck')(encoded)

    decoded = layers.Dense(128, activation=actFunc)(encoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dense(256, activation=actFunc)(decoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dense(512, activation=actFunc)(decoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)

    autoencoder = keras.Model(input_layer, decoded)
    

    # Вариант 5 (увеличенный x4)
    """
    input_layer = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(1024, activation=actFunc)(input_layer)
    encoded = layers.BatchNormalization()(encoded)
    # encoded = layers.Dropout(0.15)(encoded)
    encoded = layers.Dense(512, activation=actFunc)(encoded)
    encoded = layers.BatchNormalization()(encoded)
    # encoded = layers.Dropout(0.15)(encoded)
    encoded = layers.Dense(256, activation=actFunc)(encoded)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dense(128, activation=actFunc)(encoded)
    # encoded = layers.BatchNormalization()(encoded)

    encoded = layers.Dense(64, activation=actFunc, name='bottleneck')(encoded)

    decoded = layers.Dense(128, activation=actFunc)(encoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dense(256, activation=actFunc)(encoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dense(512, activation=actFunc)(decoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dense(1024, activation=actFunc)(decoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)

    autoencoder = keras.Model(input_layer, decoded)
    """
    
    # Вариант 6 (чутка меньше)
    """
    input_layer = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(256, activation='relu')(input_layer)
    encoded = layers.BatchNormalization()(encoded)
    # encoded = layers.Dropout(0.15)(encoded)
    encoded = layers.Dense(128, activation='relu')(encoded)
    encoded = layers.BatchNormalization()(encoded)
    # encoded = layers.Dropout(0.15)(encoded)

    encoded = layers.Dense(64, activation='relu', name='bottleneck')(encoded)

    decoded = layers.Dense(128, activation='relu')(encoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dense(256, activation='relu')(decoded)
    decoded = layers.BatchNormalization()(decoded)

    decoded = layers.Dense(input_dim, activation='linear')(decoded)

    autoencoder = keras.Model(input_layer, decoded)
    """

    
    custom_optimizer = keras.optimizers.Adam(learning_rate=learningRateNumber)

    autoencoder.compile(
        # optimizer='adam',         
        optimizer=custom_optimizer,
        loss=combined_loss,          
        metrics=['mae', reconstruction_accuracy]
    )


    encoder = keras.Model(input_layer, encoded)

    autoencoder.summary()


    """
    history = autoencoder.fit(
        X_scaled, X_scaled,
        epochs=50,
        batch_size=256,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10),
            # LearningRateLogger()     
        ]
    )
    """
    class TrainingProgressLogger(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 10 == 0: 
                print(f"Epoch {epoch:3d}: "
                    f"loss={logs['loss']:.4f}, "
                    f"val_loss={logs['val_loss']:.4f}, "
                    f"acc={logs['reconstruction_accuracy']:.4f}, "
                    f"val_acc={logs['val_reconstruction_accuracy']:.4f}")

    callbacks = [
        # keras.callbacks.EarlyStopping(patience=patienceESNumber, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=patienceRPNumber),
        TrainingProgressLogger()
    ]

    print("Training:")
    # try:
    history = autoencoder.fit(
        X_scaled, X_scaled,
        epochs=epochsTrainNumber,
        batch_size=batchSizeNumber,
        validation_split=valSplitNumber,
        verbose=1,
        callbacks=callbacks
    )
    # except KeyboardInterrupt:
    #     pass
        # autoencoder.save(f'./_saves_instacart/_{numberFilesSavesDir}_instacart_autoencoder_model.h5')
        # encoder.save(f'./_saves_instacart/_{numberFilesSavesDir}_instacart_encoder_model.h5')

    plot_training_validation_accuracy(history)
    

    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Learning Curve - Loss (MSE)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.title('Mean Absolute Error', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(history.history['reconstruction_accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_reconstruction_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Reconstruction Accuracy (<10% error)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], linewidth=2, color='purple')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
    else:
        plt.plot(history.history['loss'], label='Loss', linewidth=2, alpha=0.7)
        plt.plot(history.history['mae'], label='MAE', linewidth=2, alpha=0.7)
        plt.plot(history.history['reconstruction_accuracy'], label='Accuracy', linewidth=2, alpha=0.7)
        plt.title('All Metrics Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # print(f"\tTraining Loss: {history.history['loss'][-1]:.4f}")
    # print(f"\tValidation Loss: {history.history['val_loss'][-1]:.4f}")
    # print(f"\tTraining MAE: {history.history['mae'][-1]:.4f}")
    # print(f"\tValidation MAE: {history.history['val_mae'][-1]:.4f}")
    # print(f"\tTraining Accuracy: {history.history['reconstruction_accuracy'][-1]:.4f}")
    # print(f"\tValidation Accuracy: {history.history['val_reconstruction_accuracy'][-1]:.4f}")
    
    plt.figure(figsize=(14, 10))

    metrics_to_compare = ['loss', 'mae', 'reconstruction_accuracy']
    colors = ['red', 'blue', 'green']
    line_styles = ['-', '--', '-.']

    for i, metric in enumerate(metrics_to_compare):
        plt.subplot(2, 2, i+1)
        
        train_vals = np.array(history.history[metric])
        val_vals = np.array(history.history[f'val_{metric}'])
        
        if metric == 'reconstruction_accuracy':
            train_norm = train_vals
            val_norm = val_vals
            ylabel = 'Accuracy'
        else:
            train_max = np.max(train_vals)
            val_max = np.max(val_vals)
            train_norm = train_vals / train_max if train_max > 0 else train_vals
            val_norm = val_vals / val_max if val_max > 0 else val_vals
            ylabel = f'Normalized {metric.upper()}'
        
        plt.plot(train_norm, label=f'Train {metric}', 
                color=colors[i], linestyle=line_styles[0], linewidth=2)
        plt.plot(val_norm, label=f'Val {metric}', 
                color=colors[i], linestyle=line_styles[1], linewidth=2)
        
        plt.title(f'{metric.upper()} - Training vs Validation', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()



    autoencoder.save(f'./_saves_instacart/_{numberFilesSavesDir}_instacart_autoencoder_model.h5')
    encoder.save(f'./_saves_instacart/_{numberFilesSavesDir}_instacart_encoder_model.h5')


encoded_features = encoder.predict(X_scaled)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(encoded_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linewidth=2, markersize=8)
plt.title('Метод локтя для определения числа кластеров', fontsize=14, fontweight='bold')
plt.xlabel('Количество кластеров')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 11))
plt.show()

try:
    from kneed import KneeLocator
    kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
    n_clusters = kn.knee if kn.knee else 5
except:
    n_clusters = 5

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(encoded_features)

rfm['cluster'] = cluster_labels

# joblib.dump(kmeans, f'./_saves_instacart/_{numberFilesSavesDir}_instacart_kmeans_model.pkl')






plt.figure(figsize=(10, 6))
cluster_counts = rfm['cluster'].value_counts().sort_index()
colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))

bars = plt.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black')
plt.title('Распределение пользователей по кластерам', fontsize=14, fontweight='bold')
plt.xlabel('Номер кластера')
plt.ylabel('Количество пользователей')
plt.xticks(cluster_counts.index)

for bar, count in zip(bars, cluster_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{count}', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.show()



pca = PCA(n_components=2)
cluster_2d = pca.fit_transform(encoded_features)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(cluster_2d[:, 0], cluster_2d[:, 1], 
                    c=cluster_labels, cmap='viridis', alpha=0.7, s=30)
plt.colorbar(scatter, label='Кластер')
plt.title('Визуализация кластеров с помощью PCA', fontsize=14, fontweight='bold')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

centroids_2d = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
        marker='X', s=200, c='red', edgecolors='black', linewidth=2, label='Центроиды')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

"""
plt.figure(figsize=(14, 8))

heatmap = rfm.groupby('cluster').agg({
    'order_frequency': 'mean',
    'avg_days_between_orders': 'mean',
    'total_products': 'mean',
    'reorder_ratio': 'mean',
    'unique_products': 'mean',
    'time_based_recency': 'mean'
})

# heatmap_data_normalized = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()

# sns.heatmap(heatmap_data_normalized, annot=heatmap_data.round(2), fmt='', 
#             cmap='RdYlBu_r', center=0, linewidths=0.5)
# plt.title('Средние значения метрик по кластерам Instacart\n(аннотации - исходные значения)', 
#         fontsize=14, fontweight='bold')
# plt.tight_layout()
# plt.show()



plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('Матрица корреляций Instacart метрик', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()



# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
#             square=True, linewidths=0.5, fmt='.2f')
# plt.title('Полная матрица корреляций Instacart', fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.show()
"""

"""
def create_instacart_cluster_profiles(rfm_df):
    profiles = {}
    
    for cluster in sorted(rfm_df['cluster'].unique()):
        cluster_data = rfm_df[rfm_df['cluster'] == cluster]
        
        avg_frequency = cluster_data['order_frequency'].mean()
        avg_days_between = cluster_data['avg_days_between_orders'].mean()
        avg_reorder_ratio = cluster_data['reorder_ratio'].mean()
        avg_total_products = cluster_data['total_products'].mean()
        size = len(cluster_data)
        
        if avg_frequency > 15 and avg_reorder_ratio > 0.7:
            segment_type = "Супер-активные планировщики"
        elif avg_frequency > 10 and avg_days_between < 7:
            segment_type = "Еженедельные покупатели"
        elif avg_reorder_ratio > 0.6:
            segment_type = "Лояльные повторные покупатели"
        elif avg_days_between > 20:
            segment_type = "Сезонные/Редкие покупатели"
        elif avg_frequency < 5:
            segment_type = "Новые/Случайные покупатели"
        elif avg_total_products > 50:
            segment_type = "Крупные покупатели (семьи)"
        else:
            segment_type = "Стабильные регулярные покупатели"
            
        profiles[cluster] = {
            'size': size,
            'frequency': avg_frequency,
            'days_between': avg_days_between,
            'reorder_ratio': avg_reorder_ratio,
            'total_products': avg_total_products,
            'type': segment_type
        }
        
    return profiles
"""

def create_instacart_cluster_profiles(rfm_df):
    profiles = {}
    
    freq_q75 = rfm_df['order_frequency'].quantile(0.75)
    freq_q25 = rfm_df['order_frequency'].quantile(0.25)
    days_q75 = rfm_df['avg_days_between_orders'].quantile(0.75)
    days_q25 = rfm_df['avg_days_between_orders'].quantile(0.25)
    reorder_q75 = rfm_df['reorder_ratio'].quantile(0.75)
    reorder_q25 = rfm_df['reorder_ratio'].quantile(0.25)
    products_q75 = rfm_df['total_products'].quantile(0.75)
    products_q25 = rfm_df['total_products'].quantile(0.25)
    
    print(f"Частота: Q25={freq_q25:.1f}, Q75={freq_q75:.1f}")
    print(f"Дней между заказами: Q25={days_q25:.1f}, Q75={days_q75:.1f}")
    print(f"Коэф. перезаказа: Q25={reorder_q25:.2f}, Q75={reorder_q75:.2f}")
    print(f"Всего товаров: Q25={products_q25:.1f}, Q75={products_q75:.1f}")
    
    for cluster in sorted(rfm_df['cluster'].unique()):
        cluster_data = rfm_df[rfm_df['cluster'] == cluster]
        
        avg_frequency = cluster_data['order_frequency'].mean()
        avg_days_between = cluster_data['avg_days_between_orders'].mean()
        avg_reorder_ratio = cluster_data['reorder_ratio'].mean()
        avg_total_products = cluster_data['total_products'].mean()
        size = len(cluster_data)
        
        if (avg_frequency > freq_q75 and 
            avg_reorder_ratio > reorder_q75 and 
            avg_days_between < days_q25):
            segment_type = "Супер-активные планировщики"
        elif (avg_frequency > freq_q75 and 
              avg_days_between < days_q25):
            segment_type = "Еженедельные покупатели"
        elif (avg_reorder_ratio > reorder_q75 and 
              avg_frequency > freq_q25):
            segment_type = "Лояльные повторные покупатели"
        elif (avg_days_between > days_q75 or 
              avg_frequency < freq_q25):
            segment_type = "Сезонные/Редкие покупатели"
        elif (avg_total_products > products_q75 and 
              avg_frequency > freq_q25):
            segment_type = "Крупные покупатели (семьи)"
        elif (avg_frequency < freq_q25 and 
              avg_total_products < products_q25):
            segment_type = "Новые/Случайные покупатели"
        else:
            segment_type = "Стабильные регулярные покупатели"
            
        profiles[cluster] = {
            'size': size,
            'frequency': avg_frequency,
            'days_between': avg_days_between,
            'reorder_ratio': avg_reorder_ratio,
            'total_products': avg_total_products,
            'type': segment_type
        }
        
        print(f"\nКластер {cluster} ({size} пользователей):")
        print(f"Частота: {avg_frequency:.1f} (Q75: {freq_q75:.1f})")
        print(f"Дней между заказами: {avg_days_between:.1f} (Q25: {days_q25:.1f})")
        print(f"Коэф. перезаказа: {avg_reorder_ratio:.2f} (Q75: {reorder_q75:.2f})")
        print(f"Всего товаров: {avg_total_products:.1f} (Q75: {products_q75:.1f})")
        print(f"Сегмент: {segment_type}")
        
    return profiles


cluster_profiles = create_instacart_cluster_profiles(rfm)



instacart_strategies = {
    "Супер-активные планировщики": [
        "Персональные предложения на часто покупаемые товары",
        "Ранний доступ к акциям и новинкам",
        "Бесплатная доставка и приоритетное обслуживание"
    ],
    "Еженедельные покупатели": [
        "Подписка на регулярные заказы со скидкой",
        "Напоминания о пополнении запасов",
        "Бонусы за последовательные недели заказов"
    ],
    "Лояльные повторные покупатели": [
        "Программа лояльности с увеличенными бонусами",
        "Эксклюзивные предложения на любимые товары",
        "Персональные купоны на бренды которые они часто покупают"
    ],
    "Сезонные/Редкие покупатели": [
        "Напоминания о сезонных товарах и акциях",
        "Специальные предложения для возвращения",
        "Обзор новых услуг и возможностей платформы"
    ],
    "Новые/Случайные покупатели": [
        "Приветственная скидка на следующий заказ",
        "Образовательный контент о преимуществах планирования покупок",
        "Рекомендации на основе первого заказа"
    ],
    "Крупные покупатели (семьи)": [
        "Семейные пакеты и оптовые предложения",
        "Персональный помощник по составлению корзины",
        "Специальные условия для крупных заказов"
    ],
    "Стабильные регулярные покупатели": [
        "Регулярные персонализированные предложения",
        "Бонусы за постоянство",
        "Опросы для улучшения сервиса"
    ]
}




cell_text = []
for cluster, profile in cluster_profiles.items():
    segment_type = profile['type']
    if segment_type in instacart_strategies:
        main_strategy = instacart_strategies[segment_type][0]
    else:
        main_strategy = "Индивидуальный подход"
    
    cell_text.append([
        f"Кластер {cluster}",
        segment_type,
        f"{profile['size']} ({profile['size']/len(rfm)*100:.1f}%)",
        f"Заказы: {profile['frequency']:.1f}",
        f"Перезаказ: {profile['reorder_ratio']:.2f}",
        main_strategy
    ])

plt.figure(figsize=(16, 10))
table = plt.table(cellText=cell_text,
                colLabels=['Кластер', 'Сегмент', 'Размер', 'Частота', 'Реордер', 'Основная стратегия'],
                loc='center',
                cellLoc='center',
                colWidths=[0.1, 0.2, 0.1, 0.1, 0.1, 0.4])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

plt.axis('off')
plt.title('Маркетинговая стратегия Instacart по кластерам', 
        fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()






for cluster_id, profile in cluster_profiles.items():
    segment_type = profile['type']
    size = profile['size']
    percentage = (size / len(rfm)) * 100
    
    print(f"\nКластер {cluster_id}: {segment_type}")
    print(f"\tРазмер: {size} пользователей ({percentage:.1f}%)")
    print(f"\tХарактеристики: {profile['frequency']:.1f} заказов, "
        f"Перезааказ {profile['reorder_ratio']:.2f}, "
        f"{profile['days_between']:.1f} дней между заказами")
    
    if segment_type in instacart_strategies:
        print("\tРекомендуемые кампании:")
        for i, strategy in enumerate(instacart_strategies[segment_type], 1):
            print(f"      {i}. {strategy}")






if (not need_save): print('No saving anything.')
else:
    pipeline_metadata = {
        'timestamp': datetime.now().isoformat(),
        'input_dimension': int(input_dim),
        'encoding_dimension': int(encoding_dim),
        'n_clusters': int(n_clusters),
        'n_customers': int(len(rfm)),
        'features_used': X.columns.tolist(),
        'cluster_distribution': rfm['cluster'].value_counts().astype(int).to_dict(),
        'model_versions': {
            'tensorflow': tf.__version__,
            'keras': keras.__version__
        },
        'dataset': 'Instacart Market Basket Analysis'
    }

    with open(f'./_saves_instacart/_{numberFilesSavesDir}_instacart_pipeline_metadata.json', 'w') as f:
        json.dump(pipeline_metadata, f, indent=2, default=str)

    rfm.to_csv(f'./_saves_instacart/_{numberFilesSavesDir}_instacart_customer_clusters.csv', index=False)

    cluster_summary = rfm.groupby('cluster').agg({
        'order_frequency': ['mean', 'std'],
        'avg_days_between_orders': ['mean', 'std'],
        'reorder_ratio': ['mean', 'std'],
        'total_products': ['mean', 'std'],
        'unique_products': ['mean', 'std'],
        'user_id': 'count'
    }).round(3)

    cluster_summary.to_csv(f'./_saves_instacart/_{numberFilesSavesDir}_instacart_cluster_summary.csv')

    joblib.dump(kmeans, f'./_saves_instacart/_{numberFilesSavesDir}_instacart_kmeans_model.pkl')





example_user = {
    'order_frequency': 8,
    'avg_days_between_orders': 10.5,
    'total_products': 45,
    'avg_cart_position': 3.2,
    'reorder_ratio': 0.65,
    'time_based_recency': 150,
    'unique_products': 25,
    'unique_order_days': 4,
    'unique_order_hours': 3,
    'order_interval_std': 2.1
}

predicted_cluster = predict_instacart_cluster(example_user)
print(f"\tПример предсказания для нового пользователя:")
print(f"\tДанные: {example_user}")
print(f"\tПредсказанный кластер: {predicted_cluster}")

if predicted_cluster is not None and predicted_cluster in cluster_profiles:
    profile = cluster_profiles[predicted_cluster]
    print(f"\tСегмент: {profile['type']}")
    print(f"\tРекомендации: {instacart_strategies.get(profile['type'], ['Индивидуальный подход'])[0]}")



