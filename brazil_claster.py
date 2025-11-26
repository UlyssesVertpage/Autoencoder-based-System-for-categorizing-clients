
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

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_and_preprocess_data():
    try:
        customers = pd.read_csv('./kaggle/brazil/olist_customers_dataset.csv')
        orders = pd.read_csv('./kaggle/brazil/olist_orders_dataset.csv')
        order_items = pd.read_csv('./kaggle/brazil/olist_order_items_dataset.csv')
        payments = pd.read_csv('./kaggle/brazil/olist_order_payments_dataset.csv')

        data = orders.merge(customers, on='customer_id', how='left')
        data = data.merge(order_items, on='order_id', how='left')
        data = data.merge(payments, on='order_id', how='left')
        
        data['order_purchase_timestamp'] = pd.to_datetime(data['order_purchase_timestamp'])
        
        data = data.dropna(subset=['customer_id', 'payment_value'])
        data = data[data['payment_value'] > 0]
        
        return data, customers, orders, order_items, payments
        
    except FileNotFoundError:
        return None, None, None, None, None



data, customers, orders, order_items, payments = load_and_preprocess_data()

if data is not None:
    
    analysis_date = data['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    
    rfm = data.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (analysis_date - x.max()).days,
        'order_id': 'nunique',
        'payment_value': 'sum'
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    customer_dates = data.groupby('customer_id')['order_purchase_timestamp'].agg(['min', 'max'])
    customer_dates['customer_lifetime'] = (customer_dates['max'] - customer_dates['min']).dt.days
    rfm = rfm.merge(customer_dates[['customer_lifetime']], on='customer_id')
    
    avg_items = order_items.groupby('order_id')['order_item_id'].count().reset_index()
    avg_items.columns = ['order_id', 'items_per_order']
    customer_avg_items = data.merge(avg_items, on='order_id').groupby('customer_id')['items_per_order'].mean()
    rfm = rfm.merge(customer_avg_items, on='customer_id')
    
    category_diversity = data.groupby('customer_id')['product_id'].nunique()
    rfm = rfm.merge(category_diversity.rename('unique_products'), on='customer_id')
    
    payment_diversity = data.groupby('customer_id')['payment_type'].nunique()
    rfm = rfm.merge(payment_diversity.rename('payment_methods'), on='customer_id')
    
    rfm = rfm.fillna(0)
    


    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0,0].hist(rfm['recency'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Распределение Recency', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Дней с последней покупки')
    axes[0,0].set_ylabel('Количество клиентов')
    
    axes[0,1].hist(rfm['frequency'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0,1].set_title('Распределение Frequency', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Количество заказов')
    axes[0,1].set_ylabel('Количество клиентов')
    
    axes[0,2].hist(np.log1p(rfm['monetary']), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,2].set_title('Распределение Monetary (логарифм)', fontsize=14, fontweight='bold')
    axes[0,2].set_xlabel('Логарифм общей суммы покупок')
    axes[0,2].set_ylabel('Количество клиентов')
    
    axes[1,0].hist(rfm['customer_lifetime'], bins=50, alpha=0.7, color='gold', edgecolor='black')
    axes[1,0].set_title('Распределение Customer Lifetime', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Дней активности')
    axes[1,0].set_ylabel('Количество клиентов')
    
    axes[1,1].hist(rfm['items_per_order'], bins=50, alpha=0.7, color='violet', edgecolor='black')
    axes[1,1].set_title('Распределение Items per Order', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Среднее количество товаров в заказе')
    axes[1,1].set_ylabel('Количество клиентов')
    
    axes[1,2].hist(rfm['unique_products'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1,2].set_title('Распределение Unique Products', fontsize=14, fontweight='bold')
    axes[1,2].set_xlabel('Количество уникальных товаров')
    axes[1,2].set_ylabel('Количество клиентов')
    
    plt.tight_layout()
    plt.show()
    
    
    corr_matrix = rfm[['recency', 'frequency', 'monetary', 'customer_lifetime', 
                      'items_per_order', 'unique_products', 'payment_methods']].corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Матрица корреляций RFM метрик', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    


    
    features = ['recency', 'frequency', 'monetary', 'customer_lifetime', 
               'items_per_order', 'unique_products', 'payment_methods']
    
    X = rfm[features].copy()
    
    skew_features = ['frequency', 'monetary', 'customer_lifetime', 'items_per_order', 'unique_products']
    for feature in skew_features:
        if X[feature].std() > 0:
            X[f'{feature}_log'] = np.log1p(X[feature])
        else:
            X[f'{feature}_log'] = 0
    
    X = X.drop(columns=skew_features)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    joblib.dump(scaler, 'customer_scaler.pkl')
    
    with open('feature_names.json', 'w') as f:
        json.dump(X.columns.tolist(), f)
    




    
    input_dim = X_scaled.shape[1]
    encoding_dim = 5
    
    input_layer = keras.Input(shape=(input_dim,))
    
    encoded = layers.Dense(64, activation='relu')(input_layer)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.3)(encoded)
    
    encoded = layers.Dense(32, activation='relu')(encoded)
    encoded = layers.Dropout(0.2)(encoded)
    
    encoded = layers.Dense(16, activation='relu')(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu', name='bottleneck')(encoded)
    
    decoded = layers.Dense(16, activation='relu')(encoded)
    decoded = layers.Dense(32, activation='relu')(decoded)
    decoded = layers.Dense(64, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    encoder = keras.Model(input_layer, encoded)
    
    autoencoder.summary()
    


    history = autoencoder.fit(
        X_scaled, X_scaled,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8)
        ]
    )
    



    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Learning Curve - Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if 'lr' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['lr'], linewidth=2, color='purple')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    



    autoencoder.save('autoencoder_model.h5')
    encoder.save('encoder_model.h5')
    
    
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
    
    from kneed import KneeLocator
    kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
    n_clusters = kn.knee if kn.knee else 5
    
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(encoded_features)
    
    rfm['cluster'] = cluster_labels
    
    joblib.dump(kmeans, 'kmeans_model.pkl')
    



    plt.figure(figsize=(10, 6))
    cluster_counts = rfm['cluster'].value_counts().sort_index()
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
    
    bars = plt.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black')
    plt.title('Распределение клиентов по кластерам', fontsize=14, fontweight='bold')
    plt.xlabel('Номер кластера')
    plt.ylabel('Количество клиентов')
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
                         c=cluster_labels, cmap='viridis', alpha=0.7, s=50)
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
    

    
    cluster_profiles = rfm.groupby('cluster').agg({
        'recency': 'mean',
        'frequency': 'mean', 
        'monetary': 'mean',
        'customer_id': 'count'
    }).round(2)
    
    normalized_profiles = cluster_profiles.copy()
    for col in ['recency', 'frequency', 'monetary']:
        normalized_profiles[col] = (cluster_profiles[col] - cluster_profiles[col].min()) / (cluster_profiles[col].max() - cluster_profiles[col].min())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    metrics = ['recency', 'frequency', 'monetary']
    colors = ['red', 'green', 'blue']
    
    for i, (cluster, profile) in enumerate(normalized_profiles.iterrows()):
        if i >= len(axes):
            break
            
        values = profile[metrics].tolist()
        values += values[:1]
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        ax = axes[i]
        ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title(f'Кластер {cluster}\n(n={profile["customer_id"]})', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.suptitle('Профили кластеров по RFM метрикам', fontsize=16, fontweight='bold', y=1.02)
    plt.show()
    
    plt.figure(figsize=(12, 8))
    
    heatmap_data = rfm.groupby('cluster').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'customer_lifetime': 'mean',
        'items_per_order': 'mean',
        'unique_products': 'mean'
    })
    
    heatmap_data_normalized = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()
    
    sns.heatmap(heatmap_data_normalized, annot=heatmap_data.round(2), fmt='', 
                cmap='RdYlBu_r', center=0, linewidths=0.5)
    plt.title('Средние значения метрик по кластерам\n(аннотации - исходные значения)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    


    sample_size = min(1000, len(rfm))
    sample_rfm = rfm.sample(sample_size, random_state=42)
    
    g = sns.pairplot(sample_rfm, 
                    vars=['recency', 'frequency', 'monetary'],
                    hue='cluster', 
                    palette='viridis',
                    plot_kws={'alpha': 0.6, 's': 30},
                    diag_kind='hist')
    g.fig.suptitle('Pairplot RFM метрик по кластерам', y=1.02, fontsize=16, fontweight='bold')
    plt.show()
    
    




    def create_auto_cluster_profiles(rfm_df):
        profiles = {}
        
        for cluster in sorted(rfm_df['cluster'].unique()):
            cluster_data = rfm_df[rfm_df['cluster'] == cluster]
            
            avg_recency = cluster_data['recency'].mean()
            avg_frequency = cluster_data['frequency'].mean() 
            avg_monetary = cluster_data['monetary'].mean()
            size = len(cluster_data)
            
            if avg_recency < 60 and avg_frequency > 2 and avg_monetary > 500:
                segment_type = "Активные VIP клиенты"
            elif avg_recency < 90 and avg_frequency > 1:
                segment_type = "Лояльные клиенты"  
            elif avg_recency > 180:
                segment_type = "Неактивные клиенты"
            elif avg_monetary > 1000:
                segment_type = "Крупные покупатели"
            elif avg_frequency == 1:
                segment_type = "Однократные покупатели"
            else:
                segment_type = "Стабильные клиенты"
                
            profiles[cluster] = {
                'size': size,
                'recency': avg_recency,
                'frequency': avg_frequency, 
                'monetary': avg_monetary,
                'type': segment_type
            }
            
        return profiles
    
    cluster_profiles = create_auto_cluster_profiles(rfm)
    
    strategies = {
        "Активные VIP клиенты": [
            "Эксклюзивные предложения и ранний доступ к новинкам",
            "Персональный менеджер и VIP-обслуживание",
            "Программа лояльности с повышенными бонусами"
        ],
        "Лояльные клиенты": [
            "Регулярные персональные скидки",
            "Программа поощрения за рекомендации", 
            "Приглашения на специальные мероприятия"
        ],
        "Неактивные клиенты": [
            "Кампания 'Мы по вам скучаем' с специальными предложениями",
            "Опрос для выяснения причин снижения активности",
            "Персональные условия для возвращения"
        ],
        "Крупные покупатели": [
            "Персональные консультации по товарам",
            "Расширенная гарантия и сервис",
            "Эксклюзивные условия оплаты"
        ],
        "Однократные покупатели": [
            "Стимулы для повторной покупки",
            "Приветственная программа лояльности", 
            "Персонализированные рекомендации"
        ],
        "Стабильные клиенты": [
            "Регулярные коммуникации о новинках",
            "Программа поощрения за постоянство",
            "Специальные предложения на часто покупаемые товары"
        ]
    }
    
    

    strategy_df = pd.DataFrame([
        {
            'Кластер': cluster,
            'Тип': profile['type'],
            'Размер': profile['size'],
            'Доля': f"{(profile['size'] / len(rfm) * 100):.1f}%",
            'Приоритет': 'Высокий' if profile['type'] in ['Активные VIP клиенты', 'Крупные покупатели'] else 'Средний'
        }
        for cluster, profile in cluster_profiles.items()
    ])
    
    plt.figure(figsize=(14, 8))
    


    cell_text = []
    for cluster, profile in cluster_profiles.items():
        segment_type = profile['type']
        if segment_type in strategies:
            main_strategy = strategies[segment_type][0]
        else:
            main_strategy = "Индивидуальный подход"
        
        cell_text.append([
            f"Кластер {cluster}",
            segment_type,
            f"{profile['size']} ({profile['size']/len(rfm)*100:.1f}%)",
            main_strategy
        ])
    
    table = plt.table(cellText=cell_text,
                     colLabels=['Кластер', 'Сегмент', 'Размер', 'Основная стратегия'],
                     loc='center',
                     cellLoc='center',
                     colWidths=[0.15, 0.25, 0.15, 0.45])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.axis('off')
    plt.title('Сводная маркетинговая стратегия по кластерам', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    


    for cluster_id, profile in cluster_profiles.items():
        segment_type = profile['type']
        size = profile['size']
        percentage = (size / len(rfm)) * 100
        
        print(f"\nКластер {cluster_id}: {segment_type}")
        print(f"Размер: {size} клиентов ({percentage:.1f}%)")
        print(f"Характеристики: Recency={profile['recency']:.0f}д, "
              f"Frequency={profile['frequency']:.1f}, "
              f"Monetary=R${profile['monetary']:.0f}")
        
        if segment_type in strategies:
            print("Рекомендуемые кампании:")
            for i, strategy in enumerate(strategies[segment_type], 1):
                print(f"      {i}. {strategy}")
    
    



    pipeline_metadata = {
        'timestamp': datetime.now().isoformat(),
        'input_dimension': input_dim,
        'encoding_dimension': encoding_dim,
        'n_clusters': n_clusters,
        'n_customers': len(rfm),
        'features_used': X.columns.tolist(),
        'cluster_distribution': rfm['cluster'].value_counts().to_dict(),
        'model_versions': {
            'tensorflow': tf.__version__,
            'keras': keras.__version__
        }
    }
    
    with open('pipeline_metadata.json', 'w') as f:
        json.dump(pipeline_metadata, f, indent=2)
    
    rfm.to_csv('customer_clusters_complete.csv', index=False)
    
