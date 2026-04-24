"""
Data Preprocessing Pipeline for Hybrid Recommendation System.
Handles cleaning, encoding, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

from config import DATA_DIR, PRODUCTS_PATH, CUSTOMERS_PATH, INTERACTIONS_PATH, ENCODERS_PATH


def load_raw_data():
    """Load raw datasets from CSV files."""
    products = pd.read_csv(PRODUCTS_PATH)
    customers = pd.read_csv(CUSTOMERS_PATH)
    interactions = pd.read_csv(INTERACTIONS_PATH)
    
    # Parse dates
    interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
    customers['registration_date'] = pd.to_datetime(customers['registration_date'])
    
    return products, customers, interactions


def clean_interactions(interactions_df):
    """Clean interaction data."""
    df = interactions_df.copy()
    
    # Remove duplicates
    initial = len(df)
    df = df.drop_duplicates(subset=['customer_id', 'product_id', 'timestamp'])
    
    # Handle missing ratings (impute with product average)
    df['rating'] = df.groupby('product_id')['rating'].transform(
        lambda x: x.fillna(x.mean())
    )
    df['rating'] = df['rating'].fillna(3.0)
    
    # Handle missing interaction type
    df['interaction_type'] = df['interaction_type'].fillna('view')
    
    # Clip ratings to 1-5 range
    df['rating'] = df['rating'].clip(1.0, 5.0)
    
    print(f"[PREP] Cleaned interactions: {initial} -> {len(df)} rows")
    return df


def create_user_item_matrix(interactions_df):
    """Create user-item interaction matrix."""
    matrix = interactions_df.pivot_table(
        index='customer_id',
        columns='product_id',
        values='rating',
        aggfunc='mean',
        fill_value=0
    )
    return matrix


def create_features_for_clustering(customers_df, interactions_df):
    """Create customer features for clustering/segmentation."""
    # RFM features
    reference_date = interactions_df['timestamp'].max() + pd.Timedelta(days=1)
    
    rfm = interactions_df.groupby('customer_id').agg({
        'timestamp': lambda x: (reference_date - x.max()).days,  # Recency
        'interaction_id': 'count',  # Frequency
        'purchase_amount': 'sum'  # Monetary
    }).reset_index()
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Average rating given
    avg_rating = interactions_df.groupby('customer_id')['rating'].mean().reset_index()
    avg_rating.columns = ['customer_id', 'avg_rating']
    
    # Category preferences
    cat_prefs = interactions_df.merge(
        interactions_df[['product_id']].drop_duplicates(),
        on='product_id',
        how='left'
    )
    # Get categories from products - will be merged later
    
    # Merge all
    features = rfm.merge(avg_rating, on='customer_id', how='left')
    features['avg_rating'] = features['avg_rating'].fillna(3.0)
    
    return features


def encode_ids(interactions_df):
    """Encode customer_id and product_id to numeric indices."""
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    interactions_df = interactions_df.copy()
    interactions_df['user_idx'] = user_encoder.fit_transform(interactions_df['customer_id'])
    interactions_df['item_idx'] = item_encoder.fit_transform(interactions_df['product_id'])
    
    return interactions_df, user_encoder, item_encoder


def save_encoders(user_encoder, item_encoder, path=ENCODERS_PATH):
    """Save label encoders."""
    with open(path, 'wb') as f:
        pickle.dump({'user_encoder': user_encoder, 'item_encoder': item_encoder}, f)


def load_encoders(path=ENCODERS_PATH):
    """Load label encoders."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['user_encoder'], data['item_encoder']


def get_train_test_split(interactions_df, test_ratio=0.2):
    """Split interactions into train and test sets."""
    interactions_df = interactions_df.sort_values('timestamp')
    
    split_idx = int(len(interactions_df) * (1 - test_ratio))
    train = interactions_df.iloc[:split_idx]
    test = interactions_df.iloc[split_idx:]
    
    return train, test


# Complete preprocessing pipeline
def run_preprocessing():
    """Run full preprocessing pipeline."""
    print("=" * 60)
    print("RUNNING PREPROCESSING PIPELINE")
    print("=" * 60)
    
    products, customers, interactions = load_raw_data()
    interactions = clean_interactions(interactions)
    
    # Encode IDs
    interactions, user_encoder, item_encoder = encode_ids(interactions)
    save_encoders(user_encoder, item_encoder)
    
    # Save processed data
    interactions.to_csv(f"{DATA_DIR}/interactions_processed.csv", index=False)
    products.to_csv(f"{DATA_DIR}/products_processed.csv", index=False)
    customers.to_csv(f"{DATA_DIR}/customers_processed.csv", index=False)
    
    # Create and save user-item matrix
    matrix = create_user_item_matrix(interactions)
    matrix.to_csv(f"{DATA_DIR}/user_item_matrix.csv")
    
    print("[PREP] Preprocessing complete.")
    return products, customers, interactions, matrix


if __name__ == "__main__":
    run_preprocessing()

