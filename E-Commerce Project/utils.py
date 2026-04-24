"""
Utility functions for the Hybrid Recommendation System.
Works with new data format: products.csv, customers.csv, interactions.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from config import (
    DATA_DIR,
    PRODUCTS_PATH,
    CUSTOMERS_PATH,
    INTERACTIONS_PATH,
    RANDOM_STATE
)

np.random.seed(RANDOM_STATE)


def load_interactions(path=INTERACTIONS_PATH):
    """Load interactions data."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_products(path=PRODUCTS_PATH):
    """Load products catalog."""
    return pd.read_csv(path)


def load_customers(path=CUSTOMERS_PATH):
    """Load customers data."""
    return pd.read_csv(path)


def create_rfm_features(interactions_df=None):
    """
    Create Recency, Frequency, and Monetary features for each customer.
    Uses the new interactions data format.
    
    Args:
        interactions_df: Interaction DataFrame (uses INTERACTIONS_PATH if None)
    
    Returns:
        pd.DataFrame: Customer-level RFM features + category preferences.
    """
    if interactions_df is None:
        interactions_df = load_interactions()
    
    print("[UTILS] Creating RFM features...")
    
    # Reference date (day after last transaction)
    reference_date = interactions_df["timestamp"].max() + timedelta(days=1)
    
    # Recency: days since last purchase
    recency = interactions_df.groupby("customer_id")["timestamp"].max().apply(
        lambda x: (reference_date - x).days
    ).reset_index()
    recency.columns = ["customer_id", "recency"]
    
    # Frequency: number of transactions
    frequency = interactions_df.groupby("customer_id").size().reset_index(name="frequency")
    
    # Monetary: total amount spent
    monetary = interactions_df.groupby("customer_id")["purchase_amount"].sum().reset_index()
    monetary.columns = ["customer_id", "monetary"]
    
    # Average order value
    aov = interactions_df.groupby("customer_id")["purchase_amount"].mean().reset_index()
    aov.columns = ["customer_id", "avg_order_value"]
    
    # Category preferences (merge with products first)
    products = load_products()
    merged = interactions_df.merge(products[["product_id", "category"]], on="product_id", how="left")
    
    category_prefs = merged.groupby(["customer_id", "category"]).size().unstack(fill_value=0)
    category_prefs = category_prefs.div(category_prefs.sum(axis=1), axis=0)
    category_prefs = category_prefs.add_prefix("cat_pref_").reset_index()
    
    # Merge all features
    rfm = recency.merge(frequency, on="customer_id")
    rfm = rfm.merge(monetary, on="customer_id")
    rfm = rfm.merge(aov, on="customer_id")
    rfm = rfm.merge(category_prefs, on="customer_id", how="left")
    
    # Fill any NaN category preferences with 0
    cat_cols = [col for col in rfm.columns if col.startswith("cat_pref_")]
    rfm[cat_cols] = rfm[cat_cols].fillna(0)
    
    print(f"[UTILS] RFM features created for {len(rfm)} customers.")
    return rfm


def create_transaction_basket(interactions_df=None):
    """
    Create a transaction basket dataset for association rule mining.
    Each row is a basket (customer + date) with products encoded as columns.
    
    Args:
        interactions_df: Interaction DataFrame (uses INTERACTIONS_PATH if None)
    
    Returns:
        pd.DataFrame: Encoded transaction basket.
    """
    if interactions_df is None:
        interactions_df = load_interactions()
    
    print("[UTILS] Creating transaction basket for association mining...")
    
    # Only use purchase interactions for basket analysis
    purchases = interactions_df[interactions_df["interaction_type"] == "purchase"].copy()
    
    # Create basket_id from customer_id + date
    purchases["basket_id"] = purchases["customer_id"] + "_" + purchases["timestamp"].dt.date.astype(str)
    
    # We need product names for association rules (not IDs)
    products = load_products()
    purchases = purchases.merge(products[["product_id", "subcategory"]], on="product_id", how="left")
    purchases["product_name"] = purchases["subcategory"].fillna(purchases["product_id"])
    
    # Create basket encoding
    basket = purchases.groupby(["basket_id", "product_name"]).size().unstack(fill_value=0)
    
    # Convert to binary (1 if product bought, 0 otherwise)
    basket = (basket > 0).astype(int)
    
    print(f"[UTILS] Basket created with {basket.shape[0]} transactions and {basket.shape[1]} products.")
    return basket


def save_rfm_features(rfm_df, path=None):
    """Save RFM features to CSV."""
    if path is None:
        path = os.path.join(DATA_DIR, "rfm_features.csv")
    rfm_df.to_csv(path, index=False)
    print(f"[UTILS] RFM features saved to {path}")


def save_basket(basket, path=None):
    """Save transaction basket to CSV."""
    if path is None:
        path = os.path.join(DATA_DIR, "transaction_basket.csv")
    basket.to_csv(path)
    print(f"[UTILS] Transaction basket saved to {path}")


# Backward compatibility
def load_data(path=None):
    """Backward compatible load data function."""
    if path is None:
        path = INTERACTIONS_PATH
    return load_interactions(path)


# Convenience function
def generate_and_save_dataset():
    """Generate and save all derived datasets."""
    print("=" * 60)
    print("GENERATING DERIVED DATASETS")
    print("=" * 60)
    
    rfm = create_rfm_features()
    save_rfm_features(rfm)
    
    basket = create_transaction_basket()
    save_basket(basket)
    
    print("=" * 60)
    print("DERIVED DATASETS COMPLETE")
    print("=" * 60)
    
    return rfm, basket


if __name__ == "__main__":
    generate_and_save_dataset()
