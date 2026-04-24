"""
Training script for all recommendation models.
Run this before starting the Streamlit app.
"""

import os
import sys
import pandas as pd

from config import DATA_DIR, MODELS_DIR
from data_generator import generate_all_datasets
from preprocessing import run_preprocessing
from torch_model import train_ncf
from content_based import build_and_save_content_model
from association import train_and_save_rules, load_transaction_basket
from model import train_and_save_models
from utils import create_rfm_features, create_transaction_basket


def check_data_exists():
    """Check if datasets already exist."""
    return all(os.path.exists(p) for p in [
        f"{DATA_DIR}/products.csv",
        f"{DATA_DIR}/customers.csv",
        f"{DATA_DIR}/interactions.csv"
    ])


def main():
    print("=" * 70)
    print("HYBRID RECOMMENDATION SYSTEM - MODEL TRAINING")
    print("=" * 70)
    
    # Step 1: Generate data if not exists
    if not check_data_exists():
        print("\n[1/6] Generating datasets...")
        generate_all_datasets()
    else:
        print("\n[1/6] Datasets already exist, skipping generation.")
    
    # Step 2: Preprocessing
    print("\n[2/6] Running preprocessing...")
    products, customers, interactions, matrix = run_preprocessing()
    
    # Step 3: Train NCF (PyTorch)
    print("\n[3/6] Training Neural Collaborative Filtering model...")
    train_ncf(interactions)
    
    # Step 4: Build Content-Based model
    print("\n[4/6] Building Content-Based model...")
    build_and_save_content_model(products)
    
    # Step 5: Train Association Rules
    print("\n[5/6] Training Association Rules...")
    basket = create_transaction_basket(interactions)
    train_and_save_rules(basket)
    
    # Step 6: Train clustering models
    print("\n[6/6] Training clustering models...")
    from utils import clean_transaction_data, load_data
    raw = load_data()
    rfm = create_rfm_features(raw)
    train_and_save_models(rfm)
    
    print("\n" + "=" * 70)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("You can now run: streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

