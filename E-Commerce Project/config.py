"""
Configuration constants for the E-Commerce Hybrid Recommendation System.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "ecommerce.db")

# Data file paths
PRODUCTS_PATH = os.path.join(DATA_DIR, "products.csv")
CUSTOMERS_PATH = os.path.join(DATA_DIR, "customers.csv")
INTERACTIONS_PATH = os.path.join(DATA_DIR, "interactions.csv")

# Model file paths
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
NCF_MODEL_PATH = os.path.join(MODELS_DIR, "ncf_model.pt")
CONTENT_MATRIX_PATH = os.path.join(MODELS_DIR, "content_matrix.pkl")
PRODUCT_ENCODER_PATH = os.path.join(MODELS_DIR, "product_encoder.pkl")
USER_ENCODER_PATH = os.path.join(MODELS_DIR, "user_encoder.pkl")
ASSOCIATION_RULES_PATH = os.path.join(MODELS_DIR, "association_rules.csv")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset generation parameters
N_CUSTOMERS = 1000
N_PRODUCTS = 200
N_INTERACTIONS = 15000
RANDOM_STATE = 42

# Clustering parameters
N_CLUSTERS = 4
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5

# NCF Model parameters
EMBEDDING_DIM = 64
HIDDEN_DIMS = [128, 64, 32]
DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 30

# Hybrid weights
WEIGHT_DL = 0.50
WEIGHT_CONTENT = 0.30
WEIGHT_ASSOCIATION = 0.20

# Association rules parameters
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.2
MIN_LIFT = 1.0
MIN_LENGTH = 2

# Product image URLs (placeholder)
PLACEHOLDER_IMAGE = "https://placehold.co/300x200?text=Product"
