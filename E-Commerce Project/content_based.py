"""
Content-Based Filtering using product metadata.
Computes TF-IDF on product descriptions and cosine similarity for recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os

from config import CONTENT_MATRIX_PATH, PRODUCTS_PATH


class ContentBasedRecommender:
    """
    Content-based recommendation engine using product features.
    Combines TF-IDF on descriptions with numerical feature scaling.
    """
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=100)
        self.scaler = StandardScaler()
        self.similarity_matrix = None
        self.product_features = None
        self.product_ids = None
        self.is_fitted = False
    
    def _prepare_features(self, products_df):
        """
        Combine text and numerical features into a single feature vector.
        
        Args:
            products_df: Product DataFrame with descriptions, prices, ratings, etc.
        
        Returns:
            np.ndarray: Combined feature matrix
        """
        # Text features: TF-IDF on description + category + brand
        products_df = products_df.copy()
        products_df['text_features'] = (
            products_df['category'] + ' ' +
            products_df['brand'] + ' ' +
            products_df['description']
        )
        
        tfidf_matrix = self.tfidf.fit_transform(products_df['text_features'])
        
        # Numerical features: price, rating, n_ratings
        numerical = products_df[['price', 'rating', 'n_ratings']].values
        numerical_scaled = self.scaler.fit_transform(numerical)
        
        # Combine text and numerical features
        combined = np.hstack([
            tfidf_matrix.toarray(),
            numerical_scaled
        ])
        
        return combined
    
    def fit(self, products_df):
        """
        Fit the content-based model on product catalog.
        
        Args:
            products_df: Product DataFrame
        """
        print("[CONTENT] Fitting content-based model...")
        
        self.product_ids = products_df['product_id'].tolist()
        
        # Prepare combined features
        features = self._prepare_features(products_df)
        self.product_features = features
        
        # Compute cosine similarity matrix
        self.similarity_matrix = cosine_similarity(features)
        
        self.is_fitted = True
        print(f"[CONTENT] Similarity matrix shape: {self.similarity_matrix.shape}")
        
        return self
    
    def get_similar_products(self, product_id, n_recommendations=10):
        """
        Get similar products based on content similarity.
        
        Args:
            product_id: Target product ID
            n_recommendations: Number of similar products
        
        Returns:
            list of tuples: (similar_product_id, similarity_score)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        
        if product_id not in self.product_ids:
            return []
        
        idx = self.product_ids.index(product_id)
        
        # Get similarity scores for all products
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort by similarity (excluding self)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
        
        # Return top N
        top_n = sim_scores[:n_recommendations]
        return [(self.product_ids[i], float(score)) for i, score in top_n]
    
    def get_product_vector(self, product_id):
        """Get feature vector for a product."""
        if product_id not in self.product_ids:
            return None
        idx = self.product_ids.index(product_id)
        return self.product_features[idx]
    
    def compute_content_score(self, product_id_1, product_id_2):
        """
        Compute content similarity score between two products (0-1).
        """
        if not self.is_fitted:
            return 0.0
        
        if product_id_1 not in self.product_ids or product_id_2 not in self.product_ids:
            return 0.0
        
        idx1 = self.product_ids.index(product_id_1)
        idx2 = self.product_ids.index(product_id_2)
        
        # Cosine similarity is already 0-1 for positive vectors
        score = self.similarity_matrix[idx1][idx2]
        return float(score)
    
    def save(self, path=CONTENT_MATRIX_PATH):
        """Save fitted model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'similarity_matrix': self.similarity_matrix,
                'product_features': self.product_features,
                'product_ids': self.product_ids,
                'tfidf': self.tfidf,
                'scaler': self.scaler
            }, f)
        print(f"[CONTENT] Model saved to {path}")
    
    def load(self, path=CONTENT_MATRIX_PATH):
        """Load fitted model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.similarity_matrix = data['similarity_matrix']
        self.product_features = data['product_features']
        self.product_ids = data['product_ids']
        self.tfidf = data['tfidf']
        self.scaler = data['scaler']
        self.is_fitted = True
        
        print(f"[CONTENT] Model loaded from {path}")
        return self


def build_and_save_content_model(products_df):
    """Build and save content-based model."""
    print("=" * 60)
    print("BUILDING CONTENT-BASED MODEL")
    print("=" * 60)
    
    model = ContentBasedRecommender()
    model.fit(products_df)
    model.save()
    
    print("[CONTENT] Content-based model ready.")
    return model


if __name__ == "__main__":
    from data_generator import generate_products
    products = generate_products()
    build_and_save_content_model(products)
