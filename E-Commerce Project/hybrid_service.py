"""
Hybrid Recommendation Service.
Combines Neural Collaborative Filtering, Content-Based Filtering, and Association Rules.
final_score = (w1 * DL_score) + (w2 * content_score) + (w3 * association_score)
"""

import pandas as pd
import numpy as np
import torch
import pickle
import os
from collections import defaultdict

from config import (
    NCF_MODEL_PATH, USER_ENCODER_PATH, PRODUCT_ENCODER_PATH,
    CONTENT_MATRIX_PATH, ASSOCIATION_RULES_PATH,
    WEIGHT_DL, WEIGHT_CONTENT, WEIGHT_ASSOCIATION
)
from torch_model import NCFModel, DEVICE
from content_based import ContentBasedRecommender
from association import load_rules


class HybridRecommender:
    """
    Hybrid recommendation engine combining multiple algorithms.
    """
    
    def __init__(self, products_df, interactions_df, 
                 weight_dl=WEIGHT_DL, 
                 weight_content=WEIGHT_CONTENT, 
                 weight_association=WEIGHT_ASSOCIATION):
        """
        Initialize hybrid recommender.
        
        Args:
            products_df: Product catalog DataFrame
            interactions_df: User interaction DataFrame
            weight_dl: Weight for deep learning (NCF) scores
            weight_content: Weight for content-based scores
            weight_association: Weight for association rule scores
        """
        self.products_df = products_df.set_index('product_id')
        self.interactions_df = interactions_df
        self.weight_dl = weight_dl
        self.weight_content = weight_content
        self.weight_association = weight_association
        
        # Load components
        self.ncf_model = None
        self.user_encoder = None
        self.item_encoder = None
        self.content_model = None
        self.association_rules = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all trained models."""
        print("[HYBRID] Loading recommendation models...")
        
        # Load NCF
        if os.path.exists(NCF_MODEL_PATH) and os.path.exists(USER_ENCODER_PATH):
            with open(USER_ENCODER_PATH, 'rb') as f:
                self.user_encoder = pickle.load(f)
            with open(PRODUCT_ENCODER_PATH, 'rb') as f:
                self.item_encoder = pickle.load(f)
            
            n_users = len(self.user_encoder.classes_)
            n_items = len(self.item_encoder.classes_)
            
            self.ncf_model = NCFModel(n_users, n_items).to(DEVICE)
            self.ncf_model.load_state_dict(torch.load(NCF_MODEL_PATH, map_location=DEVICE))
            self.ncf_model.eval()
            print("[HYBRID] NCF model loaded.")
        
        # Load content-based
        if os.path.exists(CONTENT_MATRIX_PATH):
            self.content_model = ContentBasedRecommender()
            self.content_model.load(CONTENT_MATRIX_PATH)
            print("[HYBRID] Content-based model loaded.")
        
        # Load association rules
        if os.path.exists(ASSOCIATION_RULES_PATH):
            self.association_rules = load_rules()
            print("[HYBRID] Association rules loaded.")
    
    def get_ncf_score(self, customer_id, product_id):
        """
        Get NCF predicted score for user-item pair.
        
        Returns:
            float: Predicted score (0-1)
        """
        if self.ncf_model is None:
            return 0.5
        
        try:
            user_idx = self.user_encoder.transform([customer_id])[0]
            item_idx = self.item_encoder.transform([product_id])[0]
            
            score = self.ncf_model.predict([user_idx], [item_idx])[0]
            return float(score)
        except:
            return 0.5
    
    def get_content_score(self, customer_id, product_id):
        """
        Get content-based score.
        Based on similarity to user's previously purchased items.
        
        Returns:
            float: Content score (0-1)
        """
        if self.content_model is None:
            return 0.0
        
        # Get user's past purchases
        user_history = self.interactions_df[
            (self.interactions_df['customer_id'] == customer_id) &
            (self.interactions_df['interaction_type'] == 'purchase')
        ]['product_id'].unique()
        
        if len(user_history) == 0:
            return 0.0
        
        # Average similarity to user's past items
        similarities = []
        for past_product in user_history[-10:]:  # Last 10 purchases
            sim = self.content_model.compute_content_score(past_product, product_id)
            similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def get_association_score(self, customer_id, product_id):
        """
        Get association rule score.
        Based on confidence of rules from user's past items to this product.
        
        Returns:
            float: Association score (0-1)
        """
        if self.association_rules is None or len(self.association_rules) == 0:
            return 0.0
        
        user_history = self.interactions_df[
            (self.interactions_df['customer_id'] == customer_id) &
            (self.interactions_df['interaction_type'] == 'purchase')
        ]['product_id'].unique()
        
        if len(user_history) == 0:
            return 0.0
        
        max_confidence = 0.0
        for past_product in user_history:
            # Find rules where past_product -> product_id
            matching_rules = self.association_rules[
                (self.association_rules['antecedents'].apply(lambda x: past_product in x)) &
                (self.association_rules['consequents'].apply(lambda x: product_id in x))
            ]
            
            if len(matching_rules) > 0:
                max_confidence = max(max_confidence, matching_rules['confidence'].max())
        
        return float(min(1.0, max_confidence))
    
    def recommend(self, customer_id, n_recommendations=10, exclude_purchased=True):
        """
        Generate hybrid recommendations for a customer.
        
        Args:
            customer_id: Customer ID
            n_recommendations: Number of recommendations
            exclude_purchased: Whether to exclude already purchased items
        
        Returns:
            pd.DataFrame: Recommendations with scores and reasons
        """
        print(f"[HYBRID] Generating recommendations for {customer_id}...")
        
        # Get candidate products
        all_products = self.products_df.index.tolist()
        
        if exclude_purchased:
            purchased = set(self.interactions_df[
                (self.interactions_df['customer_id'] == customer_id) &
                (self.interactions_df['interaction_type'] == 'purchase')
            ]['product_id'].unique())
            candidates = [p for p in all_products if p not in purchased]
        else:
            candidates = all_products
        
        if len(candidates) == 0:
            candidates = all_products
        
        # Score each candidate
        results = []
        for product_id in candidates[:100]:  # Sample for efficiency
            ncf_score = self.get_ncf_score(customer_id, product_id)
            content_score = self.get_content_score(customer_id, product_id)
            assoc_score = self.get_association_score(customer_id, product_id)
            
            # Hybrid score
            final_score = (
                self.weight_dl * ncf_score +
                self.weight_content * content_score +
                self.weight_association * assoc_score
            )
            
            # Determine primary reason
            scores = {
                'Collaborative Filtering': ncf_score,
                'Content-Based': content_score,
                'Association Rules': assoc_score
            }
            primary_reason = max(scores, key=scores.get)
            
            product_info = self.products_df.loc[product_id]
            
            results.append({
                'product_id': product_id,
                'product_name': product_info['product_name'],
                'category': product_info['category'],
                'brand': product_info['brand'],
                'price': product_info['price'],
                'rating': product_info['rating'],
                'image_url': product_info.get('image_url', ''),
                'final_score': final_score,
                'ncf_score': ncf_score,
                'content_score': content_score,
                'association_score': assoc_score,
                'primary_reason': primary_reason
            })
        
        # Sort by final score
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('final_score', ascending=False).head(n_recommendations)
        
        return results_df
    
    def get_similar_products(self, product_id, n=5):
        """Get similar products using content-based filtering."""
        if self.content_model is None:
            return pd.DataFrame()
        
        similar = self.content_model.get_similar_products(product_id, n)
        
        results = []
        for pid, score in similar:
            if pid in self.products_df.index:
                info = self.products_df.loc[pid]
                results.append({
                    'product_id': pid,
                    'product_name': info['product_name'],
                    'category': info['category'],
                    'price': info['price'],
                    'rating': info['rating'],
                    'similarity_score': score
                })
        
        return pd.DataFrame(results)
    
    def get_trending_products(self, n=10):
        """Get trending products based on recent interactions."""
        recent = self.interactions_df[
            self.interactions_df['timestamp'] >= 
            self.interactions_df['timestamp'].max() - pd.Timedelta(days=30)
        ]
        
        trending = recent.groupby('product_id').agg({
            'interaction_type': 'count',
            'rating': 'mean'
        }).reset_index()
        trending.columns = ['product_id', 'interaction_count', 'avg_rating']
        trending = trending.sort_values('interaction_count', ascending=False).head(n)
        
        # Merge with product info
        results = []
        for _, row in trending.iterrows():
            pid = row['product_id']
            if pid in self.products_df.index:
                info = self.products_df.loc[pid]
                results.append({
                    'product_id': pid,
                    'product_name': info['product_name'],
                    'category': info['category'],
                    'price': info['price'],
                    'rating': info['rating'],
                    'interaction_count': int(row['interaction_count']),
                    'trend_score': float(row['interaction_count'] * row['avg_rating'] / 5.0)
                })
        
        return pd.DataFrame(results)
    
    def get_because_you_bought(self, customer_id, n=5):
        """
        'Because You Bought X' recommendations using association rules.
        """
        if self.association_rules is None:
            return pd.DataFrame()
        
        recent_purchases = self.interactions_df[
            (self.interactions_df['customer_id'] == customer_id) &
            (self.interactions_df['interaction_type'] == 'purchase')
        ].sort_values('timestamp', ascending=False)['product_id'].head(3).tolist()
        
        recommendations = []
        
        for product in recent_purchases:
            matching = self.association_rules[
                self.association_rules['antecedents'].apply(lambda x: product in x)
            ].sort_values('confidence', ascending=False).head(2)
            
            for _, rule in matching.iterrows():
                for consequent in rule['consequents']:
                    if consequent in self.products_df.index:
                        info = self.products_df.loc[consequent]
                        recommendations.append({
                            'product_id': consequent,
                            'product_name': info['product_name'],
                            'category': info['category'],
                            'price': info['price'],
                            'rating': info['rating'],
                            'because_you_bought': product,
                            'confidence': rule['confidence'],
                            'lift': rule['lift']
                        })
        
        # Remove duplicates and sort
        rec_df = pd.DataFrame(recommendations)
        if len(rec_df) > 0:
            rec_df = rec_df.drop_duplicates(subset=['product_id'])
            rec_df = rec_df.sort_values('confidence', ascending=False).head(n)
        
        return rec_df
    
    def update_weights(self, w_dl=None, w_content=None, w_association=None):
        """Update hybrid weights dynamically."""
        if w_dl is not None:
            self.weight_dl = w_dl
        if w_content is not None:
            self.weight_content = w_content
        if w_association is not None:
            self.weight_association = w_association
        
        # Normalize to sum to 1
        total = self.weight_dl + self.weight_content + self.weight_association
        self.weight_dl /= total
        self.weight_content /= total
        self.weight_association /= total
        
        print(f"[HYBRID] Updated weights: DL={self.weight_dl:.2f}, "
              f"Content={self.weight_content:.2f}, Assoc={self.weight_association:.2f}")


# Convenience function for cold-start
def get_cold_start_recommendations(products_df, n=10):
    """
    Recommendations for new users (cold start).
    Returns popular, highly-rated products across categories.
    """
    popular = products_df.nlargest(n, 'n_ratings')
    return popular[['product_id', 'product_name', 'category', 'brand', 
                    'price', 'rating', 'n_ratings', 'description', 'image_url']]


if __name__ == "__main__":
    from data_generator import generate_all_datasets
    products, customers, interactions = generate_all_datasets()
    
    hybrid = HybridRecommender(products, interactions)
    recs = hybrid.recommend("CUST_0001", n_recommendations=5)
    print(recs[['product_name', 'final_score', 'primary_reason']])

