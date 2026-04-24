"""
Association Rule Mining using Apriori and FP-Growth algorithms.
Generates product recommendations with support, confidence, and lift metrics.
Works with new data format.
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules
import os

from config import (
    MIN_SUPPORT,
    MIN_CONFIDENCE,
    MIN_LIFT,
    MIN_LENGTH,
    DATA_DIR
)


DEFAULT_BASKET_PATH = os.path.join(DATA_DIR, "transaction_basket.csv")
DEFAULT_RULES_PATH = os.path.join(DATA_DIR, "..", "models", "association_rules.csv")


def load_transaction_basket(basket_path=None):
    """
    Load transaction basket data for association mining.
    
    Args:
        basket_path (str, optional): Path to basket CSV.
    
    Returns:
        pd.DataFrame: Encoded transaction basket.
    """
    if basket_path is None:
        basket_path = DEFAULT_BASKET_PATH
    
    if not os.path.exists(basket_path):
        raise FileNotFoundError(f"Transaction basket not found at {basket_path}. Run data generation first.")
    
    basket = pd.read_csv(basket_path, index_col=0)
    print(f"[ASSOCIATION] Loaded basket with {basket.shape[0]} transactions, {basket.shape[1]} products.")
    return basket


def find_frequent_itemsets(basket, algorithm="apriori", min_support=MIN_SUPPORT):
    """
    Find frequent itemsets using Apriori or FP-Growth.
    
    Args:
        basket (pd.DataFrame): Encoded transaction basket.
        algorithm (str): "apriori" or "fpgrowth".
        min_support (float): Minimum support threshold.
    
    Returns:
        pd.DataFrame: Frequent itemsets.
    """
    print(f"[ASSOCIATION] Finding frequent itemsets using {algorithm} (min_support={min_support})...")
    
    if algorithm == "apriori":
        itemsets = apriori(basket, min_support=min_support, use_colnames=True, low_memory=True)
    elif algorithm == "fpgrowth":
        itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
    else:
        raise ValueError("Algorithm must be 'apriori' or 'fpgrowth'")
    
    print(f"[ASSOCIATION] Found {len(itemsets)} frequent itemsets.")
    return itemsets


def generate_rules(itemsets, metric="lift", min_threshold=MIN_LIFT, min_confidence=MIN_CONFIDENCE):
    """
    Generate association rules from frequent itemsets.
    
    Args:
        itemsets (pd.DataFrame): Frequent itemsets.
        metric (str): "confidence", "lift", or "leverage".
        min_threshold (float): Minimum metric threshold.
        min_confidence (float): Minimum confidence threshold.
    
    Returns:
        pd.DataFrame: Association rules.
    """
    print(f"[ASSOCIATION] Generating rules (metric={metric}, threshold={min_threshold})...")
    
    rules = association_rules(
        itemsets, 
        metric=metric, 
        min_threshold=min_threshold,
        support_only=False
    )
    
    # Filter by minimum confidence and rule length
    rules = rules[rules["confidence"] >= min_confidence]
    rules = rules[rules["consequents"].apply(lambda x: len(x) >= MIN_LENGTH)]
    
    # Sort by lift (most interesting first)
    rules = rules.sort_values("lift", ascending=False)
    
    print(f"[ASSOCIATION] Generated {len(rules)} rules.")
    return rules


def format_rules_for_display(rules, top_n=20):
    """
    Format association rules for Streamlit display.
    
    Args:
        rules (pd.DataFrame): Association rules.
        top_n (int): Number of top rules to return.
    
    Returns:
        pd.DataFrame: Formatted rules with readable antecedents/consequents.
    """
    def format_items(items):
        return ", ".join(list(items))
    
    formatted = rules.head(top_n).copy()
    formatted["antecedents"] = formatted["antecedents"].apply(format_items)
    formatted["consequents"] = formatted["consequents"].apply(format_items)
    
    # Round metrics for display
    display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    formatted[display_cols[2:]] = formatted[display_cols[2:]].round(4)
    
    return formatted[display_cols]


def get_product_recommendations(product, rules, top_n=5):
    """
    Get top recommendations for a specific product.
    
    Args:
        product (str): Product to find recommendations for.
        rules (pd.DataFrame): Association rules.
        top_n (int): Number of top recommendations.
    
    Returns:
        pd.DataFrame: Top recommended products.
    """
    # Find rules where product is in antecedents
    product_rules = rules[
        rules["antecedents"].apply(lambda x: product in x)
    ].head(top_n)
    
    if len(product_rules) == 0:
        # Fallback: most popular consequents
        fallback = rules.nlargest(top_n, "lift")[["consequents", "confidence", "lift"]]
        fallback["antecedents"] = product
        return fallback
    
    return product_rules


def get_recommended_for_you(customer_id, transaction_df, rules, top_n=10):
    """
    Generate personalized recommendations for a customer.
    
    Args:
        customer_id (str): Customer ID.
        transaction_df (pd.DataFrame): Transaction data.
        rules (pd.DataFrame): Association rules.
        top_n (int): Number of recommendations.
    
    Returns:
        list: Personalized product recommendations.
    """
    # Get customer's recent purchases (last 30 days)
    recent_cutoff = transaction_df["timestamp"].max() - pd.Timedelta(days=30)
    customer_recent = transaction_df[
        (transaction_df["customer_id"] == customer_id) & 
        (transaction_df["timestamp"] >= recent_cutoff)
    ]["product_id"].unique()
    
    recommendations = []
    
    for product in customer_recent:
        recs = get_product_recommendations(product, rules, top_n=3)
        for _, row in recs.iterrows():
            rec_product = list(row["consequents"])[0]
            if rec_product not in customer_recent:
                recommendations.append({
                    "product": rec_product,
                    "confidence": row["confidence"],
                    "lift": row["lift"],
                    "from_product": list(row["antecedents"])[0]
                })
    
    # Sort by confidence and take top N
    recommendations = sorted(recommendations, key=lambda x: x["confidence"], reverse=True)[:top_n]
    return recommendations


def train_and_save_rules(basket=None):
    """
    Full pipeline: load basket, train both algorithms, generate rules, save results.
    
    Args:
        basket (pd.DataFrame, optional): Pre-loaded basket.
    """
    print("=" * 60)
    print("TRAINING ASSOCIATION RULES")
    print("=" * 60)
    
    if basket is None:
        basket = load_transaction_basket()
    
    # Find frequent itemsets with both algorithms
    apriori_itemsets = find_frequent_itemsets(basket, "apriori")
    fpgrowth_itemsets = find_frequent_itemsets(basket, "fpgrowth")
    
    # Generate rules
    apriori_rules = generate_rules(apriori_itemsets)
    fpgrowth_rules = generate_rules(fpgrowth_itemsets)
    
    # Use FP-Growth rules (usually faster and similar quality)
    final_rules = fpgrowth_rules
    
    # Save rules
    os.makedirs(os.path.dirname(DEFAULT_RULES_PATH), exist_ok=True)
    final_rules.to_csv(DEFAULT_RULES_PATH, index=False)
    print(f"[ASSOCIATION] Rules saved to {DEFAULT_RULES_PATH}")
    
    print("=" * 60)
    print("ASSOCIATION RULE MINING COMPLETE")
    print("=" * 60)
    
    return {
        "rules": final_rules,
        "apriori_itemsets": apriori_itemsets,
        "fpgrowth_itemsets": fpgrowth_itemsets
    }


def load_rules(path=None):
    """Load pre-trained association rules."""
    if path is None:
        path = DEFAULT_RULES_PATH
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Rules not found at {path}. Run train_and_save_rules() first.")
    
    rules = pd.read_csv(path)
    
    # Convert string representations back to frozensets if needed
    if isinstance(rules["antecedents"].iloc[0], str):
        rules["antecedents"] = rules["antecedents"].apply(lambda x: frozenset([i.strip() for i in x.strip("{}").split(",")]))
        rules["consequents"] = rules["consequents"].apply(lambda x: frozenset([i.strip() for i in x.strip("{}").split(",")]))
    
    print(f"[ASSOCIATION] Loaded {len(rules)} rules.")
    return rules


if __name__ == "__main__":
    train_and_save_rules()
