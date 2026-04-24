"""
Enhanced E-Commerce Data Generator with ratings, brands, descriptions, and metadata.
Produces realistic data for hybrid recommendation system training.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import json

from config import (
    DATA_DIR, RANDOM_STATE, N_CUSTOMERS, N_PRODUCTS, N_INTERACTIONS
)

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ============================================================================
# PRODUCT CATALOG WITH RICH METADATA
# ============================================================================

PRODUCT_CATALOG = {
    "Electronics": {
        "brands": ["Apple", "Samsung", "Sony", "OnePlus", "HP", "Dell", "Logitech", "Bose"],
        "items": {
            "mobile": {"price_range": (15000, 120000), "rating": 4.2, "desc": "Smartphone with advanced camera and 5G connectivity"},
            "laptop": {"price_range": (35000, 150000), "rating": 4.4, "desc": "High-performance laptop for work and gaming"},
            "tablet": {"price_range": (20000, 80000), "rating": 4.1, "desc": "Portable tablet with stylus support"},
            "earphones": {"price_range": (1500, 25000), "rating": 4.3, "desc": "Wireless earbuds with noise cancellation"},
            "mouse": {"price_range": (800, 8000), "rating": 4.5, "desc": "Ergonomic wireless mouse with precision tracking"},
            "keyboard": {"price_range": (1500, 12000), "rating": 4.4, "desc": "Mechanical keyboard with RGB backlight"},
            "monitor": {"price_range": (12000, 60000), "rating": 4.3, "desc": "4K Ultra HD monitor with HDR support"},
            "charger": {"price_range": (500, 3000), "rating": 4.0, "desc": "Fast charging adapter with multiple ports"},
            "smartwatch": {"price_range": (5000, 45000), "rating": 4.2, "desc": "Fitness tracking smartwatch with heart rate monitor"},
            "camera": {"price_range": (25000, 150000), "rating": 4.6, "desc": "Mirrorless camera with 4K video recording"}
        }
    },
    "Fashion": {
        "brands": ["Nike", "Adidas", "Zara", "H&M", "Levi's", "Puma", "Gucci", "Ray-Ban"],
        "items": {
            "dress": {"price_range": (1200, 15000), "rating": 4.1, "desc": "Elegant evening dress for special occasions"},
            "heels": {"price_range": (1500, 12000), "rating": 4.0, "desc": "Stylish high heels with comfortable padding"},
            "shirt": {"price_range": (800, 5000), "rating": 4.3, "desc": "Formal cotton shirt with wrinkle resistance"},
            "jeans": {"price_range": (1500, 8000), "rating": 4.4, "desc": "Slim fit denim jeans with stretch fabric"},
            "watch": {"price_range": (2000, 50000), "rating": 4.5, "desc": "Classic analog watch with leather strap"},
            "sunglasses": {"price_range": (1000, 15000), "rating": 4.2, "desc": "UV protection sunglasses with polarized lenses"},
            "handbag": {"price_range": (2000, 25000), "rating": 4.3, "desc": "Designer leather handbag with multiple compartments"},
            "shoes": {"price_range": (2000, 12000), "rating": 4.4, "desc": "Running shoes with cushioned sole"},
            "jacket": {"price_range": (3000, 20000), "rating": 4.3, "desc": "Waterproof winter jacket with thermal lining"},
            "tshirt": {"price_range": (500, 3000), "rating": 4.2, "desc": "Casual graphic t-shirt with premium cotton"}
        }
    },
    "Home": {
        "brands": ["IKEA", "Home Centre", "Urban Ladder", "Philips", "Borosil", "Prestige"],
        "items": {
            "sofa": {"price_range": (15000, 80000), "rating": 4.3, "desc": "3-seater fabric sofa with washable covers"},
            "lamp": {"price_range": (800, 6000), "rating": 4.2, "desc": "LED table lamp with adjustable brightness"},
            "curtains": {"price_range": (1500, 8000), "rating": 4.0, "desc": "Blackout curtains with thermal insulation"},
            "cushion": {"price_range": (300, 2000), "rating": 4.1, "desc": "Decorative cushion covers with velvet finish"},
            "mirror": {"price_range": (1000, 10000), "rating": 4.3, "desc": "Full-length wall mirror with LED border"},
            "clock": {"price_range": (500, 4000), "rating": 4.0, "desc": "Vintage wall clock with silent movement"},
            "vase": {"price_range": (600, 5000), "rating": 4.1, "desc": "Ceramic flower vase with hand-painted design"},
            "rug": {"price_range": (2000, 15000), "rating": 4.2, "desc": "Persian style area rug with anti-slip backing"},
            "bedsheet": {"price_range": (800, 5000), "rating": 4.4, "desc": "King size cotton bedsheet with pillow covers"},
            "cookware": {"price_range": (1500, 12000), "rating": 4.5, "desc": "Non-stick cookware set with induction base"}
        }
    },
    "Beauty": {
        "brands": ["L'Oreal", "Maybelline", "The Body Shop", "MAC", "Nivea", "Lakme", "Forest Essentials"],
        "items": {
            "perfume": {"price_range": (1500, 12000), "rating": 4.4, "desc": "Long-lasting Eau de Parfum with floral notes"},
            "lipstick": {"price_range": (500, 3000), "rating": 4.3, "desc": "Matte finish lipstick with 12-hour stay"},
            "moisturizer": {"price_range": (300, 2500), "rating": 4.5, "desc": "Hydrating face moisturizer with SPF 30"},
            "shampoo": {"price_range": (300, 1500), "rating": 4.2, "desc": "Sulfate-free shampoo for damaged hair"},
            "serum": {"price_range": (800, 5000), "rating": 4.6, "desc": "Vitamin C serum for glowing skin"},
            "mask": {"price_range": (200, 1500), "rating": 4.1, "desc": "Charcoal face mask for deep cleansing"},
            "cream": {"price_range": (400, 3000), "rating": 4.3, "desc": "Anti-aging night cream with retinol"},
            "liner": {"price_range": (300, 1500), "rating": 4.2, "desc": "Waterproof eyeliner with precision tip"},
            "sunscreen": {"price_range": (400, 2000), "rating": 4.5, "desc": "Broad spectrum SPF 50 sunscreen"},
            "facewash": {"price_range": (200, 1200), "rating": 4.3, "desc": "Gentle face wash for all skin types"}
        }
    },
    "Sports": {
        "brands": ["Nike", "Adidas", "Decathlon", "Yonex", "Wilson", "Reebok", "Puma"],
        "items": {
            "yoga_mat": {"price_range": (800, 4000), "rating": 4.4, "desc": "Non-slip yoga mat with alignment lines"},
            "dumbbell": {"price_range": (500, 3000), "rating": 4.5, "desc": "Adjustable dumbbell set with quick lock"},
            "bottle": {"price_range": (300, 1500), "rating": 4.2, "desc": "Insulated stainless steel water bottle"},
            "shoes": {"price_range": (2500, 15000), "rating": 4.4, "desc": "Cross-training shoes with breathable mesh"},
            "gloves": {"price_range": (400, 2500), "rating": 4.1, "desc": "Gym gloves with wrist support"},
            "ball": {"price_range": (500, 3000), "rating": 4.3, "desc": "Official size football with butyl bladder"},
            "racket": {"price_range": (1500, 12000), "rating": 4.4, "desc": "Graphite badminton racket with string tension"},
            "band": {"price_range": (300, 1500), "rating": 4.2, "desc": "Resistance band set with 5 strength levels"},
            "treadmill": {"price_range": (20000, 100000), "rating": 4.3, "desc": "Motorized treadmill with incline feature"},
            "cycle": {"price_range": (8000, 50000), "rating": 4.4, "desc": "Mountain bike with 21-speed gear system"}
        }
    }
}


def generate_products(n_products=N_PRODUCTS):
    """
    Generate product catalog with rich metadata.
    
    Returns:
        pd.DataFrame: Product catalog with descriptions, brands, prices, ratings.
    """
    print("[DATA] Generating product catalog...")
    
    products = []
    product_id = 1
    
    for category, data in PRODUCT_CATALOG.items():
        for item_name, item_data in data["items"].items():
            # Create multiple variants per product
            n_variants = random.randint(2, 4)
            for _ in range(n_variants):
                brand = random.choice(data["brands"])
                price_min, price_max = item_data["price_range"]
                price = round(random.uniform(price_min, price_max), 2)
                
                # Rating with some variance
                base_rating = item_data["rating"]
                rating = round(min(5.0, max(1.0, base_rating + random.uniform(-0.5, 0.5))), 1)
                n_ratings = random.randint(50, 5000)
                
                products.append({
                    "product_id": f"PROD_{product_id:05d}",
                    "product_name": f"{brand} {item_name.title()}",
                    "category": category,
                    "subcategory": item_name,
                    "brand": brand,
                    "price": price,
                    "rating": rating,
                    "n_ratings": n_ratings,
                    "description": item_data["desc"],
                    "image_url": f"https://placehold.co/300x200?text={brand}+{item_name.title()}"
                })
                product_id += 1
                
                if len(products) >= n_products:
                    break
        if len(products) >= n_products:
            break
    
    df = pd.DataFrame(products)
    print(f"[DATA] Generated {len(df)} products across {df['category'].nunique()} categories.")
    return df


def generate_customers(n_customers=N_CUSTOMERS):
    """
    Generate customer profiles with demographics.
    
    Returns:
        pd.DataFrame: Customer data with age, gender, location.
    """
    print("[DATA] Generating customer profiles...")
    
    first_names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Amir", "Ayaan",
                   "Aanya", "Diya", "Saanvi", "Ananya", "Navya", "Myra", "Ira", "Sara"]
    last_names = ["Sharma", "Kumar", "Singh", "Patel", "Gupta", "Reddy", "Nair", "Desai",
                  "Shah", "Mehta", "Joshi", "Rao", "Malhotra", "Agarwal", "Iyer"]
    
    cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad",
              "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam"]
    
    customers = []
    for i in range(1, n_customers + 1):
        gender = random.choice(["M", "F"])
        first = random.choice(first_names)
        last = random.choice(last_names)
        
        customers.append({
            "customer_id": f"CUST_{i:04d}",
            "name": f"{first} {last}",
            "email": f"{first.lower()}.{last.lower()}{i}@email.com",
            "age": random.randint(18, 65),
            "gender": gender,
            "city": random.choice(cities),
            "registration_date": datetime(2022, 1, 1) + timedelta(days=random.randint(0, 500))
        })
    
    df = pd.DataFrame(customers)
    print(f"[DATA] Generated {len(df)} customers.")
    return df


def generate_interactions(customers_df, products_df, n_interactions=N_INTERACTIONS):
    """
    Generate user-product interactions with ratings and timestamps.
    Simulates realistic purchase behaviour with category preferences.
    
    Returns:
        pd.DataFrame: Interaction data (customer_id, product_id, rating, timestamp, purchase_amount)
    """
    print("[DATA] Generating user interactions...")
    
    customer_ids = customers_df["customer_id"].tolist()
    product_ids = products_df["product_id"].tolist()
    
    # Create customer category preferences
    categories = products_df["category"].unique()
    customer_prefs = {}
    for cust_id in customer_ids:
        # Each customer has 2-3 preferred categories
        prefs = random.sample(list(categories), k=random.randint(2, 3))
        weights = np.random.dirichlet(np.ones(len(prefs))) * 0.7 + 0.1
        customer_prefs[cust_id] = dict(zip(prefs, weights))
    
    interactions = []
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 6, 30)
    date_range = (end_date - start_date).days
    
    for _ in range(n_interactions):
        customer_id = random.choice(customer_ids)
        
        # Select product based on customer preferences
        cust_cats = customer_prefs[customer_id]
        category = random.choices(list(cust_cats.keys()), weights=list(cust_cats.values()))[0]
        
        # Filter products by preferred category
        cat_products = products_df[products_df["category"] == category]
        product = cat_products.sample(1).iloc[0]
        
        product_id = product["product_id"]
        price = product["price"]
        
        # Rating influenced by product rating + customer satisfaction variance
        base_rating = product["rating"]
        rating = min(5.0, max(1.0, base_rating + random.uniform(-1.0, 0.5)))
        rating = round(rating * 2) / 2  # Round to nearest 0.5
        
        # Timestamp
        random_days = random.randint(0, date_range)
        timestamp = start_date + timedelta(days=random_days)
        
        # Purchase amount (may include multiple quantities)
        quantity = random.choices([1, 2, 3], weights=[0.75, 0.20, 0.05])[0]
        purchase_amount = round(price * quantity, 2)
        
        # Interaction type
        interaction_type = random.choices(
            ["purchase", "cart_add", "view"],
            weights=[0.6, 0.25, 0.15]
        )[0]
        
        interactions.append({
            "interaction_id": f"INT_{len(interactions)+1:07d}",
            "customer_id": customer_id,
            "product_id": product_id,
            "rating": rating,
            "timestamp": timestamp,
            "quantity": quantity,
            "purchase_amount": purchase_amount,
            "interaction_type": interaction_type
        })
    
    df = pd.DataFrame(interactions)
    
    # Add duplicates and missing values for realism
    n_duplicates = int(len(df) * 0.01)
    duplicates = df.sample(n=n_duplicates, replace=True)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    n_missing = int(len(df) * 0.02)
    missing_idx = random.sample(range(len(df)), n_missing)
    df.loc[missing_idx[:n_missing//2], "rating"] = np.nan
    df.loc[missing_idx[n_missing//2:], "interaction_type"] = np.nan
    
    print(f"[DATA] Generated {len(df)} interactions.")
    return df


def save_datasets(products_df, customers_df, interactions_df):
    """Save all datasets to CSV files."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    products_df.to_csv(f"{DATA_DIR}/products.csv", index=False)
    customers_df.to_csv(f"{DATA_DIR}/customers.csv", index=False)
    interactions_df.to_csv(f"{DATA_DIR}/interactions.csv", index=False)
    
    print(f"[DATA] Datasets saved to {DATA_DIR}/")
    print(f"  - products.csv: {len(products_df)} rows")
    print(f"  - customers.csv: {len(customers_df)} rows")
    print(f"  - interactions.csv: {len(interactions_df)} rows")


def generate_all_datasets():
    """Generate complete dataset pipeline."""
    print("=" * 60)
    print("GENERATING HYBRID RECOMMENDATION DATASETS")
    print("=" * 60)
    
    products_df = generate_products()
    customers_df = generate_customers()
    interactions_df = generate_interactions(customers_df, products_df)
    
    save_datasets(products_df, customers_df, interactions_df)
    
    print("=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    
    return products_df, customers_df, interactions_df


if __name__ == "__main__":
    generate_all_datasets()
