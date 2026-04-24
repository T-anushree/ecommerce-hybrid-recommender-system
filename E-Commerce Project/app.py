"""
E-Commerce Hybrid Recommendation System - Streamlit Frontend.
Modern Netflix/Amazon-style UI with personalized recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(
    page_title="ShopSmart AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from auth import init_auth_state, is_authenticated, get_current_user, show_login_form, show_signup_form, logout_user
from db import init_db
from config import DATA_DIR, PRODUCTS_PATH, CUSTOMERS_PATH, INTERACTIONS_PATH, WEIGHT_DL, WEIGHT_CONTENT, WEIGHT_ASSOCIATION

init_db()
init_auth_state()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 3rem;
        margin-bottom: 2rem;
        color: white;
    }
    .hero-title { font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem; }
    .hero-subtitle { font-size: 1.1rem; opacity: 0.9; }
    .product-card {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        height: 100%;
    }
    .product-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    .product-image {
        width: 100%;
        height: 160px;
        object-fit: cover;
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
    }
    .product-info { padding: 1rem; }
    .product-name {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.25rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .product-brand { font-size: 0.75rem; color: #666; margin-bottom: 0.5rem; }
    .product-meta { display: flex; justify-content: space-between; align-items: center; }
    .product-price { font-size: 1rem; font-weight: 700; color: #667eea; }
    .product-rating { font-size: 0.8rem; color: #f59e0b; }
    .section-title { font-size: 1.5rem; font-weight: 700; color: #1a1a2e; margin: 2rem 0 1rem 0; }
    .reason-tag {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .reason-collaborative { background: #dbeafe; color: #1e40af; }
    .reason-content { background: #dcfce7; color: #166534; }
    .reason-association { background: #fef3c7; color: #92400e; }
    .metric-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #1a1a2e; }
    .metric-label { font-size: 0.85rem; color: #666; margin-top: 0.25rem; }
    .cold-start-banner {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #f59e0b;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_all_data():
    products = pd.read_csv(PRODUCTS_PATH)
    customers = pd.read_csv(CUSTOMERS_PATH)
    interactions = pd.read_csv(INTERACTIONS_PATH)
    interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
    return products, customers, interactions

@st.cache_resource(show_spinner=False)
def get_recommender(products_df, interactions_df):
    from hybrid_service import HybridRecommender
    return HybridRecommender(products_df, interactions_df)

def render_product_card(product, show_reason=False):
    price = f"₹{product['price']:,.0f}"
    rating = product.get('rating', 4.0)
    img_url = product.get('image_url', '')
    if not img_url or pd.isna(img_url):
        img_url = f"https://placehold.co/300x200?text={str(product['product_name'])[:15]}"
    reason_html = ""
    if show_reason and 'primary_reason' in product:
        reason_map = {
            'Collaborative Filtering': ('reason-collaborative', 'AI Match'),
            'Content-Based': ('reason-content', 'Similar'),
            'Association Rules': ('reason-association', 'Frequently Bought')
        }
        css_class, label = reason_map.get(product['primary_reason'], ('reason-trending', 'Trending'))
        reason_html = f'<span class="reason-tag {css_class}">{label}</span>'
    return f"""
    <div class="product-card">
        <img class="product-image" src="{img_url}" alt="{product['product_name']}">
        <div class="product-info">
            <div class="product-name">{product['product_name']}</div>
            <div class="product-brand">{product.get('brand', product.get('category', ''))}</div>
            {reason_html}
            <div class="product-meta" style="margin-top:0.5rem;">
                <span class="product-price">{price}</span>
                <span class="product-rating">⭐ {rating:.1f}</span>
            </div>
        </div>
    </div>
    """

def render_product_row(products_df, cols_per_row=6, show_reason=False):
    if products_df is None or len(products_df) == 0:
        st.info("No products to display.")
        return
    for i in range(0, len(products_df), cols_per_row):
        row_products = products_df.iloc[i:i+cols_per_row]
        cols = st.columns(cols_per_row)
        for idx, (_, product) in enumerate(row_products.iterrows()):
            if idx < len(cols):
                with cols[idx]:
                    st.markdown(render_product_card(product, show_reason), unsafe_allow_html=True)

def render_auth_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; margin: 2rem 0;">
            <h1 style="font-size:3rem; font-weight:800; color:#667eea;">🛍️ ShopSmart AI</h1>
            <p style="font-size:1.1rem; color:#666;">Hybrid Recommendation Engine</p>
        </div>
        """, unsafe_allow_html=True)
        tab_login, tab_signup = st.tabs(["🔐 Login", "📝 Sign Up"])
        with tab_login:
            show_login_form()
        with tab_signup:
            show_signup_form()
        st.markdown("---")
        st.markdown("### ✨ Powered By")
        feat_cols = st.columns(3)
        features = [
            ("🧠", "Deep Learning", "Neural Collaborative Filtering with PyTorch"),
            ("📊", "Content Analysis", "TF-IDF similarity on product metadata"),
            ("🔗", "Association Mining", "Apriori & FP-Growth for frequent patterns")
        ]
        for col, (icon, title, desc) in zip(feat_cols, features):
            with col:
                st.markdown(f"""
                <div class="metric-container" style="border-left-color:#667eea;">
                    <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
                    <div style="font-weight:600; margin-bottom:0.25rem;">{title}</div>
                    <div style="font-size:0.8rem; color:#666;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

def render_home_page(products, customers, interactions, recommender):
    user = get_current_user()
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title">Welcome back, {user['username']}! 👋</div>
        <div class="hero-subtitle">Discover products curated just for you using AI-powered hybrid recommendations.</div>
    </div>
    """, unsafe_allow_html=True)
    customer_id = f"CUST_{hash(user['username']) % 1000 + 1:04d}"
    user_purchases = interactions[(interactions['customer_id'] == customer_id) & (interactions['interaction_type'] == 'purchase')]
    is_cold_start = len(user_purchases) < 3
    if is_cold_start:
        st.markdown("""
        <div class="cold-start-banner">
            <strong>👋 New here?</strong> We're showing you our most popular picks to get started!
            Browse and purchase to unlock personalized AI recommendations.
        </div>
        """, unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Recommended for You</div>', unsafe_allow_html=True)
    with st.spinner("Generating AI recommendations..."):
        if is_cold_start:
            from hybrid_service import get_cold_start_recommendations
            recs = get_cold_start_recommendations(products, n=12)
        else:
            recs = recommender.recommend(customer_id, n_recommendations=12)
    render_product_row(recs, cols_per_row=6, show_reason=not is_cold_start)
    if not is_cold_start:
        because_df = recommender.get_because_you_bought(customer_id, n=6)
        if len(because_df) > 0:
            st.markdown('<div class="section-title">🛒 Because You Bought</div>', unsafe_allow_html=True)
            render_product_row(because_df, cols_per_row=6)
    st.markdown('<div class="section-title">🔥 Trending Now</div>', unsafe_allow_html=True)
    trending = recommender.get_trending_products(n=12)
    render_product_row(trending, cols_per_row=6)
    st.markdown('<div class="section-title">📂 Browse by Category</div>', unsafe_allow_html=True)
    for category in products['category'].unique()[:3]:
        cat_products = products[products['category'] == category].nlargest(6, 'n_ratings')
        st.markdown(f"##### {category}")
        render_product_row(cat_products, cols_per_row=6)
        st.markdown("<br>", unsafe_allow_html=True)

def render_recommendations_page(products, interactions, recommender):
    st.markdown('<div class="section-title" style="font-size:2rem;">🔍 Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown("### ⚙️ Algorithm Weights")
    col1, col2, col3 = st.columns(3)
    with col1:
        w_dl = st.slider("Deep Learning (NCF)", 0.0, 1.0, WEIGHT_DL, 0.05)
    with col2:
        w_content = st.slider("Content-Based", 0.0, 1.0, WEIGHT_CONTENT, 0.05)
    with col3:
        w_assoc = st.slider("Association Rules", 0.0, 1.0, WEIGHT_ASSOCIATION, 0.05)
    if st.button("🔄 Update Recommendations", use_container_width=True):
        recommender.update_weights(w_dl, w_content, w_assoc)
        st.success("Weights updated!")
        st.rerun()
    st.markdown("---")
    user = get_current_user()
    customer_id = f"CUST_{hash(user['username']) % 1000 + 1:04d}"
    with st.spinner("Generating personalized recommendations..."):
        recs = recommender.recommend(customer_id, n_recommendations=15)
    if len(recs) > 0:
        st.markdown('<div class="section-title">Your Top Recommendations</div>', unsafe_allow_html=True)
        score_df = recs[['product_name', 'ncf_score', 'content_score', 'association_score', 'final_score']].head(10)
        score_df = score_df.melt(id_vars=['product_name'], var_name='Algorithm', value_name='Score')
        fig = px.bar(score_df, x='product_name', y='Score', color='Algorithm', barmode='group',
                     title='Score Breakdown by Algorithm',
                     color_discrete_map={'ncf_score': '#667eea', 'content_score': '#10b981',
                                         'association_score': '#f59e0b', 'final_score': '#1a1a2e'})
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        render_product_row(recs, cols_per_row=5, show_reason=True)
    st.markdown("---")
    st.markdown('<div class="section-title">🔎 Find Similar Products</div>', unsafe_allow_html=True)
    selected_product = st.selectbox("Select a product:", products['product_name'].tolist())
    product_id = products[products['product_name'] == selected_product]['product_id'].iloc[0]
    similar = recommender.get_similar_products(product_id, n=6)
    if len(similar) > 0:
        similar_full = similar.merge(products, on='product_id', how='left')
        render_product_row(similar_full, cols_per_row=6)
    st.markdown("---")
    st.markdown('<div class="section-title">🔗 Association Rules</div>', unsafe_allow_html=True)
    from association import load_rules, format_rules_for_display
    try:
        rules = load_rules().head(10)
        formatted = format_rules_for_display(rules)
        st.dataframe(formatted, use_container_width=True)
    except Exception as e:
        st.info(f"Association rules not available yet. {e}")

def render_dashboard_page(products, customers, interactions):
    st.markdown('<div class="section-title" style="font-size:2rem;">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="metric-container"><div class="metric-value">{customers["customer_id"].nunique():,}</div><div class="metric-label">Total Customers</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="metric-container" style="border-left-color:#10b981;"><div class="metric-value">{products["product_id"].nunique():,}</div><div class="metric-label">Products</div></div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown(f'<div class="metric-container" style="border-left-color:#f59e0b;"><div class="metric-value">{len(interactions):,}</div><div class="metric-label">Interactions</div></div>', unsafe_allow_html=True)
    with kpi4:
        avg_rating = interactions['rating'].mean()
        st.markdown(f'<div class="metric-container" style="border-left-color:#ef4444;"><div class="metric-value">{avg_rating:.1f}⭐</div><div class="metric-label">Avg Rating</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    col_left, col_right = st.columns(2)
    with col_left:
        cat_counts = products['category'].value_counts()
        fig = px.pie(values=cat_counts.values, names=cat_counts.index, title="Product Categories", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        fig = px.histogram(products, x='rating', nbins=10, title="Product Rating Distribution", color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    daily_interactions = interactions.groupby(interactions['timestamp'].dt.date).size().reset_index()
    daily_interactions.columns = ['date', 'count']
    fig = px.line(daily_interactions, x='date', y='count', title="Daily Interactions Over Time", color_discrete_sequence=['#667eea'])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.markdown("### 🏆 Top Products by Revenue")
    revenue = interactions.groupby('product_id')['purchase_amount'].sum().reset_index()
    revenue = revenue.merge(products[['product_id', 'product_name', 'category']], on='product_id')
    revenue = revenue.nlargest(15, 'purchase_amount')
    fig = px.bar(revenue, x='purchase_amount', y='product_name', color='category', orientation='h', title="Top 15 Products by Revenue", color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

def render_profile_page(products, interactions):
    st.markdown('<div class="section-title" style="font-size:2rem;">👤 My Profile</div>', unsafe_allow_html=True)
    user = get_current_user()
    customer_id = f"CUST_{hash(user['username']) % 1000 + 1:04d}"
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f'<div class="metric-container" style="text-align:center;"><div style="font-size:4rem; margin-bottom:1rem;">👤</div><div style="font-size:1.3rem; font-weight:700;">{user["username"]}</div><div style="color:#666; margin-top:0.5rem;">{user["email"]}</div><div style="color:#999; font-size:0.85rem; margin-top:0.25rem;">Customer ID: {customer_id}</div></div>', unsafe_allow_html=True)
    with col2:
        user_interactions = interactions[interactions['customer_id'] == customer_id]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Orders", len(user_interactions[user_interactions['interaction_type'] == 'purchase']))
        with c2:
            st.metric("Total Spent", f"₹{user_interactions['purchase_amount'].sum():,.0f}")
        with c3:
            st.metric("Avg Rating", f"{user_interactions['rating'].mean():.1f}⭐")
    st.markdown("---")
    st.markdown("### 📜 Purchase History")
    purchases = user_interactions[user_interactions['interaction_type'] == 'purchase'].sort_values('timestamp', ascending=False)
    if len(purchases) > 0:
        purchases_display = purchases.merge(products[['product_id', 'product_name', 'category']], on='product_id')
        purchases_display['timestamp'] = purchases_display['timestamp'].dt.strftime('%Y-%m-%d')
        st.dataframe(purchases_display[['timestamp', 'product_name', 'category', 'quantity', 'purchase_amount', 'rating']].rename(columns={'timestamp': 'Date', 'product_name': 'Product', 'category': 'Category', 'quantity': 'Qty', 'purchase_amount': 'Amount (₹)', 'rating': 'Rating'}), use_container_width=True)
    else:
        st.info("No purchase history yet. Start shopping to see your history here!")

def main():
    if is_authenticated():
        with st.sidebar:
            st.markdown("### 🛍️ ShopSmart AI")
            st.markdown("---")
            page = st.radio("Navigate", ["🏠 Home", "🔍 Recommendations", "📊 Dashboard", "👤 Profile"], index=0, label_visibility="collapsed")
            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True):
                logout_user()
                st.rerun()
    else:
        page = "auth"
    if page == "auth":
        render_auth_page()
        return
    try:
        products, customers, interactions = load_all_data()
        recommender = get_recommender(products, interactions)
    except Exception as e:
        st.error("⚠️ Data not found. Please run training first.")
        st.code("python train_models.py", language="bash")
        return
    if page == "🏠 Home":
        render_home_page(products, customers, interactions, recommender)
    elif page == "🔍 Recommendations":
        render_recommendations_page(products, interactions, recommender)
    elif page == "📊 Dashboard":
        render_dashboard_page(products, customers, interactions)
    elif page == "👤 Profile":
        render_profile_page(products, interactions)

if __name__ == "__main__":
    main()

