# E-Commerce Customer Behaviour Analysis & Recommendation System - Implementation Plan

## Information Gathered
- Workspace is empty (no existing files)
- Need to build a complete end-to-end system from scratch
- Technology Stack: Python, Streamlit, SQLite, scikit-learn, mlxtend, matplotlib/plotly
- Target: Production-ready, resume-grade project

## Project Structure
```
E-Commerce Project/
├── app.py                  # Main Streamlit application
├── auth.py                 # Authentication logic
├── db.py                   # SQLite database operations
├── model.py                # Clustering models (KMeans, DBSCAN, Hierarchical)
├── association.py          # Association rule mining (Apriori, FP-Growth)
├── utils.py                # Data generation & utility functions
├── config.py               # Configuration constants
├── requirements.txt        # Dependencies
├── data/
│   └── ecommerce_data.csv  # Sample dataset
├── models/
│   ├── kmeans_model.pkl    # Saved KMeans model
│   ├── dbscan_model.pkl    # Saved DBSCAN model
│   ├── hierarchical_model.pkl  # Saved Hierarchical model
│   └── scaler.pkl          # Saved StandardScaler
└── ecommerce.db            # SQLite database
```

## Implementation Steps

### Phase 1: Foundation
- [ ] Step 1: Create `requirements.txt` with all dependencies (streamlit, scikit-learn, mlxtend, plotly, pandas, numpy)
- [ ] Step 2: Create `config.py` with constants and paths
- [ ] Step 3: Create `utils.py` with dummy data generator (customers, transactions, products)
- [ ] Step 4: Create `db.py` with SQLite schema (users table with hashed passwords)

### Phase 2: Backend Logic
- [ ] Step 5: Create `auth.py` with secure login/signup using bcrypt hashing
- [ ] Step 6: Create `model.py` with:
  - Data preprocessing pipeline
  - KMeans clustering with elbow method visualization
  - DBSCAN clustering
  - Hierarchical clustering with dendrogram
  - Model persistence (save/load .pkl)
  - Cold-start handling (default cluster assignment)
- [ ] Step 7: Create `association.py` with:
  - Transaction encoding for association mining
  - Apriori algorithm implementation
  - FP-Growth algorithm implementation
  - Rule generation (support, confidence, lift)
  - "Recommended for You" logic

### Phase 3: Frontend (Streamlit)
- [ ] Step 8: Create `app.py` with:
  - Session state management
  - Login/Signup pages
  - Sidebar navigation
  - Dashboard page (cluster visualizations: pie charts, scatter plots, 3D plots)
  - Recommendation page (association rules table, product recommendations)
  - "Recommended for You" section
  - Clean UI with cards, metrics, and tables

### Phase 4: Integration & Testing
- [ ] Step 9: Integrate all modules in `app.py`
- [ ] Step 10: Generate sample dataset
- [ ] Step 11: Train and save all models
- [ ] Step 12: Test full application flow
- [ ] Step 13: Verify `streamlit run app.py` works without errors

## Key Technical Decisions
- **Database**: SQLite (as required), users table with bcrypt-hashed passwords
- **Clustering Features**: Recency, Frequency, Monetary (RFM) + product category preferences
- **Association Mining**: mlxtend's apriori and fpgrowth with transaction basket data
- **Visualization**: Plotly for interactive charts
- **Model Storage**: joblib for .pkl serialization
- **Cold Start**: Assign new users to largest cluster or popularity-based recommendations

## Follow-up Steps
- Run `pip install -r requirements.txt`
- Execute `python -c "from utils import generate_data; generate_data()"` to create dataset
- Execute `python -c "from model import train_and_save_models; train_and_save_models()"` to train models
- Run `streamlit run app.py` to launch the app

## Confirm This Plan?
Please confirm to proceed with implementation. I will build all files with production-ready code, comprehensive comments, and error handling.

