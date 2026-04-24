# 🛒 E-Commerce Hybrid Recommendation System

An industry-level **Hybrid Recommendation System** that combines **Deep Learning (PyTorch), Content-Based Filtering, and Association Rule Mining** to deliver personalized product recommendations—similar to platforms like Amazon and Netflix.

---

## 🚀 Features

* 🔐 **Authentication System** (Login / Signup)
* 🤖 **Hybrid Recommendation Engine**

  * Neural Collaborative Filtering (PyTorch)
  * Content-Based Filtering (TF-IDF / Similarity)
  * Association Rules (Apriori / FP-Growth)
* 🧠 **Personalized Recommendations**

  * Recommended for You
  * Because You Bought X
  * Similar Products
  * Trending Items
* 📊 **Customer Segmentation**

  * K-Means Clustering
* 📈 **Interactive Dashboard**
* 🎨 **Modern UI**

  * Product cards (image, price, rating)
  * Sidebar navigation
  * Clean layout (Streamlit)

---

## 🧠 Architecture Overview

```
                +----------------------+
                |   User Interaction   |
                +----------+-----------+
                           |
                           v
        +---------------------------------------+
        |     Hybrid Recommendation Engine      |
        |---------------------------------------|
        | 1. Deep Learning (PyTorch - NCF)      |
        | 2. Content-Based Filtering            |
        | 3. Association Rules (Apriori)        |
        +----------------+----------------------+
                         |
                         v
               Final Weighted Recommendation
```

### 🔥 Hybrid Scoring Formula

Final recommendation score is computed as:

```
Final Score = 
    (0.5 × Deep Learning Score) +
    (0.3 × Content Score) +
    (0.2 × Association Score)
```

---

## 📂 Project Structure

```
ecommerce-recommender/
│
├── app/
│   ├── app.py                # Main Streamlit app
│   ├── auth.py               # Login & Signup
│   ├── db.py                 # Database (SQLite)
│   └── dashboard.py          # UI components
│
├── models/
│   ├── torch_model.py        # Deep Learning model
│   ├── content_based.py      # Content filtering
│   ├── association.py        # Apriori / FP-Growth
│   ├── clustering.py         # KMeans segmentation
│   └── hybrid.py             # Combines all models
│
├── utils/
│   └── preprocessing.py      # Data cleaning & prep
│
├── data/
│   └── dataset.csv
│
├── saved_models/
│   └── recommender.pt
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

* **Languages:** Python
* **Libraries:**

  * PyTorch
  * Scikit-learn
  * Pandas, NumPy
  * Mlxtend
* **Frontend:** Streamlit
* **Database:** SQLite

---

## 📊 Machine Learning Models

### 1️⃣ Deep Learning (Collaborative Filtering)

* Neural Collaborative Filtering (NCF)
* User & Item embeddings
* Binary classification (interaction prediction)

### 2️⃣ Content-Based Filtering

* TF-IDF Vectorization
* Cosine similarity between products

### 3️⃣ Association Rule Mining

* Apriori / FP-Growth
* Metrics:

  * Support
  * Confidence
  * Lift

---

## 📈 Evaluation Metrics

* Precision@K
* Recall@K
* Recommendation Accuracy

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/ecommerce-recommender.git
cd ecommerce-recommender
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
streamlit run app/app.py
```

---

## 🔐 Authentication

* User credentials stored in **SQLite database**
* Secure login/signup system integrated

---

## 🧪 Sample Use Cases

* Recommend products based on past purchases
* Identify similar products using metadata
* Discover frequently bought-together items
* Segment users for targeted marketing

---

## ❗ Challenges Solved

* Cold-start problem (new users/products)
* Data sparsity
* Combining multiple recommendation techniques
* Real-time inference optimization

---

## 🚀 Future Improvements

* 🔥 Real-time recommendation API (FastAPI)
* 🧠 Transformer-based recommendation models
* ☁️ Deployment on AWS / GCP
* 📱 React frontend (production UI)
* 🧾 Logging & monitoring system

## 💼 Resume Impact

**Built a production-ready Hybrid Recommendation System using Deep Learning (PyTorch), Content-Based Filtering, and Association Rule Mining, deployed with an interactive Streamlit UI delivering personalized product recommendations.**

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repo and submit a pull request.

---



## ⭐ Acknowledgements

* Inspired by recommendation systems used in Amazon & Netflix
* Open-source ML community

---
