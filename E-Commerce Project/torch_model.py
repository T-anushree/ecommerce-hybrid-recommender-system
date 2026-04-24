"""
Neural Collaborative Filtering (NCF) Model using PyTorch.
Implements a deep learning recommendation model with user and item embeddings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder

from config import (
    EMBEDDING_DIM, HIDDEN_DIMS, DROPOUT, 
    LEARNING_RATE, BATCH_SIZE, EPOCHS,
    NCF_MODEL_PATH, USER_ENCODER_PATH, PRODUCT_ENCODER_PATH
)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[TORCH] Using device: {DEVICE}")


class RecommendationDataset(Dataset):
    """PyTorch Dataset for user-item interactions."""
    
    def __init__(self, users, items, ratings):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class NCFModel(nn.Module):
    """
    Neural Collaborative Filtering Model.
    
    Architecture:
    - User embedding layer
    - Item embedding layer
    - Concatenated embeddings fed through MLP
    - Output: predicted interaction score (0-1)
    """
    
    def __init__(self, n_users, n_items, embedding_dim=EMBEDDING_DIM, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT):
        super(NCFModel, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Initialize embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # MLP layers
        self.fc_layers = nn.ModuleList()
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout))
            self.fc_layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, user_indices, item_indices):
        """Forward pass."""
        # Get embeddings
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        
        # Concatenate
        vector = torch.cat([user_embed, item_embed], dim=-1)
        
        # Pass through MLP
        for layer in self.fc_layers:
            vector = layer(vector)
        
        # Output
        output = self.output_layer(vector)
        output = self.sigmoid(output)
        
        return output.squeeze()
    
    def predict(self, user_indices, item_indices):
        """Predict scores for user-item pairs."""
        self.eval()
        with torch.no_grad():
            user_indices = torch.LongTensor(user_indices).to(DEVICE)
            item_indices = torch.LongTensor(item_indices).to(DEVICE)
            scores = self.forward(user_indices, item_indices)
        return scores.cpu().numpy()
    
    def recommend(self, user_id, n_items, all_item_indices):
        """
        Generate top-N recommendations for a user.
        
        Args:
            user_id: Encoded user ID
            n_items: Number of recommendations
            all_item_indices: All available item indices
        
        Returns:
            Top item indices and their scores
        """
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id] * len(all_item_indices)).to(DEVICE)
            item_tensor = torch.LongTensor(all_item_indices).to(DEVICE)
            scores = self.forward(user_tensor, item_tensor)
            
            # Get top N
            top_scores, top_indices = torch.topk(scores, min(n_items, len(scores)))
            
        return top_indices.cpu().numpy(), top_scores.cpu().numpy()


def prepare_data(interactions_df):
    """
    Prepare interaction data for NCF training.
    
    Args:
        interactions_df: DataFrame with customer_id, product_id, rating
    
    Returns:
        Processed DataFrame with encoded IDs, label encoders
    """
    print("[TORCH] Preparing data for NCF...")
    
    df = interactions_df.copy()
    
    # Remove missing ratings
    df = df.dropna(subset=["rating"])
    
    # Encode user IDs
    user_encoder = LabelEncoder()
    df["user_idx"] = user_encoder.fit_transform(df["customer_id"])
    
    # Encode item IDs
    item_encoder = LabelEncoder()
    df["item_idx"] = item_encoder.fit_transform(df["product_id"])
    
    # Normalize ratings to 0-1 scale
    df["rating_norm"] = df["rating"] / 5.0
    
    print(f"[TORCH] Users: {len(user_encoder.classes_)}, Items: {len(item_encoder.classes_)}")
    
    return df, user_encoder, item_encoder


def train_ncf(interactions_df, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """
    Train the NCF model.
    
    Args:
        interactions_df: Interaction DataFrame
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    
    Returns:
        Trained model, user encoder, item encoder
    """
    print("=" * 60)
    print("TRAINING NCF MODEL")
    print("=" * 60)
    
    # Prepare data
    df, user_encoder, item_encoder = prepare_data(interactions_df)
    
    n_users = len(user_encoder.classes_)
    n_items = len(item_encoder.classes_)
    
    # Create dataset and dataloader
    dataset = RecommendationDataset(
        df["user_idx"].values,
        df["item_idx"].values,
        df["rating_norm"].values
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = NCFModel(n_users, n_items).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_users, batch_items, batch_ratings in dataloader:
            batch_users = batch_users.to(DEVICE)
            batch_items = batch_items.to(DEVICE)
            batch_ratings = batch_ratings.to(DEVICE)
            
            # Forward
            predictions = model(batch_users, batch_items)
            loss = criterion(predictions, batch_ratings)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"[TORCH] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save model and encoders
    torch.save(model.state_dict(), NCF_MODEL_PATH)
    
    with open(USER_ENCODER_PATH, "wb") as f:
        pickle.dump(user_encoder, f)
    
    with open(PRODUCT_ENCODER_PATH, "wb") as f:
        pickle.dump(item_encoder, f)
    
    print(f"[TORCH] Model saved to {NCF_MODEL_PATH}")
    print("=" * 60)
    
    return model, user_encoder, item_encoder


def load_ncf_model(n_users, n_items):
    """Load trained NCF model."""
    model = NCFModel(n_users, n_items).to(DEVICE)
    model.load_state_dict(torch.load(NCF_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def load_encoders():
    """Load user and item encoders."""
    with open(USER_ENCODER_PATH, "rb") as f:
        user_encoder = pickle.load(f)
    with open(PRODUCT_ENCODER_PATH, "rb") as f:
        item_encoder = pickle.load(f)
    return user_encoder, item_encoder


if __name__ == "__main__":
    # Test
    from data_generator import generate_all_datasets
    _, _, interactions = generate_all_datasets()
    train_ncf(interactions)

