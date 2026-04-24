"""
SQLite database operations for user authentication and data storage.
Uses bcrypt for secure password hashing.
"""

import sqlite3
import bcrypt
import os
from datetime import datetime

from config import DB_PATH


def get_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable dict-like access
    return conn


def init_db():
    """
    Initialize the SQLite database.
    Creates the users table if it doesn't exist.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    """)
    
    # Create index for faster username lookup
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_username ON users(username)
    """)
    
    conn.commit()
    conn.close()
    print("[DB] Database initialized successfully.")


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password (str): Plain text password.
    
    Returns:
        str: Bcrypt hashed password.
    """
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        password (str): Plain text password.
        password_hash (str): Bcrypt hashed password.
    
    Returns:
        bool: True if password matches, False otherwise.
    """
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def register_user(username: str, email: str, password: str, full_name: str = "") -> dict:
    """
    Register a new user in the database.
    
    Args:
        username (str): Unique username.
        email (str): Unique email address.
        password (str): Plain text password (will be hashed).
        full_name (str): Optional full name.
    
    Returns:
        dict: Result with status and message.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Check if username already exists
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            return {"status": "error", "message": "Username already exists."}
        
        # Check if email already exists
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            return {"status": "error", "message": "Email already registered."}
        
        # Hash password and insert user
        password_hash = hash_password(password)
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name)
            VALUES (?, ?, ?, ?)
        """, (username, email, password_hash, full_name))
        
        conn.commit()
        user_id = cursor.lastrowid
        
        return {
            "status": "success",
            "message": "User registered successfully.",
            "user_id": user_id
        }
    
    except sqlite3.Error as e:
        return {"status": "error", "message": f"Database error: {str(e)}"}
    
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> dict:
    """
    Authenticate a user with username and password.
    
    Args:
        username (str): Username.
        password (str): Plain text password.
    
    Returns:
        dict: Result with status, message, and user info if successful.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT id, username, email, password_hash, full_name
            FROM users WHERE username = ?
        """, (username,))
        
        user = cursor.fetchone()
        
        if not user:
            return {"status": "error", "message": "Invalid username or password."}
        
        if not verify_password(password, user["password_hash"]):
            return {"status": "error", "message": "Invalid username or password."}
        
        # Update last login timestamp
        cursor.execute("""
            UPDATE users SET last_login = ? WHERE id = ?
        """, (datetime.now(), user["id"]))
        conn.commit()
        
        return {
            "status": "success",
            "message": "Login successful.",
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "full_name": user["full_name"]
            }
        }
    
    except sqlite3.Error as e:
        return {"status": "error", "message": f"Database error: {str(e)}"}
    
    finally:
        conn.close()


def get_user_by_username(username: str) -> dict:
    """
    Retrieve user details by username.
    
    Args:
        username (str): Username to look up.
    
    Returns:
        dict: User info or None if not found.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, username, email, full_name, created_at, last_login
        FROM users WHERE username = ?
    """, (username,))
    
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return dict(user)
    return None


def user_exists(username: str) -> bool:
    """Check if a username already exists."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


# Initialize database on module import
init_db()

