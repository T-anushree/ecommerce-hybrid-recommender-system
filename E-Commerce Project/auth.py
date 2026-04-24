"""
Authentication logic for the Streamlit application.
Handles session state management for login/signup flows.
"""

import streamlit as st
from db import register_user, authenticate_user, user_exists


# Session state keys
SESSION_USER = "authenticated_user"
SESSION_LOGIN_STATUS = "is_authenticated"


def init_auth_state():
    """Initialize authentication state in Streamlit session."""
    if SESSION_LOGIN_STATUS not in st.session_state:
        st.session_state[SESSION_LOGIN_STATUS] = False
    if SESSION_USER not in st.session_state:
        st.session_state[SESSION_USER] = None


def is_authenticated() -> bool:
    """Check if user is currently authenticated."""
    return st.session_state.get(SESSION_LOGIN_STATUS, False)


def get_current_user() -> dict:
    """Get the currently authenticated user's info."""
    return st.session_state.get(SESSION_USER, None)


def login_user(username: str, password: str) -> bool:
    """
    Attempt to log in a user.
    
    Args:
        username (str): Username.
        password (str): Password.
    
    Returns:
        bool: True if login successful, False otherwise.
    """
    result = authenticate_user(username, password)
    
    if result["status"] == "success":
        st.session_state[SESSION_LOGIN_STATUS] = True
        st.session_state[SESSION_USER] = result["user"]
        return True
    else:
        return False


def logout_user():
    """Log out the current user and clear session state."""
    st.session_state[SESSION_LOGIN_STATUS] = False
    st.session_state[SESSION_USER] = None
    # Clear any other app-specific state
    keys_to_clear = ["page", "selected_cluster", "recommendations"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def signup_user(username: str, email: str, password: str, full_name: str = "") -> dict:
    """
    Register a new user through the UI.
    
    Args:
        username (str): Desired username.
        email (str): Email address.
        password (str): Password.
        full_name (str): Full name (optional).
    
    Returns:
        dict: Registration result with status and message.
    """
    # Basic validation
    if len(username) < 3:
        return {"status": "error", "message": "Username must be at least 3 characters."}
    
    if len(password) < 6:
        return {"status": "error", "message": "Password must be at least 6 characters."}
    
    if "@" not in email or "." not in email:
        return {"status": "error", "message": "Please enter a valid email address."}
    
    result = register_user(username, email, password, full_name)
    return result


def require_auth():
    """
    Decorator-like function to enforce authentication.
    Call at the beginning of protected pages.
    """
    if not is_authenticated():
        st.warning("Please log in to access this page.")
        st.stop()


def show_login_form():
    """Render the login form in Streamlit."""
    st.markdown("### 🔐 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submit = st.form_submit_button("Login", use_container_width=True)
        
        if submit:
            if not username or not password:
                st.error("Please fill in all fields.")
                return False
            
            if login_user(username, password):
                st.success("Login successful! Redirecting...")
                st.rerun()
                return True
            else:
                st.error("Invalid username or password.")
                return False
    
    return False


def show_signup_form():
    """Render the signup form in Streamlit."""
    st.markdown("### 📝 Sign Up")
    
    with st.form("signup_form"):
        full_name = st.text_input("Full Name", placeholder="John Doe")
        username = st.text_input("Username", placeholder="Choose a username")
        email = st.text_input("Email", placeholder="john@example.com")
        password = st.text_input("Password", type="password", placeholder="Min 6 characters")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password")
        
        submit = st.form_submit_button("Create Account", use_container_width=True)
        
        if submit:
            if not all([username, email, password, confirm_password]):
                st.error("Please fill in all required fields.")
                return False
            
            if password != confirm_password:
                st.error("Passwords do not match.")
                return False
            
            result = signup_user(username, email, password, full_name)
            
            if result["status"] == "success":
                st.success("Account created successfully! Please log in.")
                return True
            else:
                st.error(result["message"])
                return False
    
    return False

