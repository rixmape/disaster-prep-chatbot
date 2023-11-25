"""
This module contains the main Streamlit app.
"""

import streamlit as st

# Initialize a session state variable for token validation
if "token_valid" not in st.session_state:
    st.session_state.token_valid = False

# Show the token input only if the token has not been validated yet
if not st.session_state.token_valid:
    st.title("Token Validation")
    access_token = st.text_input(
        "Enter your access token to continue:",
        type="password",
        help="Get your access token from the app developer.",
    )

    # Check if the entered token is valid
    if access_token and access_token in st.secrets["access_tokens"]:
        st.success("You have successfully entered a valid token!")
        st.session_state.token_valid = True
        st.rerun()
    elif access_token:
        st.error("Please enter a valid token to access the main page.")
else:
    # Display the sidebar and main content
    st.sidebar.title("Sidebar")
    st.sidebar.write("This is the sidebar content.")

    st.title("Main Page")
    st.write("Welcome to the main page!")
