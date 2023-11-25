"""
This module contains the main Streamlit app for the chatbot.
"""

import streamlit as st

# Initialize session state variables
st.session_state.setdefault("token_valid", False)
st.session_state.setdefault("files", [])

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
    sidebar_message = st.sidebar.empty()

    files = st.sidebar.file_uploader(
        "Upload some files:",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "html", "md", "pptx"],
    )

    if files != st.session_state.files:
        sidebar_message.warning("Unsaved files!")

    *_, col = st.sidebar.columns(4)  # Right align the button
    if col.button("Save"):
        if files != st.session_state.files:
            st.session_state.files = files
            sidebar_message.success("Files saved!")
        else:
            sidebar_message.warning("No changes made.")

    st.title("Let's Chat!")
    st.expander("Session State", expanded=False).json(st.session_state)
