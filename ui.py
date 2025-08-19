import streamlit as st
import logging

def display_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")

def get_user_input():
    return st.text_input("Your message", key="chat_input", placeholder="Type your question here...")

def process_user_input(user_input, agent):
    logging.basicConfig(level=logging.INFO)
    if "last_user_input" not in st.session_state:
        st.session_state.last_user_input = ""
    if user_input and user_input != st.session_state.last_user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        try:
            response = agent({"input": user_input})
            answer = response.get("output", "No output from agent.")
        except Exception as e:
            answer = f"Sorry, there was an error processing your request: {e}"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_user_input = user_input
        st.rerun()

def handle_chat_interaction(agent):
    display_chat_history()
    user_input = get_user_input()
    process_user_input(user_input, agent)
