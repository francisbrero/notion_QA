import streamlit as st
from streamlit.logger import get_logger
import os

from utils import refresh_chat_widget, add_sidebar


LOGGER = get_logger(__name__)

from chat import run_agent


def run():
    st.set_page_config(
        page_title="Nygel, your AI-powered Support Agent",
        page_icon="👋",
    )

    add_sidebar(st)

    refresh_chat_widget()

    # initiate the agent
    run_agent()

if __name__ == "__main__":
    run()