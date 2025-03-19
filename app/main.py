import streamlit as st
# from app.views.pages import main_page
from app.utils.config import load_config

def main():
    config = load_config()
    st.set_page_config(**config['app']['page_config'])
    # main_page.render(config)

if __name__ == "__main__":
    main()