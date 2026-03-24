import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv("../Data/immo_data_clean.csv")
    return df