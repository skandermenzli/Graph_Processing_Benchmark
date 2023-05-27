import streamlit as st
import pandas as pd


st.title('Datasets')

st.markdown("<span style='font-size:20px;'>The list of graphs used to build our model</span>", unsafe_allow_html=True)

df = pd.read_csv('csv/graphs.csv')
st.dataframe(df)



