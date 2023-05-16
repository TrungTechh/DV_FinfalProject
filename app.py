import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("## What factors influence life expectancy the most?")

feature_index = ["HIV/AIDS", "Adult Mortality", "Income composition of resources", "Schooling", "thinness 5-9 years"]
feature_value = [0.379045, 0.280811, 0.209,0.030993, 0.016]


fig = plt.figure(figsize=(10, 4))
sns.barplot(x=feature_value, y=feature_index)
st.pyplot(fig)


st.markdown("If a country want to improve their life expectency, the most important thing they should are:")

with st.container():
    st.markdown("- Prevent HIV/AIDS ")
    st.markdown("- Care in take care of adults")
    st.markdown("- Optimal utilization of available resources")
    st.markdown("- Invest in education")
    st.markdown("- Focus on nutrition for children 5 - 9 years")



