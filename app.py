import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.markdown("# Query")

feature_index = ["HIV/AIDS", "Adult Mortality", "Income composition of resources", "Schooling", "thinness 5-9 years"]
feature_value = [0.379045, 0.280811, 0.209,0.030993, 0.016]


fig = plt.figure(figsize=(10, 4))
sns.barplot(x=feature_value, y=feature_index)
st.pyplot(fig)
