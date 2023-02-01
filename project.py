# Import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Streamlit input
st.header('A mental health survey')
st.subheader('Programming for Data Science - Final project ')
st.write('Short explenation of the project')


# EXPLORATION AND CLEANING OF THE DATASET 
st.title('Content of the dataset')

# Import the dataset
mental_health_df = pd.read_csv("survey.csv")

# Useful informations  
mental_health_df.info()
st.write('Informazioni')

# Understanding of the columns 
print(mental_health_df.head(5))
st.write('Columns')

# Dropping columns
mental_health_df.drop(columns=['Timestamp'], inplace = True)






