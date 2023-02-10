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

# Converting the column's name to lowercase characters 
mental_health_df.columns = mental_health_df.columns.map(str.lower)

# Dropping columns
# timestamp and comments column
mental_health_df.isnull().sum()
mental_health_df.drop(columns=['timestamp','comments'], inplace = True)
mental_health_df['country'].value_counts()

# country column
eu_states = pd.read_csv('europe_states.csv')
l_eu = list()
count_eu = 0
count_nothing = 0
cont_usa = 0
cont_canada = 0
for i in range(len(mental_health_df)):
    if str(mental_health_df.iloc[i]['country']) in list(eu_states['name']):
        count_eu+=1
        l_eu.append(mental_health_df.iloc[i]['country'])
    elif str(mental_health_df.iloc[i]['country']) == 'United States':
        cont_usa+=1
    elif str(mental_health_df.iloc[i]['country']) == 'Canada':
        cont_canada +=1
    else:
        count_nothing+=1


print('People from Europe:',count_eu, '\nPeople from USA: ', cont_usa, '\nPeople from Canada: ',cont_canada, '\nPeople from other countries: ',count_nothing)

eu_serie = pd.Series(l_eu)
eu_serie.value_counts()

# state column
mental_health_df['state'].value_counts()

mental_health_df.drop(columns=['state'], inplace = True)
