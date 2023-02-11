# Import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st


@st.cache(allow_output_mutation = True)
def get_data(url):
    mental_health_df = pd.read_csv(url)
    return mental_health_df

@st.cache
def get_downloadable_data(df):
    return mental_health_df.to_csv().encode('utf-8')

### TITLE AND CONTEXT ###
st.title('A mental health survey')
st.write('Programming for Data Science - Final project ')

st.header('Context')
st.write('''
This dataset is from a 2014 survey that measures attitudes towards mental health and frequency of mental health disorders 
in the tech workplace. The organization that has collected this data is the "Open Sourcing Mental Health" organization (OSMH). 
Open Sourcing Mental Health is a non-profit, corporation dedicated to raising awareness, educating, and providing resources 
to support mental wellness in the tech and open source communities.')
''')

### DOWNLOAD RAW DATA ###
url = 'https://raw.githubusercontent.com/EmmaTosato/Programming_Project/main/survey.csv'
mental_health_df_static = get_data(url)
mental_health_df = mental_health_df_static.copy()

st.header('Download raw data')
st.download_button('Download', get_downloadable_data(mental_health_df_static), file_name = 'survey.csv')



### EXPLORATION AND CLEANING OF THE DATASET ###
st.header('Content of the dataset')

# USEFUL INFORMATIONS  
st.write('''
- In this dataset there are 1259 rows and 27 columnes (attributes)
- All the attribitues have object values, except for the age column that has integer values
''')

# MEANING OF THE COLUMNS 


# CONVERTING THE COLUMN'S NAME TO LOWERCASE CHARACTERS
mental_health_df.columns = mental_health_df.columns.map(str.lower)

# DROPPING COLUMNS

mental_health_df.isnull().sum()
# timestamp and comments column
mental_health_df.drop(columns=['timestamp','comments'], inplace = True)

# state column
mental_health_df['state'].value_counts()
mental_health_df.drop(columns=['state'], inplace = True)

# country column
mental_health_df['country'].value_counts()
 Grouping states
eu_states = pd.read_csv('eu_states.csv')

# Grouping states
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

# Visualizing european states
eu_serie = pd.Series(l_eu)
eu_serie.value_counts()


# CLEANING FROM NON SENSE AND NULL VALUES
# gender column
print('Unique values for the gender column:\n',mental_health_df['gender'].unique())

female_list = ['Female', 'female', 'Cis Female', 'F' , 'Woman', 'f', 'Femake' , 'woman', 'Female ', 'cis-female/femme', 'Female (cis)', 'femail' ]
male_list = ['M','Male','male', 'm' ,'Male-ish', 'maile', 'Cis Male' , 'Mal', 'Male (CIS)', 'Make', 'Male ', 'Man','msle','Mail' , 'cis male', 'Malr', 'Cis Man' ]
other_list = ['Trans-female', 'something kinda male?', 'queer/she/they', 'non-binary', 'Nah', 'All','Enby', 'fluid','Genderqueer', 'Androgyne','Agender', 'Guy (-ish) ^_^','male leaning androgynous', 'Trans woman', 'Neuter','Female (trans)','queer', 'A little about you', 'p', 'ostensibly male, unsure what that really means']

mental_health_df.replace(female_list, 'Female', inplace=True)
mental_health_df.replace(male_list, 'Male', inplace=True)
mental_health_df.replace(other_list, 'Other', inplace=True)

print('Unique values of the gender column after the cleaning \n', mental_health_df['gender'].unique())
print('\nLabel counts:')
print(mental_health_df['gender'].value_counts())

# age column
print('Unique values for the age column:\n', mental_health_df['age'].unique())
print('\n\nFirst 10 min and max values for the age column:')
print('Max values: ', np.sort(list(mental_health_df['age'].nlargest(10))))
print('\nMin values: ', np.sort(list(mental_health_df['age'].nsmallest(10))))

mental_health_df.drop(mental_health_df[mental_health_df['age'] < 18].index, inplace= True)
mental_health_df.drop(mental_health_df[mental_health_df['age'] > 72].index, inplace= True)

print('Unique values of the age column after the cleaning \n', np.sort(mental_health_df['age'].unique()))
print('\nMin and max age:',mental_health_df['age'].min(),'-',mental_health_df['age'].max() )
print('\nAverage age: ', int(mental_health_df['age'].mean()))

# self_employed and work_interfere
print('NaN values for the column self_employed: ', mental_health_df['self_employed'].isnull().sum() )
print('NaN values for the column work_interfere: ', mental_health_df['work_interfere'].isnull().sum() )

print('\nCounting values for the column self_employed')
print(mental_health_df['self_employed'].value_counts())

print('\nCounting values for the column self_employed')
print(mental_health_df['work_interfere'].value_counts())

# self_employed column
mental_health_df['self_employed'].fillna(value = 'No', inplace = True)

# work_interfere column
mental_health_df['work_interfere'].fillna(value = 'Don\'t know', inplace = True)

# final control
for column in mental_health_df.iloc[:, 3:]:
    print(column, mental_health_df[column].unique())


### INTERESTING PLOTS ###
