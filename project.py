# Import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib import colors as mcolors
import matplotlib.ticker as mtik
import seaborn as sb
import streamlit as st
from sklearn import preprocessing
import plotly.express as px



@st.cache(allow_output_mutation = True)
def get_data(url):
    mental_health_df_raw = pd.read_csv(url)
    return mental_health_df_raw

@st.cache
def get_downloadable_data(df):
    return mental_health_df_raw.to_csv().encode('utf-8')

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
mental_health_df_raw = mental_health_df_static.copy()

st.header('Download raw data')
st.download_button('Download', get_downloadable_data(mental_health_df_static), file_name = 'survey.csv')



### EXPLORATION AND CLEANING OF THE DATASET ###
st.header('Content of the dataset')

# USEFUL INFORMATIONS  
st.write('''
- In this dataset there are 1259 rows and 27 columnes (attributes)
- All the attribitues have object values, except for the age column that has integer values
''')

st.dataframe(mental_health_df_raw)

# MEANING OF THE COLUMNS 
st.write('''
The first 4 columns (2 to 5) concern general informations about the individuals. Every attribute contains answers to a specific question.

* **Timestamp:** contains date, month, year and time

* **Age**

* **Gender**

* **Country**

* **state:** If you live in the United States, which state or territory do you live in?

* **self_employed:** Are you self-employed?

* **family_history:** Do you have a family history of mental illness?

* **treatment:** Have you sought treatment for a mental health condition?

* **work_interfere:** If you have a mental health condition, do you feel that it interferes with your work?

* **no_employees:** How many employees does your company or organization have?

* **remote_work:** Do you work remotely (outside of an office) at least 50% of the time?

* **tech_company:** Is your employer primarily a tech company/organization?

* **benefits:** Does your employer provide mental health benefits?

* **care_options:** Do you know the options for mental health care your employer provides?

* **wellness_program:** Has your employer ever discussed mental health as part of an employee wellness program?

* **seek_help:** Does your employer provide resources to learn more about mental health issues and how to seek help?

* **anonymity:** Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?

* **leave:** How easy is it for you to take medical leave for a mental health condition?

* **mentalhealthconsequence:** Do you think that discussing a mental health issue with your employer would have negative consequences?

* **physhealthconsequence:** Do you think that discussing a physical health issue with your employer would have negative consequences?

* **coworkers:** Would you be willing to discuss a mental health issue with your coworkers?

* **physhealthinterview:** Would you bring up a physical health issue with a potential employer in an interview?

* **mentalvsphysical:** Do you feel that your employer takes mental health as seriously as physical health?

* **obs_consequence:** Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?

* **comments:** Any additional notes or comments
''')
mental_health_df_raw.head()


# CONVERTING THE COLUMN'S NAME TO LOWERCASE CHARACTERS
mental_health_df_raw.columns = mental_health_df_raw.columns.map(str.lower)

# DROPPING COLUMNS

mental_health_df_raw.isnull().sum()
# timestamp and comments column
mental_health_df_raw.drop(columns=['timestamp','comments'], inplace = True)

# state column
mental_health_df_raw['state'].value_counts()
mental_health_df_raw.drop(columns=['state'], inplace = True)

# country column
mental_health_df_raw['country'].value_counts()

#Grouping states
def group_states(mental_health_df_raw):
    # Grouping states
    eu_states = pd.read_csv('eu_states.csv')

    l_eu = list()
    count_eu = 0
    count_nothing = 0
    cont_usa = 0
    cont_canada = 0
    for i in range(len(mental_health_df_raw)):
        if str(mental_health_df_raw.iloc[i]['country']) in list(eu_states['name']):
            count_eu+=1
            l_eu.append(mental_health_df_raw.iloc[i]['country'])
        elif str(mental_health_df_raw.iloc[i]['country']) == 'United States':
            cont_usa+=1
        elif str(mental_health_df_raw.iloc[i]['country']) == 'Canada':
            cont_canada +=1
        else:
            count_nothing+=1
    return count_eu, cont_usa, cont_canada, count_nothing, l_eu


count_eu, cont_usa, cont_canada, count_nothing, l_eu = group_states(mental_health_df_raw)
print('People from Europe:',count_eu, '\nPeople from USA: ', cont_usa, '\nPeople from Canada: ',cont_canada, '\nPeople from other countries: ',count_nothing)

# Visualizing european states
eu_serie = pd.Series(l_eu)
eu_serie.value_counts()


# CLEANING FROM NON SENSE AND NULL VALUES
# gender column
print('Unique values for the gender column:\n',mental_health_df_raw['gender'].unique())

female_list = ['Female', 'female', 'Cis Female', 'F' , 'Woman', 'f', 'Femake' , 'woman', 'Female ', 'cis-female/femme', 'Female (cis)', 'femail' ]
male_list = ['M','Male','male', 'm' ,'Male-ish', 'maile', 'Cis Male' , 'Mal', 'Male (CIS)', 'Make', 'Male ', 'Man','msle','Mail' , 'cis male', 'Malr', 'Cis Man' ]
other_list = ['Trans-female', 'something kinda male?', 'queer/she/they', 'non-binary', 'Nah', 'All','Enby', 'fluid','Genderqueer', 'Androgyne','Agender', 'Guy (-ish) ^_^','male leaning androgynous', 'Trans woman', 'Neuter','Female (trans)','queer', 'A little about you', 'p', 'ostensibly male, unsure what that really means']

mental_health_df_raw.replace(female_list, 'Female', inplace=True)
mental_health_df_raw.replace(male_list, 'Male', inplace=True)
mental_health_df_raw.replace(other_list, 'Other', inplace=True)

print('Unique values of the gender column after the cleaning \n', mental_health_df_raw['gender'].unique())
print('\nLabel counts:')
print(mental_health_df_raw['gender'].value_counts())

# age column
print('Unique values for the age column:\n', mental_health_df_raw['age'].unique())
print('\n\nFirst 10 min and max values for the age column:')
print('Max values: ', np.sort(list(mental_health_df_raw['age'].nlargest(10))))
print('\nMin values: ', np.sort(list(mental_health_df_raw['age'].nsmallest(10))))

mental_health_df_raw.drop(mental_health_df_raw[mental_health_df_raw['age'] < 18].index, inplace= True)
mental_health_df_raw.drop(mental_health_df_raw[mental_health_df_raw['age'] > 72].index, inplace= True)

print('Unique values of the age column after the cleaning \n', np.sort(mental_health_df_raw['age'].unique()))
print('\nMin and max age:',mental_health_df_raw['age'].min(),'-',mental_health_df_raw['age'].max() )
print('\nAverage age: ', int(mental_health_df_raw['age'].mean()))

# self_employed and work_interfere
print('NaN values for the column self_employed: ', mental_health_df_raw['self_employed'].isnull().sum() )
print('NaN values for the column work_interfere: ', mental_health_df_raw['work_interfere'].isnull().sum() )

print('\nCounting values for the column self_employed')
print(mental_health_df_raw['self_employed'].value_counts())

print('\nCounting values for the column self_employed')
print(mental_health_df_raw['work_interfere'].value_counts())

# self_employed column
mental_health_df_raw['self_employed'].fillna(value = 'No', inplace = True)

# work_interfere column
mental_health_df_raw['work_interfere'].fillna(value = 'Don\'t know', inplace = True)

# final control
for column in mental_health_df_raw.iloc[:, 3:]:
    print(column, mental_health_df_raw[column].unique())

# Reindexing
mental_health_df_raw.reset_index(drop=True, inplace=True)

# Clean dataset
mental_health_df = mental_health_df_raw.copy()


### DOWNLOAD CLEAN DATASET ###
mental_health_df_static_clean = mental_health_df.copy()

st.header('Download clean dataset')
st.download_button('Download', get_downloadable_data(mental_health_df_static_clean), file_name = 'survey_clean.csv')

st.write('''
    Some highlights:
    - timestamp, comments and state columns were dropped
    - gender and age columns were clean from null and non sense values
    - self_employed and work_interfere columns have their null values substituted 
'''
)

### INTERESTING PLOTS ###

# Gender + Treatment
