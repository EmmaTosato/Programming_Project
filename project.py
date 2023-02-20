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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


### FUNCTIONS ###A
@st.cache(allow_output_mutation = True)
def get_data(url):
    mental_health_df_raw = pd.read_csv(url)
    return mental_health_df_raw

@st.cache
def get_downloadable_data(df):
    return mental_health_df_raw.to_csv().encode('utf-8')

# New column for the countries
def new_column(df):
    eu_states = pd.read_csv('eu_states.csv')
    eu_l = list()
    temp_l = list()

    for i in range(len(df)):
        if str(df.iloc[i]['country']) in list(eu_states['name']):
            eu_l.append(df.iloc[i]['country'])
            temp_l.append('Europe')
        elif str(df.iloc[i]['country']) == 'United States':
            temp_l.append('United States')
        elif str(df.iloc[i]['country']) == 'Canada':
            temp_l.append('Canada')
        else:
            temp_l.append('Other countries')

    country2 = pd.Series(temp_l)

    return eu_l, country2

# Computing percenteges (without *100)
def percent(df, row): 
	perc = list() 
	for column in df: 
		values = df[column].loc[row] 
		total = df[column].sum() 
		perc.append((round((values/ total),2))) 
	return perc

# Label Encoding the categorical variables
def encoding(mental_health_df):
    mh_df_econded = mental_health_df.copy()
    mh_df_econded.drop(columns= ['country'], inplace= True)

    object_cols = ['gender', 'self_employed', 'family_history','treatment', 'work_interfere','no_employees','remote_work','tech_company',
    'benefits','care_options', 'wellness_program','seek_help','anonymity','leave','mental_health_consequence','phys_health_consequence',
    'coworkers','supervisor', 'mental_health_interview','phys_health_interview','mental_vs_physical','obs_consequence']

    label_encoder = LabelEncoder()
    for col in object_cols:
        label_encoder.fit(mental_health_df[col])
        mh_df_econded[col] = label_encoder.transform(mental_health_df[col])
    
    return  mh_df_econded



### TITLE AND CONTEXT ###
st.title('A mental health survey')
st.write('Programming for Data Science : final project ')
st.text("")

st.header('Dataset Information')
st.write('''
This dataset is from a 2014 survey that measures attitudes towards mental health and frequency of mental health disorders 
in the tech workplace. The organization that has collected this data is the "Open Sourcing Mental Health" organization (OSMH). 
Open Sourcing Mental Health is a non-profit, corporation dedicated to raising awareness, educating, and providing resources 
to support mental wellness in the tech and open source communities.')
''')
st.text("")


### EXPLORATION AND CLEANING OF THE DATASET ###
st.header('Content')

# THE DATASET
url = 'https://raw.githubusercontent.com/EmmaTosato/Programming_Project/main/survey.csv'
mental_health_df_static_raw = get_data(url)
mental_health_df_raw = mental_health_df_static_raw.copy()

# USEFUL INFORMATIONS AND MEANING OF THE COLUMNS 
mental_health_df_raw.head()
mental_health_df_raw.info()
st.text("")

# Streamlit
st.write('''Each column (or attribute) in the dataset contains the responses of each respondent to a specific question. 
            The first questions are both general in nature, while the following questions address the main theme of this survey.''')

with st.expander("Expand for the specific content of each column"):
    st.write('''
        * **Timestamp:** contains date, month, year and time

        * **Age**: How old are you?

        * **Gender**: Which gender do you identify with?

        * **Country**: What country do you live in?

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
st.text("\n")
st.subheader("Some highlights")
st.write('''
- In this dataset there are 1259 rows and 27 columns (attributes).
- All the attribitue have object values, except for the age column that has integer values.\n\n
''')

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
eu_l, country2 = new_column(mental_health_df_raw)
print('Number of people from different places:')
print(country2.value_counts())

# Visualizing european states
eu_serie = pd.Series(eu_l)
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


# CLEAN DATASET
mental_health_df = mental_health_df_raw.copy()
mental_health_df_static_clean = mental_health_df.copy()
mental_health_df.info()

# Streamlit
st.write('''
    After the cleaning of the dataset:
    - there are 1251 entries and 24 columns. 
    - *timestamp*, *comments* and *state* columns were dropped.
    - gender and age columns were clean from null and non sense values.
    - *self_employed* and *work_interfere columns* have their null values substituted. 
'''
)
st.text("\n")

# Exploration of the dataset
st.subheader('Exploration of the dataset')

st.write("Select an option (or both) if you want to explore the dataset:")
raw_option = st.checkbox("Raw dataset")
clean_option = st.checkbox("Clean dataset")

if raw_option:
    st.write('RAW dataset')
    st.dataframe(mental_health_df_static_raw)
st.text("")
if clean_option:
    st.write('CLEAN dataset')
    st.dataframe(mental_health_df_static_clean)

st.text("")

# DOWNLOAD Tabs 
st.header("Download data")
st.download_button('Download raw dataset', get_downloadable_data(mental_health_df_static_raw), file_name = 'survey.csv')
st.download_button('Download the clean dataset', get_downloadable_data(mental_health_df_static_clean), file_name = 'survey_clean.csv')
st.write('Dataset source: [click link]('+ url +')')
st.text("")
st.text("")


### INTERESTING PLOTS ##
st.header("Interesting plots")

# USEFUL CODE
# Useful data
tot_rows = mental_health_df.shape[0]
colors = ['steelblue', 'lightblue']

# Default settings
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams['axes.linewidth'] = 0.5 
plt.rcParams['axes.edgecolor'] = 'black'



# AGE, GENDER AND COUNTRY DISTRIBUTIONS  (different from the jupyter version)
# Data
counts_age = mental_health_df['age'].value_counts().values
labels_ages = mental_health_df['age'].value_counts().index
age_df = pd.DataFrame({'Age':labels_ages, 'Count': counts_age})
age_df.sort_values(by=['Age', 'Count'], inplace= True)

eu_l, country2 = new_column(mental_health_df)
countries = country2.value_counts().index.to_list()
cc = list(country2.value_counts().values)
countries_df = pd.DataFrame({'Country':countries, 'Count': cc})

counts_gender = mental_health_df['gender'].value_counts().to_list()
labels_gender = mental_health_df['gender'].value_counts().index.to_list()
gender_df = pd.DataFrame({'Gender':labels_gender, 'Count': counts_gender})

colors = [ "#8bd3c7",  "lightblue", "steelblue", "#df979e", ]

# Plots
fig1 = px.histogram( age_df, x=labels_ages, y= counts_age, nbins=70, 
                    labels= {'x': 'Ages', 'y': 'Counts'},
                    title = 'Age distribution')
fig2 = px.pie(countries_df, values='Count', names='Country', color_discrete_map = colors, title = 'Countries distribution')
fig3 = px.pie(gender_df, values='Count', names='Gender', color_discrete_map = colors, title = 'Gender distribution')

# Show
st.subheader("Some interactive graphs")
tab1, tab2, tab3 = st.tabs(["Age", "Countries", "Gender"])

with tab1:
   st.plotly_chart(fig1)

with tab2:
   st.plotly_chart(fig2)

with tab3:
   st.plotly_chart(fig3)

st.text("\n\n")


# TREATMENT AND WORK INTERFERENCE DISTRIBUTIONS
# 1
labels_treat = mental_health_df['treatment'].value_counts().index.to_list()               # Yes, No
counts_treat = list(map(lambda x: round((x/tot_rows),2), mental_health_df['treatment'].value_counts()))

# 2
labels_work = mental_health_df['work_interfere'].value_counts().index.to_list()          # 'Sometimes', "Don't know", 'Never', 'Rarely', 'Often'
counts_work = list(map(lambda x: round((x/tot_rows),2), mental_health_df['work_interfere'].value_counts())) 

# 3
df_treat_work = mental_health_df.groupby(['work_interfere', 'treatment'])['treatment'].count().unstack(0)
df_treat_work = df_treat_work.reindex(index = labels_treat, columns = labels_work)
yes_ans = list(df_treat_work[:].loc['Yes'].values)
no_ans = list(df_treat_work[:].loc['No'].values)


# The label locations
x = np.arange(len(labels_work))  
width = 0.4  # the width of the bars

# Plot
fig1, (ax1, ax2, ax3) = plt.subplots(1,3,figsize =(15, 6))
ax1.bar(labels_treat, counts_treat, color ='steelblue', label=labels_treat)
ax2.bar(labels_work, counts_work, color ='steelblue', label=labels_work)
ax3.bar(x - width/2, yes_ans, width, color= 'steelblue', label= 'Yes')
ax3.bar(x + width/2, no_ans, width, color= 'lightblue', label ='No')

# Labels, ticks 
ax1.set_xlabel("Treatment search", labelpad= 10.0, fontname="Arial", fontsize=11, fontweight = 'medium')
ax2.set_xlabel("Work interference", labelpad= 10.0, fontname="Arial", fontsize=11, fontweight = 'medium')
ax3.set_xlabel("Work interference", labelpad= 10.0, fontname="Arial", fontsize=11, fontweight = 'medium')
ax1.set_ylabel("Percentages", labelpad= 10.0, fontname="Arial", fontsize=11, fontweight = 'medium')
ax2.set_ylabel("Percentages", labelpad= 10.0, fontname="Arial", fontsize=11, fontweight = 'medium')
ax3.set_ylabel("Counts", labelpad= 10.0, fontname="Arial", fontsize=11, fontweight = 'medium')
ax1.set_yticks(np.arange(0.0,1.1, 0.1))
ax2.set_yticks(np.arange(0.0,1.1, 0.1))
ax2.set_xticks(list(range(0,5)), labels_work, rotation=30, ha='right')
ax3.set_xticks(list(range(0,5)), labels_work, rotation=30, ha='right')

# Percenteges
for p in ax1.patches:
    height = p.get_height()
    ax1.annotate("{}%".format(round(height*100)), (p.get_x() + p.get_width() / 2, height+ 0.01), ha='center')

for p in ax2.patches:
    height = p.get_height()
    ax2.annotate("{}%".format(round(height*100)), (p.get_x() + p.get_width() / 2, height+ 0.01), ha='center')

# Title
ax1.set_title('Employees who have \nsought treatment', fontsize= 14, fontweight= 'heavy', color = 'black', y=1.15, pad=10)
ax2.set_title('Work interference of the\n mental health condition', fontsize= 14, fontweight= 'heavy', color = 'black', y=1.15, pad=10)
ax3.set_title('Seeking treatment \nand work interference', fontsize= 14, fontweight= 'heavy', color = 'black', y=1.15, pad=10)

# Legend
ax3.legend(title = "Treatment search",loc = 'upper right', bbox_to_anchor=(0.65, 0.55, 0.5, 0.5))

# show
plt.tight_layout()
st.subheader("Treatment and work interference")
st.pyplot(fig1)
st.text("\n\n")


# REMOTE WORKING AND WORK INTERFERENCE DISTRIBUTIONS
# 1
labels_rw = mental_health_df['remote_work'].value_counts().index.to_list()[::-1]               # Yes, No
counts_rw = list(map(lambda x: round((x/tot_rows),2), mental_health_df['remote_work'].value_counts().reindex(index= labels_rw)))

# 2
labels_work = mental_health_df['work_interfere'].value_counts().index.to_list()          # 'Sometimes', "Don't know", 'Never', 'Rarely', 'Often'
df_rw_interf = mental_health_df.groupby(['work_interfere', 'remote_work'])['remote_work'].count().unstack(0)
df_rw_interf = df_rw_interf.reindex(index = ['Yes', 'No'], columns=labels_work)
yes_ans = list(df_rw_interf[:].loc['Yes'].values)
no_ans = list(df_rw_interf[:].loc['No'].values)


# The label locations
x = np.arange(len(labels_work))  
width = 0.4  # the width of the bars

# Plots
fig1, (ax1, ax2) = plt.subplots(1,2,figsize =(13, 5))
fig1.subplots_adjust(wspace=0.3)

wedges, text, autotext = ax1.pie(counts_rw, labeldistance=1.15, wedgeprops = { 'linewidth' : 1, 'edgecolor' : 'white' }, 
        colors = colors , autopct='%1.0f%%', textprops={'fontsize': 12})
ax2.bar(x - width/2, yes_ans, width, color= 'steelblue', label= 'Yes')
ax2.bar(x + width/2, no_ans, width, color= 'lightblue', label ='No')

# Labels, ticks 
ax2.set_xlabel("Work interference", labelpad= 10.0, fontname="Arial", fontsize=11, fontweight = 'medium')
ax2.set_ylabel("Remote work", labelpad= 10.0, fontname="Arial", fontsize=11, fontweight = 'medium')
ax2.set_xticks(list(range(0,5)), labels_work, rotation=30, ha='right')

# Titles
ax1.set_title('Working remotely', fontsize= 14, fontweight= 'heavy', color = 'black', y=1.15, pad=10)
ax2.set_title('Work interference of mental \nhealth condition and working type', fontsize= 14, fontweight= 'heavy', color = 'black', y=1.15, pad=10)

# Legend
ax1.legend(wedges, labels_rw,
          title="Remote working", loc = 'upper right',
          bbox_to_anchor=(0.60, 0.60, 0.5, 0.5))
ax2.legend(title = "Remote working", loc = 'upper right', bbox_to_anchor=(0.55, 0.55, 0.5, 0.5))

# Show
st.subheader("Remote working and work interference")
st.pyplot(fig1)
st.text("\n\n")


# FAMILY HISTORY AND TREATMENT 
# Data
# 1
labels_ans = list(reversed(mental_health_df['family_history'].value_counts().index.to_list()))                         # Yes, No
reverse_serie = mental_health_df['family_history'].value_counts().reindex(index= mental_health_df['family_history'].value_counts().index[::-1])
counts_ans = list(map(lambda x: round((x/tot_rows),2), reverse_serie))

# 2
df_family = mental_health_df.groupby(['family_history', 'treatment'])['treatment'].count().unstack(0)
df_family = df_family.reindex(index= df_family.index[::-1], columns=df_family.columns[::-1])               
yes_ans = list(df_family[:].loc['Yes'].values)
no_ans = list(df_family[:].loc['No'].values)

# The label locations
x = np.arange(len(labels_ans))  
width = 0.35  

# Plots
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize =(13, 5))
fig1.subplots_adjust(wspace=0.3)

wedges, text, autotext = ax1.pie(counts_ans, labels = labels_ans, labeldistance=1.15, wedgeprops = { 'linewidth' : 1, 'edgecolor' : 'white' }, 
        colors = colors , autopct='%1.0f%%', textprops={'fontsize': 12})
ax2.bar(x - width/2, yes_ans, width, color= 'steelblue', label= 'Yes')
ax2.bar(x + width/2, no_ans, width, color= 'lightblue', label ='No')

# Labels, ticks 
ax1.legend(wedges, labels_ans,
          title="Family history", loc = 'upper right',
          bbox_to_anchor=(0.60, 0.60, 0.5, 0.5))
ax2.set_xlabel("Family history", labelpad= 13.0, fontname="Arial", fontsize=11, fontweight = 'medium')
ax2.set_ylabel("Treatment search", labelpad= 13.0, fontname="Arial", fontsize=11, fontweight = 'medium')
ax2.set_xticks(x, labels_ans)

# Titles
ax1.set_title("Family History of the respondents", fontsize= 13, fontweight= 'heavy', color = 'black', y=1.1, pad=10)
ax2.set_title("Family History and seeking for treatment", fontsize= 13, fontweight= 'heavy', color = 'black', y=1.1, pad=10)

# Legend
ax2.legend(title = "Treatment search", loc = 'upper right', bbox_to_anchor=(0.80, 0.60, 0.5, 0.5))

# Show
st.subheader("Family history and treatment")
st.pyplot(fig1)
st.text("\n\n")


# CONSEQUENCE OF DISCUSSION OF MENTAL AND PHYSICAL HEALTH WITH THE EMPLOYER
# Data
# 1
mhc = mental_health_df['mental_health_consequence'].value_counts().reindex(index= mental_health_df['mental_health_consequence'].value_counts().index[::-1])
phc = mental_health_df['phys_health_consequence'].value_counts().reindex(index= mental_health_df['phys_health_consequence'].value_counts().index[::-1])

labels = mhc.index.to_list()                 
counts_mhc = list(map(lambda x: round((x/tot_rows),2), mhc))
counts_phc = list(map(lambda x: round((x/tot_rows),2), phc))
  
# 2
label_vs = mental_health_df['mental_vs_physical'].value_counts().reindex(['Yes', 'Don\'t know','No']).index.to_list()
counts_vs = list(map(lambda x: round((x/tot_rows),2), mental_health_df['mental_vs_physical'].value_counts().reindex(label_vs))) 

# The label locations
x = np.arange(len(labels))  
width = 0.35  

# Plots
fig1, (ax1,ax2) = plt.subplots(1,2,figsize =(15, 5))
fig1.subplots_adjust(wspace=0.5)

ax1.bar(x - width/2, counts_mhc, width, color= 'steelblue', label= 'Mental health')
ax1.bar(x + width/2, counts_phc, width, color= 'lightblue', label ='Physical health')
ax2.bar(label_vs, counts_vs, color ='steelblue', label=label_vs)


# Labels, ticks 
ax1.set_xlabel("Answers", labelpad= 13.0, fontname="Arial", fontsize=11, fontweight = 'medium')
ax1.set_ylabel("Percentages", labelpad= 13.0, fontname="Arial", fontsize=11, fontweight = 'medium')
ax2.set_xlabel("Answers", labelpad= 13.0, fontname="Arial", fontsize=11, fontweight = 'medium')
ax2.set_ylabel("Percentages", labelpad= 13.0, fontname="Arial", fontsize=11, fontweight = 'medium')
ax1.set_xticks(x, labels)
ax1.set_yticks(np.arange(0.0, 1.1, 0.1))
ax2.set_yticks(np.arange(0.0, 1.1, 0.1))

# Percentages
for p in ax1.patches:
   height = p.get_height() 
   ax1.annotate("{}%".format(round(height*100)), (p.get_x() + p.get_width() / 2, height+ 0.01), ha='center')

for p in ax2.patches:
   height = p.get_height() 
   ax2.annotate("{}%".format(round(height*100)), (p.get_x() + p.get_width() / 2, height+ 0.01), ha='center')

# Titles
ax1.set_title("Negative consequences of discussing of Mental \nand Physical health with Employer", fontsize= 13, fontweight= 'heavy', color = 'black', y=1.1, pad=10)
ax2.set_title("Seriousness of mental and physical illness compared", fontsize= 13, fontweight= 'heavy', color = 'black', y=1.1, pad=10)

# Legend
ax1.legend(title = 'Type of health', loc = 'upper right', bbox_to_anchor=(0.65, 0.55, 0.5, 0.5))

# Show
st.subheader("Consequence of discussion of mental and physical health with the employer")
st.pyplot(fig1)
st.text("\n\n")


# BENEFITS, WELNESS PROGRAM AND NUMBER OF EMPLOYEES
# 1
labels_ben = mental_health_df['benefits'].value_counts().index.to_list()               # 'Yes', "Don't know", 'No'
counts_ben = list(map(lambda x: round((x/tot_rows),2), mental_health_df['benefits'].value_counts()))

# 2
labels_no = [ '1-5', '6-25','26-100',  '100-500' , '500-1000',  'More than 1000']

df_no_ben = mental_health_df.groupby(['no_employees', 'benefits'])['benefits'].count().unstack(0)
df_no_ben = df_no_ben.reindex(index = labels_ben , columns= labels_no)
yes_perc1 = percent(df_no_ben, 'Yes')
no_perc1 = percent(df_no_ben, 'Don\'t know')
idk_perc1 = percent(df_no_ben, 'No')

# The label locations
x = np.arange(len(labels_no))  
width = 0.2  # the width of the bars

# Plots
fig1, (ax1, ax2)= plt.subplots(1, 2, figsize =(15, 5))
fig1.subplots_adjust(wspace=0.3)
ax1.bar(labels_ben, counts_ben, color ='steelblue', label=labels_ben)
ax2.bar(x - 0.2, yes_perc1, width, color= 'steelblue', label= 'Yes')
ax2.bar(x, no_perc1, width, color= 'lightblue', label ='No')
ax2.bar(x + 0.2, idk_perc1, width, color= 'lightsteelblue', label ='Don\'t know')

# Labels, ticks 
ax1.set_xlabel("Benefits", labelpad= 30.0, fontname="Arial", fontsize=12, fontweight = 'medium')
ax2.set_xlabel("Number of employees", labelpad= 10.0, fontname="Arial", fontsize=12, fontweight = 'medium')
ax1.set_ylabel("Percentages", labelpad= 10.0, fontname="Arial", fontsize=12, fontweight = 'medium')
ax2.set_ylabel("Percentages", labelpad= 10.0, fontname="Arial", fontsize=12, fontweight = 'medium')
ax2.set_xticks(list(range(0,6)), labels_no, rotation = 30, ha='right')
ax1.set_yticks(np.arange(0.0,1.1, 0.1))
ax2.set_yticks(np.arange(0.0,1.1, 0.1))

# Percenteges
for p in ax1.patches:
    height = p.get_height()
    ax1.annotate("{}%".format(round(height*100)), (p.get_x() + p.get_width() / 2, height+ 0.01), ha='center')

for p in ax2.patches:
    height = p.get_height()
    ax2.annotate("{}%".format(round(height*100)), (p.get_x() + p.get_width() / 2, height+ 0.01), ha='center', fontsize=8)

# Title
ax1.set_title('Commission of mental health benefits', fontsize= 14, fontweight= 'heavy', color = 'black', y=1.15, pad=10)
ax2.set_title('Number of employees and commission of mental health benefits', fontsize= 14, fontweight= 'heavy', color = 'black', y=1.15, pad=10)

# Legend
ax2.legend(title = 'Mental health benefits', loc = 'upper right', bbox_to_anchor=(0.65, 0.60, 0.5, 0.5))


# 3
labels_well = mental_health_df['wellness_program'].value_counts().reindex(labels_ben).index.to_list()          #'Yes', "Don't know", 'No'
counts_well = list(map(lambda x: round((x/tot_rows),2), mental_health_df['wellness_program'].value_counts().reindex(labels_well))) 

# 4
df_no_well = mental_health_df.groupby(['no_employees', 'wellness_program'])['wellness_program'].count().unstack(0)
df_no_well = df_no_well.reindex(index =labels_well , columns= labels_no)
yes_perc2 = percent(df_no_well, 'Yes')
no_perc2 = percent(df_no_well, 'Don\'t know')
idk_perc2 = percent(df_no_well, 'No')

# The label locations
x = np.arange(len(labels_no))  
width = 0.2  # the width of the bars

# Plot
fig2, (ax3, ax4)= plt.subplots(1, 2, figsize =(17, 5))
fig2.subplots_adjust(wspace=0.3)
ax3.bar(labels_well, counts_well, color ='steelblue', label=labels_well)
ax4.bar(x - 0.2, yes_perc2, width, color= 'steelblue', label= 'Yes')
ax4.bar(x, no_perc2, width, color= 'lightblue', label ='No')
ax4.bar(x + 0.2, idk_perc2, width, color= 'lightsteelblue', label ='Don\'t know')

# Labels, ticks 
ax3.set_ylabel("Percentages", labelpad= 10.0, fontname="Arial", fontsize=12, fontweight = 'medium')
ax3.set_ylabel("Percentages", labelpad= 10.0, fontname="Arial", fontsize=12, fontweight = 'medium')
ax4.set_xticks(list(range(0,6)), labels_no, rotation = 30, ha='right')
ax3.set_yticks(np.arange(0.0,1.1, 0.1))
ax4.set_yticks(np.arange(0.0,1.1, 0.1))

# Perc
for p in ax3.patches:
    height = p.get_height()
    ax3.annotate("{}%".format(round(height*100)), (p.get_x() + p.get_width() / 2, height+ 0.01), ha='center')

for p in ax4.patches:
    height = p.get_height()
    ax4.annotate("{}%".format(round(height*100)), (p.get_x() + p.get_width() / 2, height+ 0.01), ha='center', fontsize=8)

# Titles
ax3.set_title('Mental health as part of an employee wellness program', fontsize= 14, fontweight= 'heavy', color = 'black', y=1.15, pad=10)
ax4.set_title('Number of employees and Mental health as part of an \nemployee wellness program', fontsize= 14, fontweight= 'heavy', color = 'black', y=1.10, pad=10)

# Legend
ax4.legend(title = 'Mental health benefits', loc = 'upper right', bbox_to_anchor=(0.65, 0.60, 0.5, 0.5))

# Show
st.subheader("Benefits, wellness program and number of employees")
st.pyplot(fig1)
st.pyplot(fig2)
st.text("\n\n")


# DISCUSSING OF MENTAL HEALTH ISSUE WITH COWORKERS AND SUPERVISOR
# Data
co = mental_health_df['coworkers'].value_counts().reindex(index= mental_health_df['coworkers'].value_counts().index[::-1])
sup = mental_health_df['supervisor'].value_counts()

labels = co.index.to_list()                 
counts_co = list(map(lambda x: round((x/tot_rows),2), co))
counts_sup = list(map(lambda x: round((x/tot_rows),2), sup))
  
# The label locations
x = np.arange(len(labels))  
width = 0.35  

# Plots
fig1, ax1 = plt.subplots(figsize =(8, 5))
ax1.bar(x - width/2, counts_co, width, color= 'steelblue', label= 'Coworkers')
ax1.bar(x + width/2, counts_sup, width, color= 'lightblue', label ='Supervisor')

# Labels, ticks 
ax1.set_xlabel("Answers", labelpad= 13.0, fontname="Arial", fontsize=11, fontweight = 'medium')
ax1.set_ylabel("Percentages", labelpad= 13.0, fontname="Arial", fontsize=11, fontweight = 'medium')
ax1.set_xticks(x, labels)
ax1.set_yticks(np.arange(0.0, 1.1, 0.1))

# Percentages
for p in ax1.patches:
   height = p.get_height() 
   ax1.annotate("{}%".format(round(height*100)), (p.get_x() + p.get_width() / 2, height+ 0.01), ha='center')

# Titles
ax1.set_title("Discussing of Mental and Physical health issues\n with coworkers and supervisors", fontsize= 13, fontweight= 'heavy', color = 'black', y=1.1, pad=10)

# Legend
ax1.legend(title = 'Type of colleague', loc = 'upper right', bbox_to_anchor=(0.65, 0.55, 0.5, 0.5))

# Show
st.subheader("Discussing of Mental health issues with coworkers and supervisors")
st.pyplot(fig1)
st.text("\n\n")


# CONVERTING CATEGORICAL VALUES
mh_df_econded = encoding(mental_health_df)

# HEATMAP
# Compute the correlation matrix
corr = mh_df_econded.corr(numeric_only= False)

# The mask for lower left triangle 
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

# Plot
fig1, ax1 = plt.subplots( figsize=(20, 16) )
sb.heatmap(corr,  cmap = 'YlGnBu', annot = True, fmt=".3f", 
           linewidth=.5, cbar_kws={ 'orientation': 'vertical', 'shrink': 0.7 } , square=True, mask= mask,)
# Title
ax1.set_title("Heat map of the dataset", fontsize= 22, fontweight= 'heavy', color = 'black', pad= 0.7)

# Show
st.subheader("Heatmap")
st.pyplot(fig1)
st.text("\n\n")


### MODELS ###
st.header('Classification Model')

# CLASSIFCATION - Target : treatment 
st.subheader('Parameters')
st.write('Here we run model to predict if a person should be treated for a mental health condition according to the values in the dataset.')

# Classifiers
names = ['Nearest Neighbors', 'Linear SVM', 'Decision Tree', 'Random Forest', 'AdaBoost', 'Gradient Tree Boosting', 'Gaussian Naive Bayes' , 'QDA']
name_plus = names.copy()
name_plus.append('All')


classifiers = [
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(random_state=15),
    RandomForestClassifier(random_state=15),
    AdaBoostClassifier(random_state=15),
    GradientBoostingClassifier(random_state=15),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

# Attributes
attr = mh_df_econded.columns.to_list()
attr.remove('treatment')

# Declare the target variable
y = mh_df_econded['treatment']

# Select model, features and size
select_model = st.selectbox('Select model:', name_plus, index = 0)
choices = st.multiselect('Select features:', attr)
test_size = st.slider('Test size: ', min_value=0.1, max_value=0.9, step =0.1)
st.text("\n")

# If
model = KNeighborsClassifier() 
if select_model == 'Nearest Neighbors':
    model = KNeighborsClassifier()
elif select_model == 'Linear SVM':
    model = SVC()
elif select_model == 'Decision Tree':
    model = DecisionTreeClassifier(random_state=15)
elif select_model == 'Random Forest':
    model = RandomForestClassifier(random_state=15)
elif select_model == 'AdaBoost':
    model = AdaBoostClassifier(random_state=15)
elif select_model == 'Gradient Tree Boosting':
    model = GradientBoostingClassifier(random_state=15)
elif select_model == 'Gaussian Naive Bayes':
    model = GaussianNB()
elif select_model == 'QDA':
    model = QuadraticDiscriminantAnalysis()


# Run the model
if len(choices) > 0 and st.button('RUN MODEL'):
    with st.spinner('Training...'): 
        X = mh_df_econded.drop('treatment', axis = 1)
        X = X[choices]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
        x_train = x_train.to_numpy().reshape(-1, len(choices))
        x_test = x_test.to_numpy().reshape(-1, len(choices))

        # If one model il selected
        if select_model != 'All':
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.text("\n\n")
            st.subheader('Results')
            st.write(f'Accuracy = {accuracy:.2f}')

        # Il all models are selected
        else:
            accuracies = pd.Series(index = names, dtype=float)

            # iterate over classifiers
            for name, clf in zip(names, classifiers):
                model = clf
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                acc = accuracy_score(y_test, y_pred)
                accuracies.loc[name] = acc

            accuracies.sort_values(ascending=True, inplace=True)
            accuracies_df = pd.DataFrame({'Models':accuracies.index , 'Accuracies':accuracies.values})
            print(accuracies_df)

            # Plot
            fig, ax = plt.subplots(figsize = (8,4))
            ax = sb.barplot(accuracies_df, x = 'Accuracies', y ='Models', palette='Blues')
            ax.set_title("Plotting the Model Accuracies", fontsize=16, fontweight="bold", pad= 20, x = 0.35)

            # Show
            st.text("\n\n")
            st.subheader('Results')
            tab1, tab2 = st.tabs(['Accuracies chart', 'Accuracies table'])

            with tab1:
                st.pyplot(fig)
            with tab2:
                st.table(accuracies_df)

            
        


