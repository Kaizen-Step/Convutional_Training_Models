import numpy as np
import pandas as pd
import tensorflow as tf



# Loading Data

raw_data_csv = pd.read_csv('https://raw.githubusercontent.com/Kaizen-Step/Convutional_Training_Models/main/Data/15271310-Absenteeism-data.csv')

print(raw_data_csv.head(10))

df= raw_data_csv.copy()



# Exploring Data Sets features

print(df.shape)

print(raw_data_csv.describe())

print(raw_data_csv.isnull().any())

df.info()

df_prep1['Reason for Absence'].describe()

pd.options.display.max_rows = 30
pd.options.display.max_columns = 30
 
  
   

# Preprocessing Data

df_prep1 = df.copy()

df_prep1 = df_prep1.drop(['ID'],axis=1) # drop the column

df_dropped = df.reset_index(drop=True) # reset Index and remove old indexes

pd.unique(df['Reason for Absence']) # SHow the Unique values
df['Reason for Absence'].unique()

sorted(df['Reason for Absence'].unique()) # too see unique values in order


  


# Creat Dummies for Categorical features

reason_columns = pd.get_dummies(df['Reason for Absence'],drop_first = True) # drop first category to prevent the multilinearity

print(reason_columns)

reason_columns['check'] = reason_columns.sum(axis = 1) # check the dummies 

reason_columns = reason_columns.drop(['check'],axis = 1) # drop it if its ok

df.columns.values 

df = df.drop(['Reason for Absence'],axis = 1) # drop the old categorical feature from data set


# Creat group out of each category (desiese in this case)
reason_type1 =   reason_columns.loc[:,1:14].max(axis =1)
reason_type2 =   reason_columns.loc[:,15:17].max(axis =1)
reason_type3 =   reason_columns.loc[:,18:21].max(axis =1)
reason_type4 =   reason_columns.loc[:,22:].max(axis =1)

# Concatenate Columns Values

df = pd.concat([df, reason_type1, reason_type2, reason_type3,reason_type4], axis = 1)

print(df.head(10))

# Change the name of Columns

df.columns.values

columns_name = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason1', 'Reason2', 'Reason3', 'Reason4']

df.columns = columns_name

# Change the order of Columns 

df.columns.values

column_name_reorder = ['Reason1', 'Reason2', 'Reason3', 'Reason4','Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']

df = df[column_name_reorder]
 



# Work with Date Columns

type(df_reason_mod['Date'])
type(df_reason_mod['Date'][0]) # it would be string

df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'], format = '%d/%m/%Y') # Convert Date column to timestamp type

type(df_reason_mod['Date'][0]) # Must be timestampe

df_reason_mod.info() # you could see datetime on Dtype





# Extracting Month Value from Date Timestamp

df_reason_mod['Date'][0].month

list_month = []

for i in range(df_reason_mod.shape[0]):
  list_month.append(df_reason_mod['Date'][i].month)      # For loop is not recommended

df_reason_mod['Month Value'] = list_month

df_reason_mod.head(10)




# Extract day of a week from timestamp

df_reason_mod['Date'][300].weekday()

def date_to_weekdays(date_value):
  return date_value.weekday()  

df_reason_mod['Day of the week'] = df_reason_mod['Date'].apply(date_to_weekdays)




# Count the Unique Value

df_reason_mod['Education'].unique()

df_date_mode = df_reason_mod.copy()

df_date_mode['Education'].unique()

df_date_mode['Education'].value_counts() # Count each unique values



# change the Values of Columns with map method

df_date_mode['Education'] = df_date_mode['Education'].map({1:0, 2:1, 3:1, 4:1})

df_date_mode['Education'].unique()









