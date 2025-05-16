## Import libraries/packages
import numpy as np
import pandas as pd
from datetime import datetime as dt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

## Define states dictionary to convert state abbreviations to names
state_dict = {
    'AL': 'Alabama',
    'AK': 'Alaska',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming',
    'DC': 'District of Columbia',
    'PR': 'Puerto Rico'
}

## Import data
df = pd.read_csv('medical_clean.csv')
## Map state abbrv to name
if len(df['State'][0]) == 2:
    df['State'] = df['State'].map(state_dict)
## Drop extraneous columns
df1 = df.drop([
    'CaseOrder',
    'Customer_id',
    'Interaction',
    'UID',
    'City',
    'State',
    'County', 
    'Zip', 
    'Lat', 
    'Lng',
    'TimeZone', 
    'Job',
    'Item1', 
    'Item2', 
    'Item3', 
    'Item4',
    'Item5', 
    'Item6', 
    'Item7', 
    'Item8'
], axis=1)
## Separate string columns from numeric columns
df2 = pd.DataFrame()
for col in df1.columns:
    if df1[col].dtype == object:
        df2[col] = df1[col]
        df1 = df1.drop(col, axis=1)
## Remove outliers
df1 = df1.drop(zscore(df1)[abs(zscore(df1)) > 3].dropna(thresh=1).index.values)
## Remove outlier observations from string columns
df2 = df2.loc[df1.index]
## Encode binary variables
labels = {'Yes': 1, 'No':0}
for col in df2.columns:
    if 'Yes' in df2[col].unique():
        df1[col] = df2[col].map(labels)
        df2 = df2.drop(col, axis=1)
## Keep relevant rows and columns
df[df1.columns] = df1
df = df[df.columns[4:]]
df = df.loc[df1.index]
df = df.reset_index(drop=True)
## Get dummy variables
df1 = pd.concat([df1, pd.get_dummies(df2)], axis=1)
## Reset index
df1 = df1.reset_index(drop=True)
## Split training and testing data
X, y = df1.drop('ReAdmis', axis=1), df1[['ReAdmis']]
X_train, X_test, y_train, y_test = train_test_split(X, y)
## Train logistic regression model
clf = LogisticRegression(random_state=10).fit(X_train, y_train)
## Get the mean accuracy score
print(f'Model Accuracy: {round(clf.score(X_test, y_test)*100, 2)}%')
## Get the expected readmission rate
df['ExpectedReAdmis'] = clf.predict(X)
## Create admission column
df['Admissions'] = 1
## Store clean data
df.to_csv('medical_data_clean.csv', index=False)

## Import data
df = pd.read_csv('FY_2023_Hospital_Readmissions_Reduction_Program_Hospital.csv')
## Drop extraneous columns
df.drop([
    'Measure Name', 'Footnote', 
    'Excess Readmission Ratio', 
    'Predicted Readmission Rate',
], axis=1, inplace=True)
## Drop missing values
df.dropna(inplace=True)
## Drop duplicate values
df.drop(df[df.duplicated()].index, inplace=True)
## Verify no duplicates left
print(f'Duplicate Observations: {df.duplicated().sum()}')
## Reset df index
df.reset_index(drop=True, inplace=True)
## Change numeric values to integers
df['Number of Discharges'] = df['Number of Discharges'].astype(int)
df['Number of Readmissions'] = df['Number of Readmissions'].astype(int)
## Extract start and end year
start = dt.strptime(df['Start Date'][0], '%m/%d/%Y').year
end = dt.strptime(df['End Date'][0], '%m/%d/%Y').year
## Calculate average annual values
df['Number of Discharges'] = round(df['Number of Discharges'] / (end - start)).astype(int)
df['Number of Readmissions'] = round(df['Number of Readmissions'] / (end - start)).astype(int)
## Drop date columns
df.drop(['Start Date', 'End Date'], axis=1, inplace=True)
## Rename columns
df.columns = [
    'Hospital Name', 
    'Hospital ID', 
    'State', 
    'Avg Annual Admissions', 
    'Avg Expected Readmission Rate', 
    'Avg Annual Readmissions'
]
## Map state abbvr to name
df['State'] = df['State'].map(state_dict)
## Store clean data
df.to_csv('readm_clean.csv', index=False)