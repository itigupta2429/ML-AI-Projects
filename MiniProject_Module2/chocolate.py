##Import pandas
import pandas as pd
##loading the data
data=pd.read_csv('MiniProject_Module2/flavors_of_cacao.csv')
print(data.head())
##removing the NA values
new_data=data.dropna()

##Renaming columns to remove new line character from column names
new_data = new_data.rename(columns=lambda x: x.strip().replace("\n", " "))

##How many tuples are there in the dataset?
print(new_data.shape[0])

##How many unique company names are there in the dataset?
print(new_data.iloc[:,0].nunique())

##How many reviews are made in 2013 in the dataset?
new_data["Review Date"] = pd.to_datetime(new_data["Review Date"], format="%Y")
reviews_2013 = new_data[new_data["Review Date"].dt.year == 2013].shape[0]
print(reviews_2013)

##In the BeanType Column, how many missing values are there?
print(new_data['Bean Type'].astype(str).apply(lambda x: x.count('\xa0')).sum())

##Histogram
import matplotlib.pyplot as plt

'''
plt.hist(new_data['Rating'])
plt.title("Ratings Distribution")
plt.show()
'''
##The histogram shows that most of the chocolates has rating in between 3-3.5
pd.set_option('display.max_columns', None)
new_data['Cocoa Percent'] = new_data['Cocoa Percent'].astype(str).str.replace('%', '').astype(float)
print(new_data.head())

##Scatter plot
'''
plt.scatter(x=new_data['Cocoa Percent'],y=new_data['Rating'],alpha=0.1)
plt.show()
'''
##Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Rating_normalized = scaler.fit_transform(new_data['Rating'].values.reshape(-1, 1))
print(Rating_normalized)

##Step 7
#print(new_data.columns)
company_average_score=new_data.groupby(new_data.columns[0])['Rating'].mean()
print(company_average_score)