##Import pandas
import pandas as pd
##loading the data
data=pd.read_csv('MiniProject_Module2/flavors_of_cacao.csv')
print(data.head())
##removing the NA values
new_data=data.dropna()
##How many tuples are there in the dataset?
print(new_data.shape[0])

##How many unique company names are there in the dataset?
print(new_data.iloc[:,0].nunique())

##How many reviews are made in 2013 in the dataset?
#reviews_2013=new_data[new_data.iloc[3].dt.year==2013].shape[0]
#print(reviews_2013)
print(new_data.columns[3])