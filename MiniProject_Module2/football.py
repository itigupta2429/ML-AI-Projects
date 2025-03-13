##Import pandas
import pandas as pd
##loading the data
data=pd.read_csv('MiniProject_Module2/results.csv')
##removing the NA values
new_data=data.dropna()
print(new_data.shape)
##How many tuples are there in the dataset?
print(new_data.shape[0])
##How many tournaments are there in the dataset?
print(new_data['tournament'].nunique())
new_data['date']=new_data['date'].apply(pd.Timestamp)
#print(new_data.head)

##How many matches were played in 2018?
matches_2018=new_data[new_data['date'].dt.year==2018].shape[0]
print(matches_2018)
##How many time home tem won,lost or had a draw?
matches_won=len(new_data[(new_data['home_score'] > new_data['away_score'])])
print(matches_won)

matches_lost=len(new_data[(new_data['home_score'] < new_data['away_score'])])
print(matches_lost)

matches_draw=len(new_data[(new_data['home_score'] == new_data['away_score'])])
print(matches_draw)

##Pie chart
import matplotlib.pyplot as plt
pie_data=pd.array([matches_won,matches_lost,matches_draw])
labels=['Win', 'Lost', 'Draw']
print(pie_data, labels)

plt.pie(pie_data, labels=labels, autopct='%1.2f%%')
plt.title("Distribution of Wins, Losses & Draws")
plt.show()
