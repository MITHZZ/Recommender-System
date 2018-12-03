import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#column headers for the dataset
data_cols = ['user id','movie id','rating','timestamp']
item_cols = ['movie id','movie title','release date',
'video release date','IMDb URL','unknown','Action',
'Adventure','Animation','Childrens','Comedy','Crime',
'Documentary','Drama','Fantasy','Film-Noir','Horror',
'Musical','Mystery','Romance ','Sci-Fi','Thriller',
'War' ,'Western']
user_cols = ['user id','age','gender','occupation',
'zip code']

#importing the data files onto dataframes
users = pd.read_csv('u.user', sep='|',
names=user_cols, encoding='latin-1')
item = pd.read_csv('u.item', sep='|',
names=item_cols, encoding='latin-1')
data = pd.read_csv('u.data', sep='\t',
names=data_cols, encoding='latin-1')


#printing the head of these dataframes
print(users.head())
print(item.head())
print(data.head())

#printing the info of the data frame
print(users.info())
print(item.info())
print(data.info())

#First we merge the three dataframes into one single dataframe
dataset = pd.merge(pd.merge(item, data),users)
print(dataset.head())

#Next we use groupby to group the movies by their titles.
#Then we use the size function to returns the total number of 
#entries under each movie title. This will help us get the number 
#of people who rated the movie/ the number of ratings.

ratings_total = dataset.groupby('movie title').size()
print(ratings_total.head())

#Next we try to take the mean ratings of each movie using the mean function.
#First we groupby movie title. From the resulting dataframe we select only 
#the movie title and the rating headers. Then we use the mean function on them.

ratings_mean = (dataset.groupby('movie title'))['movie title','rating'].mean()
print(ratings_mean.head())

#Now if you check ratings_total then you will find its a Series and not a Data Frame. 
#So we will convert that into a dataframe. In the ratings_mean we will see that
#the movie title has been converted from a column to an index. So we make that a 
#column again.

ratings_total = pd.DataFrame({'movie title':ratings_total.index,
'total ratings': ratings_total.values})
ratings_mean['movie title'] = ratings_mean.index

#Now we head for the merging part. Now we sort the values by the total rating
#and this helps us sort the data frame by the number of people who viewed the movie.
final = pd.merge(ratings_mean, ratings_total).sort_values(by = 'total ratings',
ascending= False)
print(final.head())
print(final.describe())

final = final[:300].sort_values(by = 'rating',ascending = False)
final = final.rename(columns={'rating': 'mean rating'})
print(final)