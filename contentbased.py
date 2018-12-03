#content based recommender

# Imports
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read dataframes
df_movies = pd.read_csv('movies.csv')
df_links = pd.read_csv('links.csv')
df_ratings = pd.read_csv('ratings.csv')
df_genome_tags = pd.read_csv('genome-tags.csv')
df_genome_scores = pd.read_csv('        genome-scores.csv')

# Merge scores and tags
df_movie_tags_in_text = pd.merge(df_genome_scores, df_genome_tags, on='tagId')[['movieId', 'tag', 'relevance']]

# Only keep tags with relevance higher than 0.3
df_movie_tags = df_genome_scores[df_genome_scores.relevance > 0.3][['movieId', 'tagId']]

#Encode features
df_tags_to_movies = pd.merge(df_movie_tags, df_genome_tags, on='tagId', how='left')[['movieId', 'tagId']]
df_tags_to_movies['tagId'] = df_tags_to_movies.tagId.astype(str)

def _concatenate_tags_of_movie(tags):
    tags_as_str = ' '.join(set(tags))
    return tags_as_str

#List of movies as per differ tag on each    
df_tags_per_movie = df_tags_to_movies.groupby('movieId')['tagId'].agg(_concatenate_tags_of_movie)
df_tags_per_movie.name = 'movie_tags'
df_tags_per_movie = df_tags_per_movie.reset_index()

df_tags_per_movie[df_tags_per_movie['movieId'] == 1]

df_avg_ratings  = df_ratings.groupby('movieId')['rating'].agg(['mean', 'median', 'size'])
df_avg_ratings.columns = ['rating_mean', 'rating_median', 'num_ratingsdf_tags_per_movie']
df_avg_ratings = df_avg_ratings.reset_index()

df_movies_with_ratings = pd.merge(df_movies, df_avg_ratings, how='left', on='movieId')

df_data = pd.merge(df_movies_with_ratings, df_tags_per_movie, how='left', on='movieId')

df_data_with_tags = df_data[~df_data.movie_tags.isnull()].reset_index(drop=True)



#TF-IDF vectors used for list in the dataset so that we get a vector of the df_data_with_tags

#Object initiatize the TDIF method
tf_idf = TfidfVectorizer()


#transform the dataframe by fiting to the tdif function
df_movies_tf_idf_described = tf_idf.fit_transform(df_data_with_tags.movie_tags)

#We have the vector version of df_data_with_tags dataframe , So it make simple to find the cosine similarity.

m2m = cosine_similarity(df_movies_tf_idf_described)

df_tfidf_m2m = pd.DataFrame(cosine_similarity(df_movies_tf_idf_described))

index_to_movie_id = df_data_with_tags['movieId']

#Pulling out the movieid and its corresdponding values to make the matrix of vector converted cosine similarity of tags

df_tfidf_m2m.columns = [str(index_to_movie_id[int(col)]) for col in df_tfidf_m2m.columns]

df_tfidf_m2m.index = [index_to_movie_id[idx] for idx in df_tfidf_m2m.index]

print(df_tfidf_m2m.head())

#Recommeding for users let it be user 1

df_user_ratings = df_ratings[df_ratings.userId == 1]

df_user_data_with_tags = df_data_with_tags.reset_index().merge(df_user_ratings, on='movieId')

print(df_user_data_with_tags[['title', 'rating']])

# weight calculation of rating by dividing by 5

df_user_data_with_tags['weight'] = df_user_data_with_tags['rating']/5.

#dot matrix of vectors weight and index of the df_user_data_with_tags

user_profile = np.dot(df_movies_tf_idf_described[df_user_data_with_tags['index'].values].toarray().T, df_user_data_with_tags['weight'].values)

# similarity finding between the matric of weighted and tags which are tdif
C = cosine_similarity(np.atleast_2d(user_profile), df_movies_tf_idf_described)

#Sorting the similaity list

R = np.argsort(C)[:, ::-1]

recommendations = [i for i in R[0] if i not in df_user_data_with_tags['index'].values]

# list of top recommeded movies for user 1

print(df_data_with_tags['title'][recommendations].head(10))







