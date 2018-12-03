# Recommender-System
Implementation of Recommender system based on MovieLens Dataset

Deducing interpretations from your raw data is tricky and we need to succeed below scenarios:
1.Understand what the users’ needs are
2.Prioritise all matches

DataSet : We have got two types of dataset here ml-100k and 20m and each have different types of files (csv and compressed tar files). So I have tried to make it out for both dataset.


Basic Steps Invloved : 

1)Choosing the dataset

2)Encoding your data:
  One-hot encoding
  Term frequency–inverse document frequency (TF-IDF) encoding(Used Here)
  Word embeddings 
  
3) Recommending content:
   Similarity-based Methods(Cosine Similarity Used here)
   One-class SVMs
   Matrix Factorisation
   Supervised Learning
   Deep Learning
  
4)Generating user preference profiles


As we know we have different types of filtering in the recommender system:

1.Content based : This algorithm recommends products which are similar to the ones that a user has liked in the past.
2.Collabrative based : This algorithm uses “User Behavior” for recommending items.

Here I have written three python code to implement recommender system:
1) Basic Recommender.py(100k Datset Used) : Gives a top down list of most rated Movies
2) Collabrative.py(100k Datset Used) : Gives the top list of movies rating based on the other movies liked by other user.
3) Contentbased.py(20m Dataset Used) :


Formulas used:
The formula used to calculate TF-IDF weight for term i in document j is:

w[i,j] = tf[i,j]*log(N/df[i])

Cosine Simialrity :
the cosine of this angle computed as follows:

cos(x,y) = dot(x,y)/|x||y|

Euclidean distance :

Euc  sqrt( (x1-x0)^2 + (y1-y0)^2 )



