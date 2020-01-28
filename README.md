# Task

To build a simple music recommendation system to illustrate basic machine learning concepts

## Description 

The goal of a Recommender System is to dentify products most relevant to the user. To study patterns between users and songs and to provide recommendations based on it.

## Motivation
-Inspired by the various music recommnedation applications like Spotify and Saavn, we tried to implement the same by displaying a simple song recommendation system.

## Technologies- 
Python 3.6
Jupyter Notebook


## Approach
Popularity based model that we have implemented groups all the songs by listen count essentially picking the most popular songs listened by users ,this is not the most efficient method as it recommends the same songs for every user without taking into consideration their individual preferences.

```bash
user_id = users[5]
user_items = is_model.get_user_items(user_id)
#
print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")
is_model.recommend(user_id)
```
## Model based on Collaborative Filtering (Matrix Factorization)
identify latent (hidden) features from the input user x itemRatings matrix to represent users and items as vectors N dimensional space.
We use the following code to make predictions for a user with userid uTest

Matrix Factorization based Recommender System - Using SVD matrix factorization based collaborative filtering recommender system.
The following code implements a Singular Value Decomposition (SVD) based matrix factorization collaborative filtering recommender system. The user ratings matrix used is a small matrix as follows:

        Item0   Item1   Item2   Item3
User0     3        1       2      3
User1     4        3       4      3
User2     3        2       1      5
User3     1        6       5      2
User4     0        0       5      0

As we can see in the above matrix, all users except user 4 rate all items. The code calculates predicted recommendations for user 4.
Import the required libraries
Methods to compute SVD and recommendations
Use SVD to make predictions for a test user id
Understanding Intuition behind SVD-SVD result gives three matrices as output: U, S and Vt (T in Vt means transpose). Matrix U represents user vectors and Matrix Vt represents item vectors. In simple terms, U represents users as 2 dimensional points in the latent vector space, and Vt represents items as 2 dimensional points in the same space.Next, we print the matrices U, S and Vt and try to interpret them. Think how the points for users and items will look like in a 2 dimensional axis. For example, the following code plots all user vectors from the matrix U in the 2 dimensional space. Similarly, we plot all the item vectors in the same plot from the matrix Vt.

```bash
print("Predictied ratings:")
uTest_recommended_items = computeEstimatedRatings(urm, U, S, Vt, uTest, K, True)
print(uTest_recommended_items)
```

Precision=Number of products relevant and recommended /Number of items recommended

Recall=Number of products relevant and recommended /Number of relevant items 


## View of code

- Load music data - We read userid-songid-listen_count triplets
- Explore data - Here the music data shows how many times a user listened to a song, as well as the details of the song
- We Create a subset of the dataset
- Show the most popular songs in the dataset
- Create a song recommender
- Simple popularity-based recommender class (Can be used as a black box) and create an instance of that class 
- Use the popularity model to make some predictions
- Build a song recommender with personalization- We now create an item similarity based collaborative filtering model that allows us to make personalized recommendations to each user.
- Create an instance of item similarity based recommender class
- Use the personalized model to make some song recommendations
- We can also apply the model to find similar songs to any song in the dataset
- Quantitative comparison between the models- To formally compare the popularity and the personalized models using precision-recall curves
- Use the above precision recall calculator class to calculate the evaluation measures
- Code to plot precision recall curve
- Generate Precision Recall curve using pickled results on a larger data subset


## Futher Improvements-
Building a Bayesian network which will allow you to do probabilistic recommendations and also using Frequent Itemset Mining .

## Resources
Book- Recommender Systems An Introduction by Dietmar Jannach
Source: http://labrosa.ee.columbia.edu/millionsong/
Paper: http://ismir2011.ismir.net/papers/OS6-1.pdf


