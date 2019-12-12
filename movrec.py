#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import functions as F
from pyspark.sql import DataFrameNaFunctions as DFna
from pyspark.sql.functions import udf, col, when
import matplotlib.pyplot as plt
import pyspark as ps
import os, sys, requests, json


spark = ps.sql.SparkSession.builder             .master("local[4]")             .appName("building recommender")             .getOrCreate() # create a spark session
            
sc = spark.sparkContext # create a spark context


# In[2]:


# read movies CSV
movies_df = spark.read.csv(r"C:\Users\hp\Anaconda3\ml-20m\movies.csv",
                         header=True,       # use headers or not
                         quote='"',         # char for quotes
                         sep=",",           # char for separation
                         inferSchema=True)  # do we infer schema or not ?
movies_df.printSchema()


# In[3]:


print("line count: {}".format(movies_df.count()))


# In[4]:


# read ratings CSV
ratings_df = spark.read.csv(r"C:\Users\hp\Anaconda3\ml-20m\movies.csv",
                         header=True,       # use headers or not
                         quote='"',         # char for quotes
                         sep=",",           # char for separation
                         inferSchema=True)  # do we infer schema or not ?
ratings_df.printSchema()


# In[6]:


print("line count: {}".format(movies_df.count()))


# In[7]:


# read ratings CSV
ratings_df = spark.read.csv(r"C:\Users\hp\Anaconda3\ml-20m\ratings.csv",
                         header=True,       # use headers or not
                         quote='"',         # char for quotes
                         sep=",",           # char for separation
                         inferSchema=True)  # do we infer schema or not ?
ratings_df.printSchema()


# In[8]:


ratings = ratings_df.rdd

numRatings = ratings.count()
numUsers = ratings.map(lambda r: r[0]).distinct().count()
numMovies = ratings.map(lambda r: r[1]).distinct().count()

print ("Got %d ratings from %d users on %d movies." % (numRatings, numUsers, numMovies))


# In[9]:


movies_counts = ratings_df.groupBy(col("movieId")).agg(F.count(col("rating")).alias("counts"))
movies_counts.show()


# In[10]:


ratings_df.take(5)


# In[11]:


movies_df.take(5)


# In[12]:


training_df, validation_df, test_df = ratings_df.randomSplit([.6, .2, .2], seed=0)
training_df


# In[13]:


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql import Row
import numpy as np
import math


# In[14]:


seed = 5
iterations = 10
regularization_parameter = 0.1
ranks = range(4, 12)
errors = []
err = 0
tolerance = 0.02


# In[15]:


min_error = float('inf')
best_rank = -1
best_iteration = -1

for rank in ranks:
    als = ALS(maxIter=iterations, regParam=regularization_parameter, rank=rank, userCol="userId", itemCol="movieId", ratingCol="rating")
    model = als.fit(training_df)
    predictions = model.transform(validation_df)
    new_predictions = predictions.filter(col('prediction') != np.nan)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(new_predictions)
    errors.append(rmse)

    print ("For rank %s the RMSE is %s" % (rank, rmse))
    if rmse < min_error:
        min_error = rmse
        best_rank = rank
print ("The best model was trained with rank %s" % best_rank)


# In[16]:


als = ALS(maxIter=iterations, regParam=regularization_parameter, rank=rank, userCol="userId", itemCol="movieId", ratingCol="rating")
paramGrid = ParamGridBuilder()     .addGrid(als.regParam, [0.1, 0.01])     .addGrid(als.rank, range(4, 12))     .build()
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)
cvModel = crossval.fit(training_df)


# In[17]:


cvModel_pred = cvModel.transform(validation_df)
cvModel_pred = cvModel_pred.filter(col('prediction') != np.nan)
rmse = evaluator.evaluate(cvModel_pred)
print ("the rmse for optimal grid parameters with cross validation is: {}".format(rmse))


# In[18]:


final_als = ALS(maxIter=10, regParam=0.1, rank=6, userCol="userId", itemCol="movieId", ratingCol="rating")
final_model = final_als.fit(training_df)
final_pred = final_model.transform(validation_df)
final_pred = final_pred.filter(col('prediction') != np.nan)
rmse = evaluator.evaluate(final_pred)
print ("the rmse for optimal grid parameters with cross validation is: {}".format(rmse))


# In[19]:


# read links CSV
links_df = spark.read.csv(r'C:\Users\hp\Anaconda3\data\movies\links.csv',
                         header=True,       # use headers or not
                         quote='"',         # char for quotes
                         sep=",",           # char for separation
                         inferSchema=True)  # do we infer schema or not ?
links_df.printSchema()


# In[20]:


np.random.seed(42)
user_id = np.random.choice(numUsers)


# In[24]:


new_user_ratings = ratings_df.filter(ratings_df.userId == user_id)
new_user_ratings.sort('rating', ascending=True).take(10) # top rated movies for this user


# In[22]:


new_user_ratings.describe('rating').show()


# In[25]:


new_user_ratings.toPandas()['rating'].hist()
plt.show()


# In[26]:


new_user_rated_movieIds = [i.movieId for i in new_user_ratings.select('movieId').distinct().collect()]
movieIds = [i.movieId for i in movies_counts.filter(movies_counts.counts > 25).select('movieId').distinct().collect()]
new_user_unrated_movieIds = list(set(movieIds) - set(new_user_rated_movieIds))


# In[27]:


import time
num_ratings = len(new_user_unrated_movieIds)
cols = ('userId', 'movieId', 'timestamp')
timestamps = [int(time.time())] * num_ratings
userIds = [user_id] * num_ratings
# ratings = [0] * num_ratings
new_user_preds = spark.createDataFrame(zip(userIds, new_user_unrated_movieIds, timestamps), cols)


# In[28]:


new_user_preds = final_model.transform(new_user_preds).filter(col('prediction') != np.nan)


# In[29]:


new_user_preds.sort('prediction', ascending=False).take(10)


# In[30]:


api_key="efcfb5ada3f2766c2e7cefa9522746c4"
headers = {'Accept': 'application/json'}
payload = {'api_key': api_key} 
response = requests.get("http://api.themoviedb.org/3/configuration", params=payload, headers=headers)
response = json.loads(response.text)
base_url = response['images']['base_url'] + 'w185'

def get_poster(tmdb_id, base_url):
    
    # Query themoviedb.org API for movie poster path.
    movie_url = 'http://api.themoviedb.org/3/movie/{:}/images'.format(tmdb_id)
    headers = {'Accept': 'application/json'}
    payload = {'api_key': api_key} 
    response = requests.get(movie_url, params=payload, headers=headers)
    file_path = json.loads(response.text)['posters'][0]['file_path']
    return base_url + file_path


# In[31]:


from IPython.display import Image
from IPython.display import display


# In[32]:


new_user_ratings = new_user_ratings.sort('rating', ascending=False).join(links_df, new_user_ratings.movieId == links_df.movieId)


# In[33]:


posters = tuple(Image(url=get_poster(movie.tmdbId, base_url)) for movie in new_user_ratings.take(10))


# In[34]:


display(*posters)


# In[35]:


new_user_preds = new_user_preds.sort('prediction', ascending=False).join(links_df, new_user_preds.movieId == links_df.movieId)


# In[36]:


posters = tuple(Image(url=get_poster(movie.tmdbId, base_url)) for movie in new_user_preds.take(10))


# In[37]:


display(*posters)


# In[ ]:


