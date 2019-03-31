import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math


# Setup a SparkSession
spark = SparkSession.builder.config("spark.rpc.message.maxSize",1024).getOrCreate()

# Ratings dataset
data = pd.read_csv('data/training.csv')
data = data.drop('timestamp',axis=1)

# Users metadata
usersdb = pd.read_csv('data/users.dat',sep='::',header=None,
                      names=['UserID','Gender','Age','Occupation','Zip-code'])
usersdb = pd.get_dummies(usersdb, columns=['Gender'], drop_first=True)

# K-Means clustering according to user age group
# We chose 4 age clusters as our ratings matrix contains a lot of missing values (only 3.7% of it is filled up)
# Using more clusters only marginally improved the silhouette score and left many observations unfilled

dataKmeans = usersdb['Age']
kmeans_model = KMeans(n_clusters=4, 
                      n_init = 10, 
                      max_iter =100, 
                      random_state=1).fit(dataKmeans)
labels=kmeans_model.labels_
plt.scatter(dataKmeans.iloc[:,0],dataKmeans.iloc[:,1],c=labels)
silhouette_score(dataKmeans, labels, metric='euclidean') #0.78


# Transforming the dataset from long format into wide format sparse matrix 
# This matrix will contain all user-movie combinations and we will fill in the missing values with 
# the average rating given to a movie by the users in the same age group
data_matrix_pd = pd.pivot_table(data,
                                values='rating', 
                                index=['user'], 
                                columns=['movie'], 
                                aggfunc=np.mean)
data_matrix_pd['index1'] = data_matrix_pd.index

# Adding K-Means labels to user-movie dataset through an outer join on user ID
labels=pd.DataFrame(labels)
labels['index1'] = labels.index+1

data_matrix_merged=pd.merge(labels, data_matrix_pd, on='index1',how='outer')
data_matrix_merged=data_matrix_merged.sort_values(by=['index1'])
data_matrix_merged=data_matrix_merged.set_index('index1')
data_matrix_merged.rename(columns ={0: 'label'}, inplace =True)

# Filling in missing values with the average rating given to a movie by the users in the same age group
for i in range(0,6040):
    for j in range(data_matrix_merged.shape[1]):
        if math.isnan(data_matrix_merged.iloc[i,j]):
            cluster=data_matrix_merged['label'].iloc[i]
            data_matrix_merged.iloc[i,j]=np.mean(
                data_matrix_merged.iloc[:,j].loc[data_matrix_merged['label'] == cluster])

# There are still some missing values, fill them in with column means (total average rating of a movie)
for j in range(data_matrix_merged.shape[1]):
    data_matrix_merged.iloc[:,j]=data_matrix_merged.iloc[:,j].fillna(
                                            data_matrix_merged.iloc[:,j].mean())
    
#Convert the wide matrix back into long format suitable for ALS model
data_matrix_merged = data_matrix_merged.drop('label',axis=1)
data_matrix_long=data_matrix_merged.unstack().reset_index()
data_matrix_long.rename(columns ={'level_0': 'movie',
                                  'index1': 'user',
                                  0: 'rating'}, inplace =True)


# Convert a Pandas DF to a Spark DF
spark_df = spark.createDataFrame(data) 

# Train, test split of the dataset for cross-validation purposes
train, test = spark_df.randomSplit([0.8, 0.2], seed=427471138)

# Create an untrained ALS model
als_model = ALS(
    itemCol='movie',
    userCol='user',
    ratingCol='rating',
    regParam = 0.1,
    nonnegative=True) 

# Fitting ALS model
recommender = als_model.fit(train)

# Adding ALS predicted ratings
train_tr = recommender.transform(train)
test_tr = recommender.transform(test)

# Calculating rmse of train and test samples
evaluator = RegressionEvaluator(predictionCol="prediction", 
                                labelCol="rating",
                                metricName="rmse")

rmse_train=evaluator.evaluate(train_tr)
rmse_test=evaluator.evaluate(test_tr)


# Performing a Grid search in order to check if we could fine-tune the model parameters
param_grid = ParamGridBuilder()\
                .addGrid(als_model.rank, [5, 10, 15, 20])\
                .addGrid(als_model.maxIter, [1, 2, 3, 4, 5])\
                .addGrid(als_model.regParam, [0.05, 0.1, 0.15, 0.2])\
                .build()

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="rating",metricName="rmse")

tvs = TrainValidationSplit(
        estimator = als_model,
        estimatorParamMaps=param_grid,
        evaluator=evaluator)

model = tvs.fit(train)
best_model = model.bestModel

# Use the best model returned by Grid search to make predictions for train and test datasets
train_tr = best_model.transform(train)
test_tr = best_model.transform(test)

# Print best_model parameters
print(best_model.rank)
print(best_model._java_obj.parent().getMaxIter())
print(best_model._java_obj.parent().getRegParam())

# Best_model rmse - train dataset
rmse_train=evaluator.evaluate(train_tr)
rmse_train

# Best_model rmse - test dataset
rmse_test=evaluator.evaluate(test_tr)
rmse_test

# Grid search parameters improved rmse scores, therefore we will use these parameters to fine-tune the ALS model
# Final model
als_model = ALS(
    itemCol='movie',
    userCol='user',
    ratingCol='rating',
    regParam = 0.05,
    rank = 5,
    maxIter = 5,
    nonnegative=True) 

# Fit the model on entire dataset
recommender = als_model.fit(spark_df)

# We will use this fitted model to generate predictions for user-movie combinations in 'requests.csv'.
# This file does not contain actual ratings
# This file will be used to calculate our group score

# Case study group score will be computed in the following way:
#    - Our prediction file will be used to extract, for each user, the 5% most highly predicted movies
#    - The mean of the actual ratings of those movies (saved in a hidden testing set) will be computed
#    - Group with the highest mean wins

data_requests = pd.read_csv('requests.csv')
data_requests_spark = spark.createDataFrame(data_requests)

# Generating predictions for user-movie combinations in 'requests.csv'
data_requests_spark_tr = recommender.transform(data_requests_spark)

# Transforming the dataset to Pandas dataframe and renaming prediction column to 'rating'
data_requests_spark_pd = data_requests_spark_tr.toPandas()
data_requests_spark_pd.rename(columns={'prediction':'rating'}, inplace=True)

# Filling in any remaining missing values with the dataset mean
data_requests_spark_pd = data_requests_spark_pd.fillna(data_matrix_long['rating'].mean())

# Saving dataframe as csv file
data_requests_spark_pd.to_csv('submission.csv',columns=["user","movie","rating"], index=False)

# =============================================================================
# 03/18/2019 08:58:37 AM : DEBUG : score=4.27985347985348

# Top 5% of our predicted movie ratings had an actual average rating of 4.28
# =============================================================================





