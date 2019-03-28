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

# Training sample
data = pd.read_csv('data/training.csv')
data = data.drop('timestamp',axis=1)

# Test sample
data_requests = pd.read_csv('data/requests.csv')
data_requests_spark = spark.createDataFrame(data_requests)

# Users metadata
usersdb = pd.read_csv('data/users.dat',sep='::',header=None,
                      names=['UserID','Gender','Age','Occupation','Zip-code'])
usersdb = pd.get_dummies(usersdb, columns=['Gender'], drop_first=True)

#Kmeans clustering by age groups
dataKmeans = usersdb[['Age']]
kmeans_model = KMeans(n_clusters=4, 
                      n_init = 10, 
                      max_iter =100, 
                      random_state=1).fit(dataKmeans)
labels=kmeans_model.labels_
silhouette_score(dataKmeans, labels, metric='euclidean') #0.78
plt.scatter(dataKmeans.iloc[:,0],dataKmeans.iloc[:,1],c=labels)

#Joining kmeans labels and main dataset on user id
data_matrix_pd = pd.pivot_table(data,
                                values='rating', 
                                index=['user'], 
                                columns=['movie'], 
                                aggfunc=np.mean)
data_matrix_pd['index1'] = data_matrix_pd.index

labels=pd.DataFrame(labels)
labels['index1'] = labels.index+1

data_matrix_merged=pd.merge(labels, data_matrix_pd, on='index1',how='outer')
data_matrix_merged=data_matrix_merged.sort_values(by=['index1'])
data_matrix_merged=data_matrix_merged.set_index('index1')
data_matrix_merged.rename(columns ={0: 'label'}, inplace =True)

#Filling in missing values with the average of ratings given to a movie 
#by users in the same age cluster
data_matrix_merged1 = pd.read_csv('data/data_matrix_new.csv')
data_matrix_merged1 = data_matrix_merged1.set_index('index1')
cols=data_matrix_merged.columns
data_matrix_merged1.columns = cols
data_matrix_merged = data_matrix_merged1

for i in range(6039,6040):
    for j in range(data_matrix_merged.shape[1]):
        if math.isnan(data_matrix_merged.iloc[i,j]):
            cluster=data_matrix_merged['label'].iloc[i]
            data_matrix_merged.iloc[i,j]=np.mean(data_matrix_merged.iloc[:,j].loc[data_matrix_merged['label'] == cluster])

data_matrix_merged.to_csv('data/data_matrix_merged.csv')

#There are still some missing values, fill them in with column means, ie total average rating of a movie
for j in range(data_matrix_merged.shape[1]):
    data_matrix_merged.iloc[:,j]=data_matrix_merged.iloc[:,j].fillna(
                                            data_matrix_merged.iloc[:,j].mean())
    
#Convert the wide matrix into long format suitable for ALS model
data_matrix_merged = data_matrix_merged.drop('label',axis=1)
data_matrix_long=data_matrix_merged.unstack().reset_index()
data_matrix_long.rename(columns ={'level_0': 'movie',
                                  'index1': 'user',
                                  0: 'rating'}, inplace =True)


data_matrix_long.to_csv('data/data_matrix_long.csv', index=False)

# Convert a Pandas DF to a Spark DF
spark_df = spark.createDataFrame(data) 
spark_df = spark.read.csv('data/data_matrix_long.csv',header=True,inferSchema=True)

#Split into train and test sets
train, test = spark_df.randomSplit([0.8, 0.2], seed=427471138)

# Create an untrained ALS model
als_model = ALS(
    itemCol='movie',
    userCol='user',
    ratingCol='rating',
    regParam = 0.1,
    nonnegative=True) 

recommender = als_model.fit(train)

##Add predictions
train_tr = recommender.transform(train)
test_tr = recommender.transform(test)

evaluator = RegressionEvaluator(predictionCol="prediction", 
                                labelCol="rating",
                                metricName="rmse")

rmse_train=evaluator.evaluate(train_tr)
rmse_test=evaluator.evaluate(test_tr)

# =============================================================================
#Fitting entire dataset
model_final = tvs.fit(spark_df)
best_model_final = model_final.bestModel
 
spark_df_tr = best_model_final.transform(spark_df).toPandas()
spark_df_tr.info()

#Calculating RMSE for train and test sets
rmse_spark_df=evaluator.evaluate(spark_df_tr)
print(best_model_final.rank)
print(best_model_final._java_obj.parent().getMaxIter())
print(best_model_final._java_obj.parent().getRegParam())
 
data_requests_transformed = best_model_final.transform(data_requests_spark).toPandas()
data_requests_transformed = data_requests_transformed.fillna(name_df['rating'].mean())
# =============================================================================





