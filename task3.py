#Task: Find the frequent seasonal words. For example, are some words frequent in the evening but not in
#the rest of the day? For this, find which of the tweets are written in the evening. Consider that all
#the tweets in French were written by people living in France and all the tweets written in English
#by people in the UK.

import pandas as pd
pd.set_option('display.max_columns',100, 'display.max_colwidth',1000, 'display.max_rows',1000,
              'display.float_format', lambda x: '%.2f' % x)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sparknlp
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .master("local[4]")\
    .config("spark.driver.cores", 10)\
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5")\
    .config('spark.executor.memory', '4g') \
    .config("spark.driver.memory", "15g") \
    .config('spark.driver.maxResultSize', '4g') \
    .getOrCreate()
from sparknlp.pretrained import PretrainedPipeline
from pyspark.sql.functions import from_unixtime, to_date, year, udf, explode, split, col, length, rank, dense_rank, avg, sum
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.stat import Correlation
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import nltk
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud

from textblob import TextBlob

file_name1 = "English/" + "NoFilterEnglish2020-02-01.json"
file_name2 = "English/" + "NoFilterEnglish2020-02-02.json"
file_name3 = "English/" + "NoFilterEnglish2020-02-03.json"

# Read JSON file into dataframe
df = spark.read.json([file_name1, file_name2, file_name3])                       #.repartition(4).persist()
df = df.select('id','geo','timestamp_ms', 'retweeted', 'text', 'created_at',
                              'retweet_count','reply_count')
df.printSchema()

#CreateOrReplaceTempView will create a temporary view of the table on memory it is not presistant at this moment
#but you can run sql query on top of that .
#if you want to save it you can either persist or use saveAsTable to save.
tweets.createOrReplaceTempView('tweets')
result_pdf = tweets.select("*").toPandas()
print(result_pdf.info())

print(result_pdf.head())

#hourly_tweet = spark.sql('SELECT HOUR(FROM_UNIXTIME(created_at)) AS hour FROM tweets').toPandas().hist()
#name = "Hourly_tweets_01_02_03.png"
#plt.savefig(name, hourly_tweet)
