#Task: Save most used words and hashtags for each day seperately

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


#Load / Cache Data
#Spark dataframe should split into partitions = 2-3x the no. threads available in your CPU or cluster. I have 2 cores,
#with 2 threads each = 4, and I chose 3x, ie. 12 partitions, based on experimentation.
#Then cache tables: you can see in Spark GUI that 12 partitions are cached for each file.
#The Shuffle Read is default to 200, we don't want this to be the bottleneck, so we set this equal to partitions in our data,
#using spark.sql.shuffle.partitions. This is specific to wide shuffle transformations (e.g. GROUP BY or ORDER BY)
#that may be performed later on, and how many partitions this operation sets up to read the data.

file_name1 = "NoFilterEnglish2020-02-01"
file = "English/" + file_name1 + ".json"
tweets = spark.read.json(file).repartition(4).persist()
tweets = tweets.select('id','geo','timestamp_ms', 'retweeted', 'text', 'created_at',
                              'retweet_count','reply_count')

#CreateOrReplaceTempView will create a temporary view of the table on memory it is not presistant at this moment
#but you can run sql query on top of that .
#if you want to save it you can either persist or use saveAsTable to save.

print("---------------------Most Used Words Before Processing----------------------")
count_rdd = tweets.select("text").rdd.flatMap(lambda x: x[0].split(' ')) \
              .map(lambda x: (x, 1)).reduceByKey(lambda x,y: x+y)

result_pdf = count_rdd.select("*").toPandas()
base_filename = "English_" + "01" + '_tweets_text_not_processed.txt'
with open(base_filename,'w') as outfile:
    result_pdf['lemma_str'].to_string(outfile, header=False, index=False)
print("Saved file as: ", base_filename)

tweets.createOrReplaceTempView('tweets')
result_pdf = tweets.select("*").toPandas()
print(result_pdf.info())

print("-----------------------Top Hashtags--------------------------")
query = '''
SELECT tweets.entities.user_mentions.name AS mentions, COUNT(*) as cnt
FROM tweets
GROUP BY mentions
ORDER BY cnt DESC
'''
print(spark.sql(query).show())
