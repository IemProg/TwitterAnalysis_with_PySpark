#Task: Detech events in days where we have most the tweets.

import os, sys, string, re
import pandas as pd
pd.set_option('display.max_colwidth', -1)

from utils import *
import matplotlib # Importing matplotlib for it working on remote server
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sparknlp
from pyspark.sql import SparkSession
import pyspark.sql.functions as f

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

from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Converting part of speeches to wordnet format.
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

import missingno as mno
from wordcloud import WordCloud

from textblob import TextBlob

stop = set(stopwords.words('english'))
print("\t Number of stopwords is: ", len(stop))

file_name1 = "English/" + "NoFilterEnglish2020-02-01.json"
file_name2 = "English/" + "NoFilterEnglish2020-02-02.json"
file_name3 = "English/" + "NoFilterEnglish2020-02-03.json"
folder = "day03/"
day = "03"
# Read JSON file into dataframe
result_pdf = spark.read.json(file_name3)                       #.repartition(4).persist()
result_pdf = result_pdf.select('id','geo','timestamp_ms', 'retweeted', 'text', 'created_at',
                              'retweet_count','reply_count')
result_pdf.printSchema()

# Create a new DataFrame that contains young users only
result_pdf = result_pdf.filter( (0 < f.split(f.split(f.col('created_at'), ' ')[3], ":")[0]) \
                            & (f.split(f.split(f.col('created_at'), ' ')[3], ":")[0] <= 3))

result_pdf = result_pdf.select("*").toPandas()
print(result_pdf["text"][:10])

print("------- Cleaning tweet's text ----------------")
result_pdf['clean_text'] = result_pdf['text'].apply(processTweet)
print(result_pdf["clean_text"][:10])

result_pdf['tokens'] = result_pdf['clean_text'].apply(text_process) # tokenize style 1
result_pdf['text_final'] = result_pdf['tokens'].apply(remove_words) #tokenize style 2
result_pdf = result_pdf.drop(['tokens'], axis=1)
result_pdf['text_final'].head(n=10)

print("----------------Lemmitisazed tweets----------------")
print(result_pdf['text_final'].head())

base_filename = folder + "English_" + day + '_tweets_processed.txt'
result_pdf['text_final'].to_csv(base_filename)

sparkDF = spark.createDataFrame(result_pdf)
count_df = sparkDF.withColumn('word', f.explode(f.split(f.col('text_final'), ' ')))\
                  .groupBy('word')\
                  .count()\
                  .sort('count', ascending=False)

base_filename = folder + "English_" + day + '_tweets_event_count_words_processed.txt'
count_df.toPandas().to_csv(base_filename)
