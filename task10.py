#Task: Frequent Words After Processing

import os, sys, string, re
import pandas as pd
pd.set_option('display.max_colwidth', -1)

from utils import *
import matplotlib # Importing matplotlib for it working on remote server
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.probability import FreqDist
import csv

import re
from collections import Counter
from string import punctuation

from wordcloud import WordCloud
from textblob import TextBlob

import sparknlp
from pyspark.sql import SparkSession
import pyspark.sql.functions as f

spark = SparkSession.builder \
    .master("local[4]")\
    .config("spark.driver.cores", 10)\
    .config('spark.executor.memory', '10g') \
    .config("spark.driver.memory", "20g") \
    .config('spark.driver.maxResultSize', '6g') \
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
folder = "all_days/"
day = "01"
# Read JSON file into dataframe
#result_pdf = spark.read.json([file_name1, file_name2, file_name3])                       #.repartition(4).persist()
result_pdf = spark.read.json([file_name1, file_name2, file_name3])                       #.repartition(4).persist()
result_pdf = result_pdf.select('id','geo','timestamp_ms', 'retweeted', 'text', 'created_at','entities')
print(result_pdf.printSchema())

result_pdf = result_pdf.select("*").toPandas()
print(result_pdf["text"][:10])

print("------- Cleaning tweet's text ----------------")
result_pdf['clean_text'] = result_pdf['text'].apply(processTweet)
print(result_pdf["clean_text"][:10])


# Tokenizing the tweet base texts.
result_pdf['tokenized'] = result_pdf['clean_text'].apply(word_tokenize)

# Lower casing clean text.
result_pdf['lower'] = result_pdf['tokenized'].apply(lambda x: [word.lower() for word in x])

# Removing stopwords.
result_pdf['stopwords_removed'] = result_pdf['lower'].apply(remove_words)

# Applying part of speech tags.
result_pdf['pos_tags'] = result_pdf['stopwords_removed'].apply(nltk.tag.pos_tag)

result_pdf['wordnet_pos'] = result_pdf['pos_tags'].apply(
    lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])

# Applying word lemmatizer.
wnl = WordNetLemmatizer()
result_pdf['lemmatized'] = result_pdf['wordnet_pos'].apply(
    lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
result_pdf['lemmatized'] = result_pdf['lemmatized'].apply(
    lambda x: [word for word in x if word not in stop])
result_pdf['text_final'] = [' '.join(map(str, l)) for l in result_pdf['lemmatized']]

print("----------------Lemmitisazed tweets----------------")
print(result_pdf['text_final'].head())

#result_pdf['tokens'] = result_pdf['clean_text'].apply(text_process) # tokenize style 1
#result_pdf['text_final'] = result_pdf['lemma_str'].apply(remove_words) #tokenize style 2
#result_pdf = result_pdf.drop(['tokens'], axis=1)
#result_pdf['text_final'].head(n=10)

sparkDF = spark.createDataFrame(result_pdf["text_final"])
sparkDF.printSchema()
print("---------------------Most Used Words After Processing----------------------")
count_df = tweets.withColumn('word', f.explode(f.split(f.col('text_final'), ' ')))\
                 .groupBy('word')\
                 .count()\
                 .sort('count', ascending=False)

base_filename = "English_" + folder + "01_02_03" + '_most_words_processed.csv'
count_df.toPandas().to_csv(base_filename)
