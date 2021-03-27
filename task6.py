#Task: Mornign versus Evening words and events

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
folder = "all_days/"
day = "03"
# Read JSON file into dataframe
#result_pdf = spark.read.json([file_name1, file_name2, file_name3])                       #.repartition(4).persist()
result_pdf = spark.read.json(file_name3)                       #.repartition(4).persist()
result_pdf = result_pdf.select('id','geo','timestamp_ms', 'retweeted', 'text', 'created_at',
                              'retweet_count','reply_count')
result_pdf.printSchema()

result_pdf = result_pdf.select("*").toPandas()
print("---------------------10 First Tweets Text ----------------------")
print(result_pdf["text"].head(n=10))

# Applying helper functions
result_pdf['text'] = result_pdf['text'].apply(lambda x: remove_URL(x))
result_pdf['text'] = result_pdf['text'].apply(lambda x: remove_emoji(x))
#train_clean['text'] = train_clean['text'].apply(lambda x: give_emoji_free_text(x))
result_pdf['text'] = result_pdf['text'].apply(lambda x: remove_html(x))
result_pdf['text'] = result_pdf['text'].apply(lambda x: remove_punct(x))

# Tokenizing the tweet base texts.
result_pdf['tokenized'] = result_pdf['text'].apply(word_tokenize)

# Lower casing clean text.
result_pdf['lower'] = result_pdf['tokenized'].apply(lambda x: [word.lower() for word in x])

# Removing stopwords.
result_pdf['stopwords_removed'] = result_pdf['lower'].apply(lambda x: [word for word in x if word not in stop])

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
result_pdf['lemma_str'] = [' '.join(map(str, l)) for l in result_pdf['lemmatized']]

print("----------------Lemmitisazed tweets----------------")
print(result_pdf['lemma_str'].head(n=10))

#--------------------------Morning-------------------------------------------------------
sparkDF = spark.createDataFrame(result_pdf)
morning_pdf = sparkDF.filter( (00 <= f.split(f.split(f.col('created_at'), ' ')[3], ":")[0]) \
                            & (f.split(f.split(f.col('created_at'), ' ')[3], ":")[0] <= 14))

count_df = morning_pdf.withColumn('word', f.explode(f.split(f.col('lemma_str'), ' ')))\
                      .groupBy('word')\
                      .count()\
                      .sort('count', ascending=False)

base_filename = folder + "English_" + day + '_morning_tweets_processed.csv'
morning_pdf.toPandas()["lemma_str"].to_csv(base_filename, header=False, index=False)

base_filename = folder + "English_" + day + '_tweets_morning_count_words_processed.csv'
count_df.toPandas().to_csv(base_filename, header=False, index=False)
print("------------Done with: Morning Scanning from 00h-14h--------------------------")
#--------------------------Evening-------------------------------------------------------
sparkDF = spark.createDataFrame(result_pdf)
evening_pdf = sparkDF.filter( (15 <= f.split(f.split(f.col('created_at'), ' ')[3], ":")[0]) \
                            & (f.split(f.split(f.col('created_at'), ' ')[3], ":")[0] <= 23))

count_df = evening_pdf.withColumn('word', f.explode(f.split(f.col('lemma_str'), ' ')))\
                      .groupBy('word')\
                      .count()\
                      .sort('count', ascending=False)
base_filename = folder + "English_" + day + '_evening_tweets_processed.csv'
evening_pdf.toPandas()["lemma_str"].to_csv(base_filename, header=False, index=False)

base_filename = folder + "English_" + day + '_tweets_evening_count_words_processed.csv'
count_df.toPandas().to_csv(base_filename, header=False, index=False)
print("------------Done with: Evening Scanning from 15h-00h-------------------------")
