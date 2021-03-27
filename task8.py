#Task: Sentiment analysis aout a given day

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
    .config("spark.driver.cores", 12)\
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5")\
    .config('spark.executor.memory', '6g') \
    .config("spark.driver.memory", "15g") \
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
folder = "day01/"
day = "01_02_03"
# Read JSON file into dataframe
result_pdf = spark.read.json([file_name1, file_name2, file_name3])                       #.repartition(4).persist()
#result_pdf = spark.read.json(file_name1)                       #.repartition(4).persist()
result_pdf = result_pdf.select('id','geo','timestamp_ms', 'retweeted', 'text', 'created_at','entities')
print(result_pdf.printSchema())


result_pdf = result_pdf.select("*").toPandas()
print(result_pdf["text"][:10])

print("------- Cleaning tweet's text ----------------")
result_pdf['clean_text'] = result_pdf['text'].apply(processTweet)
print(result_pdf["clean_text"][:10])


print("------- Categorizing tweet's text ----------------")
result_pdf['category'] = result_pdf['clean_text'].apply(analyze_sentiment)
result_pdf.head(n=10)


# check the number of positive vs. negative tagged sentences
positives = result_pdf['category'][result_pdf.category == 1]
negatives = result_pdf['category'][result_pdf.category == -1]
neutrals = result_pdf['category'][result_pdf.category == 0]


print("-----------------------------------------")
print('number of positve categorized text is:  {}'.format(len(positives)))
print('number of negative categorized text is: {}'.format(len(negatives)))
print('number of neutral categorized text is: {}'.format(len(neutrals)))
print('total length of the data is:            {}'.format(result_pdf.shape[0]))

slices_len = [len(positives), len(negatives), len(neutrals)]
category = ['positives', 'negatives', 'neutrals']
colors = ['r', 'g', 'b']

plt.pie(slices_len, labels=category, colors=colors, startangle=90, autopct='%.1f%%')
name = folder + "PieFig_for_Day_" + str(day)
plt.savefig(name)
