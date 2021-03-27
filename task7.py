#Task: Sentiment Analysis about a given event

import os, sys, string, re
import pandas as pd
pd.set_option('display.max_colwidth', -1)
from utils import *
import matplotlib # Importing matplotlib for it working on remote server
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType,BooleanType,DoubleType

from operator import add
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
import pyspark.sql.functions as f

spark = SparkSession.builder.master("local[1]").appName("WordCount") \
                            .config("spark.driver.cores", 10)\
                            .config('spark.executor.memory', '4g') \
                            .config("spark.driver.memory", "15g") \
                            .config('spark.driver.maxResultSize', '4g') \
                            .getOrCreate()

from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import missingno as mno
from wordcloud import WordCloud

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

stop = set(stopwords.words('english'))
print("\t Number of stopwords is: ", len(stop))

#from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
#lemmatizer = FrenchLefffLemmatizer()
file_name1 = "English/" + "NoFilterEnglish2020-02-01.json"
file_name2 = "English/" + "NoFilterEnglish2020-02-02.json"
file_name3 = "English/" + "NoFilterEnglish2020-02-03.json"
folder = "all_days/"
day = "01_02_03"
# Read JSON file into dataframe
df = spark.read.json([file_name1, file_name2, file_name3])
#df = spark.read.json(file_name1)
df = df.select('id', 'geo','timestamp_ms', 'retweeted', 'text', 'created_at',
                              'retweet_count','entities')
df.printSchema()
#df.show()

print("Columns of our JSON file are: ", df.columns)

# Create a new DataFrame that contains only tweets which are related to an event
df = df.filter(df.text.like("%Asim%"))
result_pdf = df.select("*").toPandas()

base_filename = folder + "English_" + day + '_AsimEvent_tweets_not_processed.txt'
result_pdf['text'].to_csv(base_filename)


print("Shape of our filtered dataset is: ", result_pdf.shape)
print("----------------- Tweets Samples----------------------------")
print(result_pdf["text"][:10])


print("----------------- Geolocation Samples----------------------------")
print(result_pdf["geo"][:10])

print("------- Cleaning tweet's text ----------------")
result_pdf['clean_text'] = result_pdf['text'].apply(processTweet)
print(result_pdf["clean_text"][:10])

print("------- Cleaning Removing Stop Words ----------------")
# Tokenizing the tweet base texts.
result_pdf['tokenized'] = result_pdf['clean_text'].apply(word_tokenize)
# Lower casing clean text.
result_pdf['lower'] = result_pdf['tokenized'].apply(lambda x: [word.lower() for word in x])
# Removing stopwords.
result_pdf['stopwords_removed'] = result_pdf['lower'].apply(lambda x: [word for word in x if word not in stop])
# Applying part of speech tags.
result_pdf['pos_tags'] = result_pdf['stopwords_removed'].apply(nltk.tag.pos_tag)
result_pdf['wordnet_pos'] = result_pdf['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
# Applying word lemmatizer.
wnl = WordNetLemmatizer()
result_pdf['lemmatized'] = result_pdf['wordnet_pos'].apply(
    lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
result_pdf['lemmatized'] = result_pdf['lemmatized'].apply(
    lambda x: [word for word in x if word not in stop])
result_pdf['clean_text'] = [' '.join(map(str, l)) for l in result_pdf['lemmatized']]

print("------- Categorizing tweet's text ----------------")
result_pdf['category'] = result_pdf['clean_text'].apply(analyze_sentiment)

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
