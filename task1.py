import os, sys, string, re
import pandas as pd
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

stop = set(stopwords.words('english'))
print("\t Number of stopwords is: ", len(stop))

#from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
#lemmatizer = FrenchLefffLemmatizer()
file_name1 = "English/" + "NoFilterEnglish2020-02-01.json"
file_name2 = "English/" + "NoFilterEnglish2020-02-02.json"
file_name3 = "English/" + "NoFilterEnglish2020-02-03.json"

# Read JSON file into dataframe
df = spark.read.json([file_name1, file_name2, file_name3])
df = df.select('id','geo','timestamp_ms', 'retweeted', 'text', 'created_at',
                              'retweet_count','reply_count')
df.printSchema()
#df.show()

#print(dir(df))
print("Columns of our JSON file are: ", df.columns)


print("---------------------Most Used Words Before Processing----------------------")
count_df = df.withColumn('word', f.explode(f.split(f.col('text'), ' ')))\
    .groupBy('word')\
    .count()\
    .sort('count', ascending=False)

count_words = count_df.select("*").toPandas()
base_filename = "MostUsedWords_English_" + "01_02_03" + '_tweets_text_not_processed.txt'
with open(base_filename,'w') as outfile:
    count_words.to_string(outfile, header=False, index=False)
print("Saved file as: ", base_filename)

print("---------------------Summary of Our DataFrame ----------------------")
result_pdf = df.select("*").toPandas()
result_pdf.info()

print("---------------------10 First Tweets Text ----------------------")
print(result_pdf["text"][:10])

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
print(result_pdf['lemma_str'].head())

base_filename = "English_" + "01_02_03" + '_tweets_text_processed.txt'
with open(base_filename,'w') as outfile:
    result_pdf['lemma_str'].to_string(outfile, header=False, index=False)

print("File saved as: ", base_filename)
#Neatly allocate all columns and rows to a .txt file

#Save WordCloud for tweets text
#show_wordcloud(train_clean['lemma_str'], title = 'Prevalent words in tweets')

"""
# We create a new column to push our arrays of words
tokenizer = Tokenizer(inputCol="text", outputCol="words_token")
tokenized = tokenizer.transform(df).select('id','words_token')

print('############ Tokenized data extract:')
tokenized.show()


# Once in arrays, we can use the Apache Spark function StopWordsRemover
# A new column "words_clean" is here as an output
remover = StopWordsRemover(inputCol='words_token', outputCol='words_clean')
data_clean = remover.transform(tokenized).select('id', 'words_clean')

print('############ Data Cleaning extract:')
data_clean.show()

# Final step : like in the beginning, we can group again words and sort them by the most used
result = data_clean.withColumn('word', f.explode(f.col('words_clean'))) \
  .groupBy('word') \
  .count().sort('count', ascending=False) \

print('############ TOP20 Most used words:')
result.show()
"""

# Stop Spark Process
spark.stop()
