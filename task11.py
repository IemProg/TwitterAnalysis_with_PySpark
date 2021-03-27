#Task: Clustering Tweets

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

spark = SparkSession.builder.master("local[4]").appName("WordCount") \
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

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import  IDF

from pyspark.mllib.linalg import Vector, Vectors
from pyspark.mllib.clustering import LDA, LDAModel
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
#df = spark.read.json([file_name1, file_name2, file_name3])
df = spark.read.json(file_name1)
df = df.select('id', 'retweeted', 'text', 'created_at')
df.printSchema()
#df.show()

print("Columns of our JSON file are: ", df.columns)

# Create a new DataFrame that contains only tweets which are related to an event
#df = df.filter(df.text.like("%Asim%"))
#result_pdf = df.select("*").toPandas()

result_pdf = df.select("*").toPandas()
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
count = CountVectorizer (inputCol="stopwords_removed", outputCol="rawFeatures", vocabSize = 5000, minDF = 10.0)
model = count.fit(result_pdf)
featurizedData = model.transform(result_pdf)
featurizedData.show()

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

num_topics = 4
max_iterations = 100

lda_model = LDA.train(rescaledData[['index','features']].map(list), k = num_topics, maxIterations = max_iterations)

wordNumbers = 7
topicIndices = sc.parallelize(lda_model.describeTopics(maxTermsPerTopic = wordNumbers))
def topic_render(topic):
    terms = topic[0]
    result = []
    for i in range(wordNumbers):
        #term = vocabArray[terms[i]]
        term = terms[i]
        result.append(term)
    return result

topics_final = topicIndices.map(lambda topic: topic_render(topic)).collect()
for topic in range(len(topics_final)):
    print ("Topic" + str(topic) + ":")
    for term in topics_final[topic]:
        print (term)
    print('\n')
