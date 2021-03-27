#Task: Clustering of tweets

import os, sys, string, re
import pandas as pd
pd.set_option('display.max_colwidth', -1)

from utils import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sparknlp
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
spark = SparkSession.builder \
    .master("local[*]")\
    .config("spark.driver.cores", 10)\
    .config("spark.driver.memory","15G")\
    .config("spark.driver.maxResultSize", "8G") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5")\
    .config("spark.kryoserializer.buffer.max", "4G")\
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
folder = "all_days/"
day = "01"

pipeline = PretrainedPipeline('explain_document_dl', 'en')

# Read JSON file into dataframe
result_pdf = spark.read.json([file_name1, file_name2, file_name3])                       #.repartition(4).persist()
#[file_name1, file_name2, file_name3]
#result_pdf = spark.read.json(file_name1)                       #.repartition(4).persist()
result_pdf = result_pdf.select('id','geo','timestamp_ms', 'retweeted', 'text', 'created_at','entities')
print(result_pdf.printSchema())
#default = result_pdf.select('id','geo','timestamp_ms', 'retweeted', 'text', 'created_at','retweet_count','entities')
"""
default = default.select("*").toPandas()
hashtags = []
for i in default.items():
    print(i['entities']['hashtags']['text'])
    if len(i['entities']['hashtags']['text']) > 1:
        for k in i['entities']['hashtags']['text']:
            hashtags.append(k)
    else:
        hashtags.append(i['entities']['hashtags']['text'][0])

top_hashtags = Counter(hashtags).most_common()[:10]

print(top_hashtags)

namefile = folder + "TopHashtags_day_01_02_03" +".csv"
w = csv.writer(open(namefile, "w"))
for item in top_hashtags:
    w.writerow([item[0], item[0]])
"""
print("--------------------------- Most Used Hashtags ------------------")

result_pdf.createOrReplaceTempView('result_pdf')

query = '''
SELECT result_pdf.entities.hashtags.text as hashtags, COUNT(*) as cnt
FROM result_pdf
GROUP BY hashtags
ORDER BY cnt DESC
'''

"""
query = '''
SELECT result_pdf.entities.hashtags.text AS 'HashTag', COUNT(*) AS 'count'
FROM
(
  SELECT SUBSTRING_INDEX(SUBSTRING_INDEX(t.HashTag, ',', n.n), ',', -1) AS val
  FROM (SELECT Substring(HashTag, 2, LENGTH(HashTag) - 2) AS HashTag FROM tab) AS t
  CROSS JOIN
  (
   SELECT a.N + b.N * 10 + 1 n
     FROM
    (SELECT 0 AS N UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) a
   ,(SELECT 0 AS N UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) b
  ) n
   WHERE n.n <= 1 + (LENGTH(t.HashTag) - LENGTH(REPLACE(t.HashTag, ',', '')))
) sub
WHERE val <> ''
GROUP BY 'HashTag'
ORDER BY 'count' DESC
'''
"""

#print(spark.sql(query).show())

print("--------------------------- Bag of Words ------------------")
sw = stopwords.words("english")
def remove_stopwords(x):
    string = ''
    for x in x.split(' '):
        if x.lower() not in sw:
            string += x + ' '
        else:
            pass
    return string

nosw = udf(remove_stopwords)
spark.udf.register("nosw", nosw)
result_pdf = result_pdf.withColumn('text_nosw',nosw('text'))


def flat_list(column):
    corpus = []
    for row in column:
        for w in row.split(' '):
            corpus.append(w)
    return corpus

def corpus_creator(text_col):
    corpus = text_col.rdd \
                    .flatMap(flat_list) \
                    .map(lambda x: (x, 1)) \
                    .reduceByKey(lambda x, y: x+y ) \
                    .sortBy(lambda x: x[1], ascending=False) \
                    .toDF() \
                    .withColumnRenamed('_1','text') \
                    .withColumnRenamed('_2','count')
    return corpus

def annual_tweets(year):
    annual_tweets = result_pdf.select('text_nosw') \
                              .filter(f.split(f.col('created_at'), ' ')[2]== year) \
                              .withColumnRenamed('text_nosw','text')
    return annual_tweets

def wordcloud(corpus_sdf):
    corpus_pdf = corpus_sdf.limit(500).toPandas()

    corpus_dict = {}
    for index, row in corpus_pdf.iterrows():
        corpus_dict[row['text']] = row['count']

    wordcloud = WordCloud().generate_from_frequencies(corpus_dict)
    plt.imshow(wordcloud);


years_list = [31, 1, 2, 3]

annual_corpora = {}
for year in years_list:
    annual_corpora[str(year)] = corpus_creator(annual_tweets(year))

print("--------------------------- Entity Recognition ------------------")
def make_string(x):
    string = ''
    for x in x:
        string += x + ' '
    return string

make_string = udf(make_string)
spark.udf.register("make_string", make_string)

annual_entities = {}
for year in years_list:
    entities_filtered = pipeline.transform(annual_corpora[str(year)]) \
                                .select('text','count',
                                        col('entities.result').alias('entities'),
                                        col('pos.result').alias('pos'))
    entities_filtered = entities_filtered.withColumn('entities',make_string('entities'))\
                                        .withColumn('pos',make_string('pos'))\
                                        .filter('entities <> ""')
    annual_entities[str(year)] = entities_filtered


namefile = folder + "EntityRecognition_day_01_02_03" +".csv"
w = csv.writer(open(namefile, "w"))
for key, val in annual_entities.items():
    w.writerow([key, val])

print("--------------- Saved file as: ".format(namefile))
