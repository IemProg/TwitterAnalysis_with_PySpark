import re, string
import sys, os
import pandas as pd
from wordcloud import WordCloud
import matplotlib # Importing matplotlib for it working on remote server
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import emoji


def give_emoji_free_text(text):
    return emoji.get_emoji_regexp().sub(r'', text.decode('utf8'))

def clean_texts(df):
    # remove URL
    df['text_proc'] = df['text'].str.replace(r'http(\S)+', r'')
    df['text_proc'].fillna('', inplace=True)
    df['text_proc'] = df['text_proc'].str.replace(r'http ...', r'')
    df['text_proc'] = df['text_proc'].str.replace(r'http', r'')

    # remove RT, @
    df['text_proc'] = df['text_proc'].str.replace(r'(RT|rt)[ ]*@[ ]*[\S]+', r'')
    df['text_proc'] = df['text_proc'].str.replace(r'@[\S]+', r'')

    # remove non-ascii words and characters
    df['text_proc'] = [''.join([i if ord(i) < 128 else '' for i in text]) for text in df['text_proc'].values]
    df['text_proc'] = df['text_proc'].str.replace(r'_[\S]?', r'')

    # remove &, < and >
    df['text_proc'] = df['text_proc'].str.replace(r'&amp;?', r'and')
    df['text_proc'] = df['text_proc'].str.replace(r'&lt;', r'<')
    df['text_proc'] = df['text_proc'].str.replace(r'&gt;', r'>')

    # remove extra space
    df['text_proc'] = df['text_proc'].str.replace(r'[ ]{2, }', r' ')

    # insert space between punctuation marks
    df['text_proc'] = df['text_proc'].str.replace(r'([\w\d]+)([^\w\d ]+)', r'\1 \2')
    df['text_proc'] = df['text_proc'].str.replace(r'([^\w\d ]+)([\w\d]+)', r'\1 \2')

    # lower case and strip white spaces at both ends
    df['text_proc'] = df['text_proc'].str.lower()
    df['text_proc'] = df['text_proc'].str.strip()

# Some basic helper functions to clean text by removing urls, emojis, html tags and punctuations.

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\u200d"
                           u"\u2640-\u2642"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)


def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        max_words=50,
        max_font_size=40,
        scale=5,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(1, figsize=(10,10))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.savefig(title)
    plt.show()
