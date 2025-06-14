import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from  sklearn.feature_extraction.text import CountVectorizer

# fetch data from raw

train_data = pd.read_csv('./data/raw/train.csv')
test_data = pd.read_csv('./data/raw/test.csv')


print("____")


## Data transformation

nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text):
    lemmatizer= WordNetLemmatizer()
    text = text.split()
    text=[lemmatizer.lemmatize(y) for y in text]
    return " " .join(text)



def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text = [i for i in str(text).split() if i.lower() not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)

def removing_punctuations(text):
    ## Remove punctuations
    punctuation = r"""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""
    escaped_punct = re.escape(punctuation)
    text = re.sub('[%s]' % escaped_punct, ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df.content=df.content.apply(lambda content : lower_case(content))
    df.content=df.content.apply(lambda content : remove_stop_words(content))
    df.content=df.content.apply(lambda content : removing_numbers(content))
    df.content=df.content.apply(lambda content : removing_punctuations(content))
    df.content=df.content.apply(lambda content : removing_urls(content))
    df.content=df.content.apply(lambda content : lemmatization(content))
    return df

def normalized_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= remove_stop_words(sentence)
    sentence= removing_numbers(sentence)
    sentence= removing_punctuations(sentence)
    sentence= removing_urls(sentence)
    sentence= lemmatization(sentence)
    return sentence

train_processed_data = normalize_text(train_data)
test_processed_data = normalize_text(test_data)

data_path = os.path.join('data','processed')
os.makedirs(data_path)
## store data

train_processed_data.to_csv(os.path.join(data_path,'train_processed_data.csv'))
test_processed_data.to_csv(os.path.join(data_path,'test_processed_data.csv'))
