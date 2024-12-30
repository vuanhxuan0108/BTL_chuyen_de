import pandas as pd
from pyvi import ViTokenizer
import re

class TextPreprocessor:
    def __init__(self, stopwords_file):
        self.stopword_set = self.load_stopwords(stopwords_file)
    def load_stopwords(self, stopwords_file):
        stopword_df = pd.read_csv(stopwords_file, header=None, names=['stopword'])
        return set(stopword_df['stopword'])
    def remove_stopwords(self, line):
        words = [] 
        for word in line.strip().split(): 
            if word not in self.stopword_set: 
                words.append(word) 
        return ' '.join(words)

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\d{1,2}/\d{1,2}', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = self.remove_stopwords(text)
        text = ViTokenizer.tokenize(str(text))
        return text

    def preprocess_data(self, data):
        data['Content'] = data['Content'].apply(self.preprocess_text)
        return data

