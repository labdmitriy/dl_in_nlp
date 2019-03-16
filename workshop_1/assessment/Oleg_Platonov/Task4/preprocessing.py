import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from string import ascii_lowercase


class SkipGramBatcher():
    def __init__(self):
        self.data = []
    
    def shuffle(self):
        np.random.shuffle(self.data)
    
    def fit(self, text, window_size, shuffle=True):
        for i in range(len(text)):
            window_start = max(i - window_size, 0)
            window_end = min(i + window_size, len(text) - 1)
            for j in range(window_start, window_end + 1):
                if j != i:
                    self.data.append([text[i], text[j]])
        
        if shuffle:
            self.shuffle()
    
    def generate_batches(self, batch_size):
        inputs, targets = [], []
        for i, word_pair in enumerate(self.data, 1):
            inputs.append(word_pair[0])
            targets.append(word_pair[1])
            
            if i % batch_size == 0:
                yield inputs, targets
                inputs, targets = [], []
    
        if len(inputs) > 0:
            yield inputs, targets


class BatchTransposeTrickBatcher():
    def __init__(self):
        self.data = []
    
    def shuffle(self):
        np.random.shuffle(self.data)
    
    def fit(self, text, window_size, shuffle=True):
        for i in range(window_size, len(text) - window_size):
            self.data.append([text[j] for j in range(i - window_size, i + window_size + 1)])
        
        if shuffle:
            self.shuffle()
    
    def generate_batches(self, batch_size):
        word_windows = []
        for i, word_window in enumerate(self.data, 1):
            word_windows.append(word_window)
            
            if i % batch_size == 0:
                yield word_windows
                word_windows = []
    
        if len(word_windows) > 0:
            yield word_windows


def prepare_data(min_count, batcher=SkipGramBatcher, shuffle=True): 
    with open('text8', 'r') as file:
        text = file.readlines()[0].split(' ')
    
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words.union(set([letter for letter in ascii_lowercase]))
    stop_words = stop_words.union(set(['', 'th']))
    
    text = [word for word in text if word not in stop_words]
    counter = Counter()
    counter.update(text)
    frequent_words = dict((key, value) for key, value in counter.items() if value >= min_count)
    
    token2id = dict((key, i) for i, key in enumerate(frequent_words.keys()))
    id2token = dict((i, key) for key, i in token2id.items())
    
    text = [word for word in text if word in token2id]
    text_tokenized = [token2id[word] for word in text]
    
    batcher = batcher()
    batcher.fit(text_tokenized, window_size=3, shuffle=shuffle)
    
    return frequent_words, token2id, id2token, batcher

