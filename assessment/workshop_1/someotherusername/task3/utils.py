# utils.py

import random 
import re
from collections import Counter
from typing import Iterable, List, Tuple, Any, SupportsFloat
import numpy as np

tokenizer_regex = re.compile(r"\w+")

def preprocess(text: str) -> List[str]:
    return tokenizer_regex.findall(text.lower())

class Vocab:
    """
    A utility to convert tokens to indices and vice versa.
    Notation is loosely based on fast.ai that I spent a lot of time
    studying recently.
    """
    def __init__(self, vocab_size: int, normalization_factor: SupportsFloat=3/4) -> None:
        """
        Initialize a vocab of size `vocab_size`. Actual size will be
        `vocab_size` + 1 to account for the unknown token.
        """
        self.vocab_size = vocab_size
        self.unk_s = "xxunk"
        self.size = vocab_size
        self.stoi = dict() # map string to index
        self.itos = list() # map index to string
        self.freqs = Counter()
        self.normalization = normalization_factor
        self.normalized_counts = list()

    @staticmethod
    def normalize_counts(counts: List[int], constant: SupportsFloat=3/4) -> List:
        """
        Compute the probability distribution with reweighing using the power of `constant`
        """
        counts = np.array(counts)
        raised = np.power(counts, constant)
        denom = np.sum(raised)
        return raised / denom

    def build(self, texts: Iterable[Iterable[str]]) -> None:
        """
        Process an iterable of iterables to create a vocab.
        After counting the tokens, the vocab is trimmed to `vocab_size`
        according to tokens' frequency.
        """
        for text in texts:
            self.freqs.update(text)

        words, _ = zip(*self.freqs.most_common(self.vocab_size))
        words = list(words)
        words.append(self.unk_s)
        self.itos = words
        self.stoi = {s: i for i, s in enumerate(words)}
        if len(self.itos) < self.vocab_size + 1:
            print(f"The actual vocabulary size is {len(self.itos)}.")
        # for negative sampling
        counts = [self.freqs[self.itos[i]] for i in range(len(self.itos))]
        self.normalized_counts = self.normalize_counts(counts, self.normalization)

    def numericalize(self, tokens: Iterable[str]) -> List[int]:
        """
        Convert tokens into ids.
        """
        return [self.stoi[t] if t in self.stoi
                else self.stoi[self.unk_s]
                for t in tokens]

    def textify(self, ids: Iterable[int]) -> List[str]:
        """
        Convert ids back into tokens.
        """
        return [self.itos[i] for i in ids]

    def __repr__(self):
        return f"Vocab of size {len(self.itos)}"


class SkipGramDataGen:
    """
    Generate data for skip-gram algorithm.
    """
    def __init__(self, texts: Iterable[List], window_size: int):
        """
        Initialize a dataset to get SkipGram batches.
        :param texts: an iterable of lists of (maybe numericalized) tokens
        """
        self.texts = texts
        self.ws = window_size

    def iter_line(self) -> Tuple[Any, Any]:
        """
        Draw a pair of (center_word, context_word) from the dataset.
        """
        for text in self.texts:
            # skip the first two words, leave two in the end
            # so that not to overflow indices
            for i in range(self.ws, len(text) - self.ws):
                center_word = text[i]
                context_indices = [i-1-n for n in range(self.ws)] + [i+1+n for n in range(self.ws)]
                context_word = text[random.choice(context_indices)]
                yield (center_word, context_word)

    def iter_batch(self, bs: int) -> Tuple[List, List]:
        batch = list()
        i = 0

        for word_pair in self.iter_line():
            batch.append(word_pair)
            i += 1
            if i == bs:
                yield zip(*batch)
                i = 0
                batch = list()
        if batch:
            yield zip(*batch)


class SkipGramNSDataGen:
    """
    Generate data for skip-gram algorithm with negative sampling.
    """
    def __init__(
            self,
            texts: Iterable[List],
            window_size: int,
            distribution: Iterable[SupportsFloat],
            neg_samples: int=5
        ):
        """
        Initialize a dataset to get SkipGram batches.
        :param texts: an iterable of lists of (maybe numericalized) tokens
        :param window_size: window size to sample context words
        :param distribution: probability distribution for negative sampling
        """
        self.texts = texts
        self.ws = window_size
        self.dist = distribution
        self.ns = neg_samples
        self.indices = list(range(len(distribution)))

    def iter_line(self) -> Tuple[Any, Any, List[Any]]:
        """
        Draw a pair of (center_word, context_word) from the dataset.

        Note that this does not return negative samples! Drawing them once
        for each line is extra slow, you don't want to do it
        """
        for text in self.texts:
            indices = list(range(len(text)))
            np.random.shuffle(indices)
            for i in indices:  # draw a random word
                center_word = text[i]
                context_indices = [max(0, i-1-n) for n in range(self.ws)] + \
                                  [min(i+1+n, len(text)-1) for n in range(self.ws)]
                context_word = text[random.choice(context_indices)]
                yield center_word, context_word

    def iter_batch(self, bs: int) -> Tuple[List, List]:
        batch = list()
        i = 0

        for word_pair in self.iter_line():
            batch.append(word_pair)
            i += 1
            if i == bs:
                neg_samples = np.random.choice(self.indices, self.ns*bs, p=self.dist)
                neg_samples = neg_samples.reshape(bs, self.ns)
                centers, contexts = zip(*batch)
                contandnegs = np.c_[contexts, neg_samples]
                yield centers, contandnegs
                i = 0
                batch = list()
        if batch:
            neg_samples = np.random.choice(self.indices, self.ns*len(batch), p=self.dist)
            neg_samples = neg_samples.reshape(len(batch), self.ns)
            centers, contexts = zip(*batch)
            contandnegs = np.c_[contexts, neg_samples]
            yield centers, contandnegs
