from typing import List, Tuple
from torch import nn
from utils import Vocab
import numpy as np

def get_embeddings(model: nn.Module, mode: str="first"):
    """
    Get the embeddings from the model. Supported modes are:
    `first`: embeddings from the first layer
    `second`: embeddings from the (surprise!) second layer
    `average`: embeddings from both the layers, averaged.
    """
    if mode == "first":
        return model.emb.weight.data.clone().detach().t().cpu().numpy()
    if mode == "second":
        return model.out.weight.data.clone().detach().cpu().numpy()
    if mode == "average":
        return (model.emb.weight.data.clone().detach().t().cpu().numpy() + \
            model.out.weight.data.clone().detach().cpu().numpy()) / 2
    
def get_vector(word: str, vocab: Vocab, embeddings: np.ndarray):
    assert word in vocab.stoi, "Word is not in the vocabulary"
    i = vocab.stoi[word]
    vec = embeddings[i]
    return vec

def most_similar(word: str, vocab: Vocab, embeddings: np.ndarray, n_similar: int) -> List[Tuple[str, int]]:
    assert word in vocab.stoi, "Word is not in the vocabulary"
    i = vocab.stoi[word]
    vec = embeddings[i]
    # cosine similarity measure:
    sims = embeddings @ vec.T \
        / (np.sqrt(np.sum(np.square(embeddings), axis=1)) \
           * np.sqrt(np.sum(np.square(vec)))) 
    i_to_sim = zip(vocab.itos, sims)
    words_by_sim = sorted(i_to_sim, key=lambda x: x[1], reverse=True)
    return list(words_by_sim)[:n_similar]