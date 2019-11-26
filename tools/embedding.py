from tqdm import tqdm
import numpy as np



def emd_vector(path):
    embedding_vector = {}
    f = open(path)
    for line in tqdm(f):
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:], dtype='float32')
        embedding_vector[word] = coef

    return embedding_vector


def emd_matrix(vocab, vocab_len, emb_vec):
    embedding_matrix = np.zeros((vocab_len, 300))
    for word, i in tqdm(vocab.items()):
        embedding_value = emb_vec.get(word)
        if embedding_value is not None:
            embedding_matrix[i] = embedding_value

    return embedding_matrix