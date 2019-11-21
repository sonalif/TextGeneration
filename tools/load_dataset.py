import numpy as np
import os
from glob import glob


def open_file(path, encoding):

    if os.path.isfile(path):
        if path.endswith('.txt'):
            with open(path, encoding=encoding) as f:
                story = f.readlines()

    else:
        if os.path.isdir(path):
            files = []
            for (dirpath, _, fnames) in os.walk(path):
                for fname in fnames:
                    files.append(os.path.join(dirpath, fname))

        elif path.endswith('*'):
            files = glob(path)

        ##HANDLE MULTIPLE FILES

    return story


def make_tokens(story):
    story_in_words = []
    for i, line in enumerate(story):
        story[i] = line.lower().replace('.', ' . ').replace(',', ' , ').replace('?', ' ? ').replace('"', ' " ').replace(
            '!', ' ! ').replace(':', ' : ').replace(';', ' ; ').replace('--', ' ').replace('-', ' ').replace(',', ' , ')
        story_in_words.extend(story[i].split(' '))

    return story_in_words


def make_sentences(story_tokens, seq_len, step=1):

    sentences = []
    for i in range(0, len(story_tokens) - seq_len, step):
        sentences.append(story_tokens[i: i + seq_len + 1])
    #np.asarray(sentences).shape
    return sentences


