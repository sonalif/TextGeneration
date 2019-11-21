import numpy as np


def open_file(filename, encoding):
    with open('/content/drive/My Drive/datasets/shakespeare.txt', encoding='utf-8') as f:
        story = f.readlines()

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


