import numpy as np
import os
from glob import glob

FLAG = 0


def open_file(path, encoding):
    global FLAG
    if os.path.isfile(path):
        if path.endswith('.txt'):
            with open(path, encoding=encoding) as f:
                stories = f.readlines()

    else:
        stories = []
        FLAG = 1
        if os.path.isdir(path):
            files = []
            for (dirpath, _, fnames) in os.walk(path):
                for fname in fnames:
                    files.append(os.path.join(dirpath, fname))

        else:  ## assume glob
            files = glob(path)
            print('glob')

        for file in files:
            with open(file, encoding=encoding) as f:
                story = f.readlines()
                stories.append(story[4:6]+story[10:-26])
    return stories


def make_tokens(stories):
    story_in_words = []

    if FLAG == 0:
        for i, line in enumerate(stories):
            stories[i] = line.lower().replace('.', ' . ').replace(',', ' , ').replace('?', ' ? ').replace('"', ' " ').replace('!', ' ! ').replace(':', ' : ').replace(';', ' ; ').replace('--', ' ').replace('-', ' ').replace(',', ' , ')
            story_in_words.extend(stories[i].split(' '))

    else:
        print('okay')
        for story in stories:
            temp = []
            for i, line in enumerate(story):
                story[i] = line.lower().replace('.', ' . ').replace('\n', ' \n ').replace(',', ' , ').replace('?', ' ? ').replace('"',' " ').replace(
                    '!', ' ! ').replace(':', ' : ').replace(';', ' ; ').replace('--', ' ').replace('-', ' ').replace(',',' , ')
                temp.extend(story[i].split(' '))
            temp.append('|endofstory|')
            story_in_words.append(list(filter(None, temp)))
    return story_in_words


def make_sentences(story_tokens, seq_len, step=1):
    sentences = []
    if FLAG == 0:
        for i in range(0, len(story_tokens) - seq_len, step):
            sentences.append(story_tokens[i: i + seq_len + 1])
        #np.asarray(sentences).shape

    else:
        print('okay')
        for tokens in story_tokens:
            for i in range(0, len(tokens) - seq_len, step):
                sentences.append(tokens[i: i + seq_len + 1])
    print(sentences[0])
    return sentences


