import numpy as np
import os
from glob import glob

end_flag = 0


def open_file(path, encoding):

    global end_flag

    if os.path.isfile(path):
        if path.endswith('.txt'):
            with open(path, encoding=encoding) as f:
                stories = f.readlines()

    else:
        stories = []
        end_flag = 1
        if os.path.isdir(path):
            files = []
            for (dirpath, _, fnames) in os.walk(path):
                for fname in fnames:
                    files.append(os.path.join(dirpath, fname))

        else:  ## assume glob
            files = glob(path)

        for file in files:
            with open(file, encoding=encoding) as f:
                story = f.readlines()
                stories.append(story)

    return stories


def make_tokens(stories):
    story_in_words = []

    if end_flag == 0:
        for i, line in enumerate(stories):
            stories[i] = line.lower().replace('.', ' . ').replace(',', ' , ').replace('?', ' ? ').replace('"', ' " ').replace(
                '!', ' ! ').replace(':', ' : ').replace(';', ' ; ').replace('--', ' ').replace('-', ' ').replace(',', ' , ')
            story_in_words.extend(stories[i].split(' '))

    else:
        for story in stories:
            temp = []
            for i, line in enumerate(story):
                story[i] = line.lower().replace('.', ' . ').replace(',', ' , ').replace('?', ' ? ').replace('"',' " ').replace(
                    '!', ' ! ').replace(':', ' : ').replace(';', ' ; ').replace('--', ' ').replace('-', ' ').replace(',',' , ')
                temp.extend(story[i].split(' '))
            temp.append('|endofstory|')
            story_in_words.append(temp)

    return story_in_words


def make_sentences(story_tokens, seq_len, step=1):

    sentences = []
    if end_flag == 0:
        for i in range(0, len(story_tokens) - seq_len, step):
            sentences.append(story_tokens[i: i + seq_len + 1])
        #np.asarray(sentences).shape

    else:
        for tokens in story_tokens:
            for i in range(0, len(tokens) - seq_len, step):
                sentences.append(story_tokens[i: i + seq_len + 1])

    return sentences


