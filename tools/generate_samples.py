import numpy as np
from keras.preprocessing.sequence import pad_sequences
import tools.sample as sample


def display(story):
    full = ''
    for word in story:
        if word == '|newline|':
            full = full + '\n'
        elif word in ',.;:?!':
            full = full + word
        else:
            full = full + ' ' + word

    print(full)


class SampleGeneration():

    def __init__(self, no_of_words, seq_len, model, tokenizer, temp):
        self.no_of_words = no_of_words
        self.seq_len = seq_len
        self.temp = temp
        self.model = model
        self.tokenizer = tokenizer

    def conditional(self):
        print('\nEnter prompt: \n')
        # Randomly pick a seed sequences
        seed = input()
        sentence = self.tokenizer.texts_to_sequences([seed])[0]

        sentence = list(pad_sequences([sentence], maxlen=self.seq_len, padding='pre')[0])

        while len(sentence) > self.seq_len:
            print("Prompt too long. Enter a prompt of less than %d words:" % self.seq_len)
            seed = input()
        # seed_index = np.random.randint(len(sentences+sentences_test))
        # seed = (sentences+sentences_test)[seed_index]
        gen_story = []

        gen_story.extend(seed.split())
        print('TEMPERATURE:' + str(self.temp) + '\n')

        for i in range(self.no_of_words):
            x_pred = np.expand_dims(sentence, axis=0)

            preds = self.model.predict(x_pred, verbose=0)[0]
            next_indices = sample.sample(preds, self.temp)

            for ix in next_indices:
                if ix == 0:
                    continue
                else:
                    next_word = self.tokenizer.index_word[ix]
                    sentence = sentence[1:]
                    sentence.append(ix)
                    break

            gen_story.append(next_word)

        display(gen_story)