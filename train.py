from keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.utils import shuffle
from tools.generate_samples import SampleGeneration
from tools import load_dataset
from tensorflow.keras.utils import to_categorical

from model.model import Model

import numpy as np
import argparse


parser = argparse.ArgumentParser(
    description='Hyper parameters for text Generation with LSTM',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, required=True, help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')
parser.add_argument('--model_name', metavar='MODEL', type=str, default='LSTM', help='RNN, LSTM or BiLSTM')
parser.add_argument('--seq_len', metavar='SEQ_LEN', type=int, default=10, help='Sequence length for training data')
parser.add_argument('--sample_len', metavar='CHARS', type=int, default=500, help='Number of words to sample for generated story')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--encoding', type=str, default='utf-8', help='Set the encoding for reading and writing files.')
parser.add_argument('--temperature', type=float, default=0.5, help='Sampling temperature')


parser.add_argument('--test_only', type=bool, default=False, help='Number of words in the generated story')
parser.add_argument('--attention', metavar='ATTN', type=bool, default=False, help='Use attention mechanism')

INDEX = 0


def generator(sent, word, seq_len, vocab_len, batch_size):

    global INDEX
    while True:
        x = np.zeros((batch_size, seq_len), dtype=np.int)
        y = np.zeros((batch_size, vocab_len), dtype=np.bool)

        for i in range(batch_size):
            x[i] = sent[INDEX % len(sent)]
            y[i] = to_categorical(word[INDEX % len(word)], num_classes=vocab_len)
            INDEX = INDEX + 1
        yield x, y


def main():
    args = parser.parse_args()
    story = load_dataset.open_file(args.dataset, args.encoding)
    story_tokens = load_dataset.make_tokens(story)

    sentences = load_dataset.make_sentences(story_tokens, args.seq_len)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    sequences = np.asarray(sequences)
    vocab = tokenizer.word_counts

    VOCAB_LEN = len(tokenizer.word_counts) + 1
    # print(tokenizer.index_word[95])
    total = sequences.shape   ##CHANGE TO NUMPY

    sequences = shuffle(sequences)

    train_input = sequences[51500:, :-1]
    train_output = sequences[51500:, -1]

    test_input = sequences[:51500, :-1]
    test_output = sequences[:51500, -1]

    print(np.asarray(train_input).shape)
    print(np.asarray(train_output).shape)

    print(np.asarray(test_input).shape)
    print(np.asarray(test_output).shape)

    if args.model_name == 'LSTM':
        model = Model(VOCAB_LEN, args.seq_len).lstm()
    elif args.model == 'RNN':
        model = Model(VOCAB_LEN, args.seq_len).rnn()
    else:
        model = Model(VOCAB_LEN, args.seq_len).bidirectional_lstm()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    CHECK_DIR = "./checkpoints/Shakespeare-epoch{epoch:03d}-words%d-sequence%d-batchsize%d-" \
                "loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}.hdf5" % \
                (VOCAB_LEN, args.seq_len, args.batch_size)

    checkpoint = ModelCheckpoint(CHECK_DIR, monitor='val_acc', save_best_only=True)
    # print_callback = LambdaCallback(on_train_end=on_train_end)
    early_stopping = EarlyStopping(monitor='val_acc', patience=20)
    callbacks_list = [checkpoint, early_stopping]

    if args.test_only is not True:
        model.fit_generator(generator(train_input, train_output, args.seq_len, VOCAB_LEN, args.batch_size),
                            steps_per_epoch=int(len(train_input) / args.batch_size) + 1,
                            epochs=100,
                            callbacks=callbacks_list,
                            validation_data=generator(test_input, test_output, args.seq_len, VOCAB_LEN, args.batch_size),
                            validation_steps=int(len(test_input) / args.batch_size) + 1)

    sample_gen = SampleGeneration(args.sample_len, args.seq_len, model, tokenizer, args.temperature)
    sample_gen.conditional()
