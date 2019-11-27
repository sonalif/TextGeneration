from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, SimpleRNN


class Model():

    def __init__(self, vocab_len, seq_len, embedding, emb_matrix):
        self.vocab_len = vocab_len
        self.seq_len = seq_len
        self.embedding = embedding
        self.emb_matrix = emb_matrix

    def rnn(self):
        model = Sequential()
        if self.embedding:
            model.add(Embedding(self.vocab_len, 300, weights=[self.emb_matrix], input_length=self.seq_len, trainable=False))
        else:
            model.add(Embedding(self.vocab_len, 256, input_length=self.seq_len))
        model.add(SimpleRNN(128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(64))
        model.add(Dropout(0.2))
        model.add(Dense(self.vocab_len, activation='softmax'))
        return model

    def lstm(self):
        model = Sequential()
        if self.embedding:
            model.add(Embedding(self.vocab_len, 300, weights=[self.emb_matrix], input_length=self.seq_len))
        else:
            model.add(Embedding(self.vocab_len, 256, input_length=self.seq_len))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(128))
        model.add(Dropout(0.2))
        model.add(Dense(self.vocab_len, activation='softmax'))
        return model

    def bidirectional_lstm(self):
        model = Sequential()
        if self.embedding:
            model.add(Embedding(self.vocab_len, 300, weights=[self.emb_matrix], input_length=self.seq_len, trainable=False))
        else:
            model.add(Embedding(self.vocab_len, 256, input_length=self.seq_len))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dense(self.vocab_len, activation='softmax'))
        return model
