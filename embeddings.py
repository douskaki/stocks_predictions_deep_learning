import numpy as np
import pandas as pd
from datetime import datetime
from config import *


class Embeddings:

    def __init__(self, headlines):
        """ """
        self.headlines = headlines
        self.gloves_embs_path = './glove/glove.840B.300d.txt'
        self.embeddings_index = {}
        self.missing_words = 0
        self.threshold = 10
        self.max_headline_length = Config.max_headline_length
        self.max_daily_length = Config.max_daily_length
        self.embedding_dim = Config.embedding_dim
        self.word_counts = {}
        self.vocab_to_int = {}
        self.int_to_vocab = {}
        self.word_embedding_matrix = None
        self.int_headlines = []
        self.pad_headlines = []

    def fit_vocabulary(self):
        """
            Find the number of times each word was used and the size of the vocabulary
        :param clean_headlines:
        :return:
        """
        word_counts = {}
        for date in self.headlines:
            for headline in date:
                for word in headline.split():
                    if word not in word_counts:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1
        self.word_counts = word_counts
        print('[' + str(datetime.now()) + '] - ' + "Size of Vocabulary:", len(self.word_counts))

    def load_gloves_embeddings(self):
        """
            Loading GloVe's embeddings from file
        :return:
        """
        embeddings_index = {}
        with open(self.gloves_embs_path, encoding='utf-8') as f:
            for line in f:
                values = line.split(' ')
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = embedding
        self.embeddings_index = embeddings_index
        print('[' + str(datetime.now()) + '] - ' + 'Word embeddings: {}'.format(len(embeddings_index)))

    def missing_from_gloves(self):
        """
            Find the number of words that are missing from GloVe, and are used more than our threshold.
        :return:
        """
        missing_words = 0

        for word, count in self.word_counts.items():
            if count > self.threshold:
                if word not in self.embeddings_index:
                    missing_words += 1
        self.missing_words = missing_words
        missing_ratio = round(self.missing_words / len(self.word_counts), 4) * 100

        print('[' + str(datetime.now()) + '] - ' + "Number of words missing from GloVe: {}".format(self.missing_words))
        print('[' + str(datetime.now()) + '] - ' + "Percent of words that are missing from vocabulary: {}%".format(missing_ratio))

    def word_to_integers(self):
        """
            Dictionary to convert words to integers
        :return:
        """
        vocab_to_int = {}

        value = 0
        for word, count in self.word_counts.items():
            if count >= self.threshold or word in self.embeddings_index:
                vocab_to_int[word] = value
                value += 1
        self.vocab_to_int = vocab_to_int
        print('[' + str(datetime.now()) + '] - ' + "Number of Words we will use: {}".format(len(self.vocab_to_int)))


    def add_special_tokens_to_vocab(self):
        """
            Special tokens that will be added to our vocab
        :return:
        """
        codes = ["<UNK>", "<PAD>"]

        # Add codes to vocab
        for code in codes:
            self.vocab_to_int[code] = len(self.vocab_to_int)

    def integers_to_word(self):
        """
            Dictionary to convert integers to words
        :return:
        """
        int_to_vocab = {}

        for word, value in self.vocab_to_int.items():
            int_to_vocab[value] = word

        usage_ratio = round(len(self.vocab_to_int) / len(self.word_counts), 4) * 100
        self.int_to_vocab = int_to_vocab

        print('[' + str(datetime.now()) + '] - ' + "Total Number of Unique Words: {}".format(len(self.word_counts)))
        print('[' + str(datetime.now()) + '] - ' + "Percent of Words we will use: {}%".format(usage_ratio))

    def create_words_embeddings_matrix(self):
        """
            Create matrix with default values of zero
        :return:
        """

        nb_words = len(self.vocab_to_int)
        word_embedding_matrix = np.zeros((nb_words, self.embedding_dim))

        for word, i in self.vocab_to_int.items():
            if word in self.embeddings_index:
                word_embedding_matrix[i] = self.embeddings_index[word]
            else:
                # If word not in GloVe, create a random embedding for it
                new_embedding = np.array(np.random.uniform(-1.0, 1.0, self.embedding_dim))
                self.embeddings_index[word] = new_embedding
                word_embedding_matrix[i] = new_embedding
        self.word_embedding_matrix = word_embedding_matrix

        # Check if value matches len(vocab_to_int)
        print('[' + str(datetime.now()) + '] - ' + "Vocabulary to Integers Length: {}".format(len(self.vocab_to_int)))
        print('[' + str(datetime.now()) + '] - ' + "Word Embeddings Matrix Length: {}".format(len(self.word_embedding_matrix)))


    def convert_headlines_to_integers(self):
        """
            Change the text from words to integers
            If word is not in vocab, replace it with <UNK> (unknown)
        :param clean_headlines:
        :return:
        """
        word_count = 0
        unk_count = 0

        int_headlines = []

        for date in self.headlines:
            int_daily_headlines = []
            for headline in date:
                int_headline = []
                for word in headline.split():
                    word_count += 1
                    if word in self.vocab_to_int:
                        int_headline.append(self.vocab_to_int[word])
                    else:
                        int_headline.append(self.vocab_to_int["<UNK>"])
                        unk_count += 1
                int_daily_headlines.append(int_headline)
            int_headlines.append(int_daily_headlines)

        self.int_headlines = int_headlines
        unk_percent = round(unk_count / word_count, 4) * 100

        print('[' + str(datetime.now()) + '] - ' + "Total number of words in headlines: {}".format(word_count))
        print('[' + str(datetime.now()) + '] - ' + "Total number of UNKs in headlines:  {}".format(unk_count))
        print('[' + str(datetime.now()) + '] - ' + "Percent of words that are UNK:      {}%".format(unk_percent))

    def get_headlines_lengths(self):
        # Find the length of headlines
        lengths = []
        for date in self.int_headlines:
            lengths = [len(headline) for headline in date]

        # Create a dataframe so that the values can be inspected
        lengths = pd.DataFrame(lengths, columns=['counts'])
        print(lengths.describe())

    def get_pad_headlines(self):
        # Limit the length of a day's news to 200 words, and the length of any headline to 16 words.
        # These values are chosen to not have an excessively long training time and
        # balance the number of headlines used and the number of words from each headline.

        pad_headlines = []

        for date in self.int_headlines:
            pad_daily_headlines = []
            for headline in date:
                # Add headline if it is less than max length
                if len(headline) <= self.max_headline_length:
                    for word in headline:
                        pad_daily_headlines.append(word)
                # Limit headline if it is more than max length
                else:
                    headline = headline[:self.max_headline_length]
                    for word in headline:
                        pad_daily_headlines.append(word)

            # Pad daily_headlines if they are less than max length
            if len(pad_daily_headlines) < self.max_daily_length:
                for i in range(self.max_daily_length - len(pad_daily_headlines)):
                    pad = self.vocab_to_int["<PAD>"]
                    pad_daily_headlines.append(pad)
            # Limit daily_headlines if they are more than max length
            else:
                pad_daily_headlines = pad_daily_headlines[:self.max_daily_length]
            pad_headlines.append(pad_daily_headlines)

        self.pad_headlines = pad_headlines
        return pad_headlines

    def news_to_int(self, news):
        """
            Convert your created news into integers
        :param news:
        :return:
        """

        ints = []
        for word in news.split():
            if word in self.vocab_to_int:
                ints.append(self.vocab_to_int[word])
            else:
                ints.append(self.vocab_to_int['<UNK>'])
        return ints

    def padding_news(self, news):
        """
            Adjusts the length of your created news to fit the model's input values.
        :param news:
        :return:
        """
        padded_news = news
        if len(padded_news) < self.max_daily_length:
            for i in range(self.max_daily_length - len(padded_news)):
                padded_news.append(self.vocab_to_int["<PAD>"])
        elif len(padded_news) > self.max_daily_length:
            padded_news = padded_news[:self.max_daily_length]
        return padded_news