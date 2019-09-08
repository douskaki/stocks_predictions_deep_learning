import os

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from embeddings import *
from training import *
from dataloader import *
from config import *


def main():

    data = DataLoader()
    data.load_data()
    data.calculate_prices_periods_differences()
    data.create_prices_with_headlines_list()
    data.get_clean_headlines()

    emb = Embeddings(data.clean_headlines)
    emb.fit_vocabulary()
    emb.load_gloves_embeddings()
    emb.missing_from_gloves()
    emb.word_to_integers()
    emb.add_special_tokens_to_vocab()
    emb.integers_to_word()
    emb.create_words_embeddings_matrix()
    emb.convert_headlines_to_integers()
    emb.get_headlines_lengths()

    pad_headlines = emb.get_pad_headlines()
    norm_price = data.normalize_prices()


    train = Training(len(emb.vocab_to_int), emb.word_embedding_matrix, data.price)
    train.split_data(pad_headlines, norm_price)
    train.get_best_model()


    # deeper = False
    # wider = False
    # learning_rate = 0.001
    # dropout = 0.3

    # filename = 'question_pairs_weights_deeper={}_wider={}_lr={}_dropout={}'.format(
    #     deeper, wider, learning_rate, dropout)
    #
    # print('[' + str(datetime.now()) + '] - ' + "Predictions of model " + filename)
    # model = load_model_from_json(filename)
    # model = train.make_predictions(model, deeper, wider, dropout, learning_rate)

    # Default news that you can use
    # create_news = "Hundreds of millions of Facebook user records were exposed on Amazon cloud server"
    #
    # clean_news = DataLoader.clean_text(create_news)
    # int_news = emb.news_to_int(clean_news)
    # pad_news = emb.padding_news(int_news)
    # pad_news = np.array(pad_news).reshape((1,-1))
    # pred = model.predict([pad_news,pad_news])
    # price_change = unnormalize(pred, max(data.price), min(data.price))
    # print("The stock price should close: {}.".format(np.round(price_change[0][0],2)))


if __name__ == "__main__":
    main()





