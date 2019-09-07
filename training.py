import numpy as np
from dataloader import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

from keras.utils import plot_model

from keras import initializers
from keras.models import Sequential
from keras.layers import Merge
from keras.layers import Dropout, Embedding, Convolution1D, Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config import *
from utils import *

class Training:

    def __init__(self, nb_words, word_embedding_matrix, prices_list):
        """

        """
        self.max_daily_length = Config.max_daily_length
        self.embedding_dim = Config.embedding_dim
        self.deeper = False
        self.wider = False
        self.dropout = 0.3
        self.learning_rate = 0.001
        self.filter_length1 = 3
        self.filter_length2 = 5
        self.weights = initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=2)
        self.nb_filter = 16
        self.rnn_output_size = 128
        self.hidden_dims = 128
        self.nb_words = nb_words
        self.word_embedding_matrix = word_embedding_matrix
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.prices_list = prices_list
        self.model = None


    def split_data(self, news, prices):
        """
            Splitting original dataset to train and validation
        :return:
        """
        x_train, x_test, y_train, y_test = train_test_split(news, prices, test_size=Config.validation_split, random_state=2)

        self.x_train = np.array(x_train)
        self.x_test = np.array(x_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)

        # Check the lengths
        print('[' + str(datetime.now()) + '] - ' + 'Train dataset length: {}'.format(len(self.x_train)))
        print('[' + str(datetime.now()) + '] - ' + 'Test  dataset length: {}'.format(len(self.x_test)))

        return self.x_train, self.x_test, self.y_train, self.y_test

    def get_best_model(self):

        for deeper in [False]:
            for wider in [False]:
                for learning_rate in [0.001]:
                    for dropout in [0.3]:
        # for deeper in [True, False]:
        #     for wider in [True, False]:
        #         for learning_rate in [0.001]:
        #             for dropout in [0.3, 0.5]:

                        model = self.build_model(deeper, wider, learning_rate, dropout)

                        print()
                        print('[' + str(datetime.now()) + '] - ' +
                              "Current model: Deeper={}, Wider={}, LR={}, Dropout={}".format(
                                  deeper, wider, learning_rate, dropout))
                        print()

                        modelname = 'question_pairs_weights_deeper={}_wider={}_lr={}_dropout={}'.format(
                            deeper, wider, learning_rate, dropout)

                        save_model_to_json(model, modelname)

                        plot_model(model, to_file='./images/' + modelname + '.png', show_shapes=True, show_layer_names=True)

                        save_best_weights = './model/' + modelname + '.h5'

                        callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True),
                                     EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto'),
                                     ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=3)]

                        history = model.fit([self.x_train, self.x_train],
                                            self.y_train,
                                            batch_size=Config.batch_size,
                                            epochs=Config.epochs,
                                            validation_split=Config.validation_split,
                                            verbose=True,
                                            shuffle=True,
                                            callbacks=callbacks)

                        plot_keras_history(history, modelname)

                        self.make_predictions(model, deeper, wider, dropout, learning_rate)

    def build_model(self, deeper, wider, learning_rate, dropout):

        if wider is True:
            self.nb_filter *= 2
            self.rnn_output_size *= 2
            self.hidden_dims *= 2

        model1 = Sequential()

        model1.add(Embedding(self.nb_words,
                             self.embedding_dim,
                             weights=[self.word_embedding_matrix],
                             input_length=self.max_daily_length))
        model1.add(Dropout(dropout))

        model1.add(Convolution1D(filters=self.nb_filter,
                                 kernel_size=self.filter_length1,
                                 padding='same',
                                 activation='relu'))
        model1.add(Dropout(dropout))

        if deeper is True:
            model1.add(Convolution1D(filters=self.nb_filter,
                                     kernel_size=self.filter_length1,
                                     padding='same',
                                     activation='relu'))
            model1.add(Dropout(dropout))

        model1.add(LSTM(self.rnn_output_size,
                        activation=None,
                        kernel_initializer=self.weights,
                        dropout=dropout))

        ####

        model2 = Sequential()

        model2.add(Embedding(self.nb_words,
                             self.embedding_dim,
                             weights=[self.word_embedding_matrix],
                             input_length=self.max_daily_length))
        model2.add(Dropout(dropout))

        model2.add(Convolution1D(filters=self.nb_filter,
                                 kernel_size=self.filter_length2,
                                 padding='same',
                                 activation='relu'))
        model2.add(Dropout(dropout))

        if deeper is True:
            model2.add(Convolution1D(filters=self.nb_filter,
                                     kernel_size=self.filter_length2,
                                     padding='same',
                                     activation='relu'))
            model2.add(Dropout(dropout))

        model2.add(LSTM(self.rnn_output_size,
                        activation=None,
                        kernel_initializer=self.weights,
                        dropout=dropout))

        model = Sequential()

        model.add(Merge([model1, model2], mode='concat'))

        model.add(Dense(self.hidden_dims, kernel_initializer=self.weights))
        model.add(Dropout(dropout))

        if deeper is True:
            model.add(Dense(self.hidden_dims // 2, kernel_initializer=self.weights))
            model.add(Dropout(dropout))

        model.add(Dense(1, kernel_initializer=self.weights, name='output'))

        # we compile the model
        # using mean squared error as a loss function
        # and adam as an optimizer
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate, clipvalue=1.0))

        # we have compiled a model and it is ready to be trained
        print(model.summary())

        return model

    def evaluate(self, model, bs: int = Config.batch_size):
        """

        :return:
        """
        score = model.evaluate(
            self.x_test,  # features
            self.y_test,  # labels
            batch_size=bs,  # batch size
            verbose=1  # the mostX_test extended verbose
        )

        print('[' + str(datetime.now()) + '] - ' + 'Evaluation - Test categorical_crossentropy:', score[0])
        print('[' + str(datetime.now()) + '] - ' + 'Evaluation - Test accuracy:', score[1])

    def make_predictions(self, model, deeper, wider, dropout, learning_rate):
        """

        :return:
        """
        filename = 'question_pairs_weights_deeper={}_wider={}_lr={}_dropout={}'.format(
            deeper, wider, learning_rate, dropout)

        print('[' + str(datetime.now()) + '] - ' + "Predictions of model " + filename)

        #self.evaluate(model)

        predictions = model.predict([self.x_test, self.x_test], verbose=True)

        if np.any(np.isnan(predictions)):
            predictions = np.nan_to_num(predictions)

        mse = mean_squared_error(self.y_test, predictions)
        print('[' + str(datetime.now()) + '] - ' + "Mean Squared Error: ", mse)
        max_price = max(self.prices_list)
        min_price = min(self.prices_list)

        def unnormalize(value, max_value, min_value):
            """Revert values to their unnormalized amounts"""
            return (value * (max_value - min_value) + min_value)

        unnorm_predictions = [unnormalize(pred, max_price, min_price) for pred in predictions]
        unnorm_y_test = [unnormalize(y, max_price, min_price) for y in self.y_test]


        # Calculate the median absolute error for the predictions
        mae = median_absolute_error(unnorm_y_test, unnorm_predictions)
        print('[' + str(datetime.now()) + '] - ' + "Median Absolute Error: ", mae)

        print('[' + str(datetime.now()) + '] - ' + "Summary of actual opening price changes")
        print(pd.DataFrame(unnorm_y_test, columns=[""]).describe())
        print()
        print('[' + str(datetime.now()) + '] - ' + "Summary of predicted opening price changes")
        print(pd.DataFrame(unnorm_predictions, columns=[""]).describe())

        # Plot the predicted (blue) and actual (green) values
        plt.figure(figsize=(12, 4))
        plt.plot(unnorm_predictions, color='blue')
        plt.plot(unnorm_y_test, color='green')
        plt.title("Predicted (blue) vs Actual (green) Opening Price Changes")
        plt.xlabel("Testing instances")
        plt.ylabel("Change in Opening Price")
        #plt.show()
        plt.savefig('./images/' + filename + '_pred.png')

        # Create lists to measure if opening price increased or decreased
        direction_pred = []
        for pred in unnorm_predictions:
            if pred >= 0:
                direction_pred.append(1)
            else:
                direction_pred.append(0)

        direction_test = []
        for value in unnorm_y_test:
            if value >= 0:
                direction_test.append(1)
            else:
                direction_test.append(0)

        # Calculate if the predicted direction matched the actual direction
        direction = accuracy_score(direction_test, direction_pred)
        direction = round(direction, 4) * 100
        print('[' + str(datetime.now()) + '] - ' +
              "Predicted values matched the actual direction {}% of the time.".format(direction))

        return model


