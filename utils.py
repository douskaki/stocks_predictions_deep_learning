import os
from keras.models import model_from_json
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as pdr
from datetime import datetime


def save_model_to_json(model, model_name: str):
    """

    :param model_name:
    :return:
    """

    print('[' + str(datetime.now()) + '] - ' + 'Saving Model as {}'.format(model_name))

    json_path = os.path.join(os.path.abspath("./model/"), "{}.json".format(model_name))

    model_json = model.to_json()

    with open(json_path, "w") as json_file:
        json_file.write(model_json)

    # self.model.save_weights(weights_path)
    print('[' + str(datetime.now()) + '] - ' + "Saved model to disk on path {}".format(json_path))


def load_model_from_json(model_name: str):
    """

    :param model_name:
    :return:
    """
    print('Loading Model: {}'.format(model_name))

    json_path = os.path.join(os.path.abspath("./model/"), "{}.json".format(model_name))

    weights_path = os.path.join(os.path.abspath("./model/"), "{}.h5".format(model_name))

    with open(json_path, "r") as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    return model

def plot_keras_history(history, modelname):
    """
        Lab02 - keras_reuters_feed_forward_network
    :param history:
    :return:
    """
    # the history object gives the metrics keys.
    # we will store the metrics keys that are from the training sesion.
    metrics_names = [key for key in history.history.keys() if not key.startswith('val_')]

    for i, metric in enumerate(metrics_names):

        # getting the training values
        metric_train_values = history.history.get(metric, [])

        # getting the validation values
        metric_val_values = history.history.get("val_{}".format(metric), [])

        # As loss always exists as a metric we use it to find the
        epochs = range(1, len(metric_train_values) + 1)

        # leaving extra spaces to allign with the validation text
        training_text = "   Training {}: {:.5f}".format(metric, metric_train_values[-1])

        # metric
        plt.figure(i, figsize=(12, 6))

        plt.plot(epochs, metric_train_values, 'b', label=training_text)

        # if we validation metric exists, then plot that as well
        if metric_val_values:
            validation_text = "Validation {}: {:.5f}".format(metric, metric_val_values[-1])

            plt.plot(epochs, metric_val_values, 'g', label=validation_text)

        # add title, xlabel, ylabe, and legend
        plt.title(modelname + '\n' + 'Model Metric: {}'.format(metric))
        plt.xlabel('Epochs')
        plt.ylabel(metric.title())
        plt.legend()
        # plt.show()
        plt.savefig('./images/' + modelname + '_' + metric + '.png')


def normalize(value, max_value, min_value):
    return ((value - min_value) / (max_value - min_value))


def unnormalize(value, max_value, min_value):
    """Revert values to their unnormalized amounts"""
    return (value * (max_value - min_value) + min_value)
