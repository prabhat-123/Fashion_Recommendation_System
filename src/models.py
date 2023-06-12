import os

import json
import pickle
from tensorflow.keras.models import model_from_json


class Model():

    def __init__(self, model_dir):
        self.root_dir = os.path.dirname(os.getcwd())
        self.model_dir = model_dir
        self.path_to_model = os.path.join(self.root_dir, model_dir)

    def load_extractor_model(self, weights_file, json_file):
        self.json_file = json_file
        self.weights_file = weights_file
        read_json = open(os.path.join(self.path_to_model, json_file), 'r')
        loaded_model_json = read_json.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(os.path.join(self.path_to_model, weights_file))
        return model

    def load_pickle_model(self, pickle_file):
        self.pickle_file = pickle_file
        with open(os.path.join(self.path_to_model, pickle_file), 'rb') as file:
            pickle_obj = pickle.load(file)
            return pickle_obj
