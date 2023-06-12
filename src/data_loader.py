import os

import pandas as pd


class DataLoader():

    def __init__(self, data_dir):

        self.root_dir = os.path.dirname(os.getcwd())
        self.data_dir = data_dir
        self.path_to_data = os.path.join(self.root_dir, data_dir)

    def load_product_details(self, products_file):
        self.products_file = products_file
        path_to_products_file = os.path.join(self.path_to_data, products_file)
        if not os.path.exists(path_to_products_file):
            raise ValueError(
                "Product details dataset is not available in the desired location.")
        product_details_df = pd.read_csv(path_to_products_file)
        return product_details_df

    def load_product_embeddings(self, embeddings_file):
        self.embeddings_file = embeddings_file
        path_to_embeddings_file = os.path.join(
            self.path_to_data, embeddings_file)
        if not os.path.exists(path_to_embeddings_file):
            raise ValueError(
                "Product embeddings file is not available in the desired location")
        product_embeddings_df = pd.read_csv(path_to_embeddings_file)
        return product_embeddings_df
