import os

import cv2
import json
import pickle
import operator
import numpy as np
from PIL import Image
import streamlit as st
from models import Model
from data_loader import DataLoader


root_dir = os.path.dirname(os.getcwd())
image_dir = 'images'
model_dir = 'models'
data_dir = 'data'

json_file = 'xception_model.json'
weights_file = 'xception_model.h5'
scaler_file = 'scaler.pickle'
pca_file = 'pca.pickle'
product_embeddings_file = 'product_embeddings.csv'
product_details_file = 'product_details.csv'
image_shape = (299, 299)

model_obj = Model(model_dir)
model = model_obj.load_extractor_model(weights_file, json_file)
scaler = model_obj.load_pickle_model(scaler_file)
pca = model_obj.load_pickle_model(pca_file)

data_loader = DataLoader(data_dir)
product_details_df = data_loader.load_product_details(product_details_file)
product_embeddings_df = data_loader.load_product_embeddings(
    product_embeddings_file)


def calculate_embeddings(img, input_shape, model):

    img_arr = cv2.imread(img)
    resized_img_arr = cv2.resize(img_arr, input_shape)
    model_input = np.expand_dims(resized_img_arr, axis=0)
    embeddings = model.predict(model_input)
    return embeddings


def calculate_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def filter_product_ids(df, query_id):

    reference_categories = df.loc[df['id'] == query_id, [
        "articleType", "gender"]]
    reference_article_type = reference_categories["articleType"].values[0]
    reference_gender = reference_categories["gender"].values[0]
    searchable_products = df.loc[df['gender'] == reference_gender]
    searchable_product_ids = list(
        searchable_products.loc[searchable_products['articleType'] == reference_article_type, 'id'])
    searchable_product_ids = [
        prod_id for prod_id in searchable_product_ids if prod_id != query_id]
    return searchable_product_ids


def recommended_product(df, searchable_product_ids, reduced_query_data):

    searchable_df = df.loc[df['image_id'].isin(
        searchable_product_ids)].reset_index(drop=True)
    similarity_scores = {}
    for item in searchable_df.values:
        img_id = int(item[0])
        vector_embedding = item[1:]
        sim_score = calculate_similarity(reduced_query_data, vector_embedding)
        similarity_scores[img_id] = sim_score[0]
    sorted_similarity_scores = dict(
        sorted(similarity_scores.items(), key=operator.itemgetter(1), reverse=True))
    top_five_similarity_scores = list(sorted_similarity_scores.values())[:5]
    top_five_recommended_product_ids = list(
        sorted_similarity_scores.keys())[:5]
    return {"recommended_product_ids": top_five_recommended_product_ids, "confidence_scores": top_five_similarity_scores}


st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Fashion Product Recommendation")
st.header("Recommending Similar Fashion Products")
st.text("Search your favourite fashion product image")
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    query_img_path = os.path.join(root_dir, image_dir, uploaded_file.name)
    st.write("Query image is {}".format(query_img_path))
    query_id = int(uploaded_file.name.split('.')[0])
    st.image(query_img_path, caption='Uploaded Image.', width=200)
    st.write("Just a second...")
    st.write("See our recommendations")
    query_embedding = calculate_embeddings(query_img_path, image_shape, model)
    scaled_query_data = scaler.transform(query_embedding)
    reduced_query_data = pca.transform(scaled_query_data)
    searchable_product_ids = filter_product_ids(product_details_df, query_id)
    recommended_products = recommended_product(
        product_embeddings_df, searchable_product_ids, reduced_query_data)
    recommended_product_ids = recommended_products["recommended_product_ids"]
    recommended_product_confidence = recommended_products["confidence_scores"]
    recommended_images = [os.path.join(
        root_dir, image_dir, str(product_id) + '.jpg') for product_id in recommended_product_ids]
    st.image(recommended_images, width=100,
             caption=recommended_product_confidence)
    # for i in range(len(recommended_product_ids)):
    #     st.write("Recommended product Id is {} and confidence is {}".format(
    #         recommended_product_ids[i], recommended_product_confidence[i]))
