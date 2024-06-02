import glob
import pickle
import faiss
import random
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import streamlit as st
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Links to files
linkToPreparedData="../basicRunData"
linkToImagesTest="../Test/Level"

# Load the model (only executed once!)
# NOTE: Don't set ttl or max_entries in this case
@st.cache_resource
def load_model():
	  return torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)

backbone = load_model()
modules = list(list(backbone.children())[0].children())

class FeatureExtractor(nn.Module):
    def __init__(self, model_name='vgg19'):
        super().__init__()
        self.backbone = load_model()
        modules1 = list(list(self.backbone.children())[0].children())
        modules2 = list(self.backbone.children())[1]
        modules3 = list(list(self.backbone.children())[2].children())

        self.block = nn.Sequential(
            *modules1,
            modules2,
            nn.Flatten(),
            *modules3[:2]
        )

    def forward(self, inputs):
        return self.block(inputs)
    
feature_extractor = FeatureExtractor()
feature_extractor.eval()

model = ResNet101V2(weights='imagenet', include_top=False, pooling='avg')

def compare_two_images(img1, img2):
    # Extract features
    features1 = extract_features(img1)
    features2 = extract_features(img2)

    distance = np.linalg.norm(features1 - features2)

    return distance

def preprocess_image(img):
    # Load image in grayscale
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize image to fit ResNet input size
    img = cv2.resize(img, (224, 224))

    # Convert image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Convert image to float32 and add batch dimension
    img_array = np.expand_dims(img.astype(np.float32), axis=0)

    # Preprocess input for ResNet
    img_array = preprocess_input(img_array)

    return img_array
def extract_features(img_array):
    # Extract features using the ResNet model
    features = model.predict(img_array)
    return features

with open(os.path.join(linkToPreparedData + 'image_arrays.pkl'), 'rb') as f:
    loaded_image_arrays = pickle.load(f)

# Load saved feature embeddings
feature_embeddings = np.load(os.path.join(linkToPreparedData, 'feature_embeddings.npy'))

with open(os.path.join(linkToPreparedData, 'image_paths.pkl'), 'rb') as f:
    image_paths = pickle.load(f)

# build the index, d=size of vectors
d = feature_embeddings.shape[1]
# index = faiss.IndexFlatL2(d) # with cpu
# index = faiss.GpuIndexFlatL2(d) # with gpu

index = faiss.IndexFlatIP(d) # with cpu
# index = faiss.GpuIndexFlatIP(d) # with gpu

# normalize input vectors
faiss.normalize_L2(feature_embeddings)

# add vectors to the index
index.add(feature_embeddings)

topk = 3
def predict_seam_class(test_img):
    # Convert test_img into embeding
    test_img = np.concatenate([test_img[...,:-1][..., ::-1], test_img[..., -1][..., None]], axis=-1)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
    test_img = cv2.resize(test_img, dsize=(256, 256))
    test_img_tensor = torch.tensor(test_img, dtype=torch.float32)
    test_img_tensor = test_img_tensor / 255.
    test_img_tensor = torch.cat([test_img_tensor[None], test_img_tensor[None], test_img_tensor[None]], dim=0)

    with torch.no_grad():
        test_img_embedding = feature_extractor(test_img_tensor[None])

    test_img_embedding_np = np.array(test_img_embedding)

    # Search for top 3
    D_list, idx_list_for_image = index.search(test_img_embedding_np, topk)
    closest_image_path = None
    scores = defaultdict(float)

    # preprocess the test image
    img1 = preprocess_image(test_img)

    for i in range(len(idx_list_for_image[0])):
        image_path2 = image_paths[idx_list_for_image[0][i]]
        img2 = loaded_image_arrays[idx_list_for_image[0][i]]

        # Compare the images
        distance = compare_two_images(img1, img2)

        # If distance is less than 0.3, get it as the closest image
        if distance < 30:
            path_parts = image_path2.split('/')
            level_name = path_parts[-2]
            closest_image_path = level_name
            break
        else:
        # Else calculate and get the highest score
            score = 100 - distance
            path_parts = image_path2.split('/')
            level_name = path_parts[-2]
            scores[level_name] += score*(topk-i)
    if closest_image_path is None:
        closest_image_path = max(scores, key=scores.get)

    return closest_image_path

def main():
    st.title("Seam Class Prediction")
    
    with st.container():
        on = st.toggle('Camera')
    
    if on:
        uploaded_file = st.camera_input("Take a picture")
        if uploaded_file is not None:
            start_time = time.time()

            # Display the uploaded image
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, 1)
            st.image(image, caption="Seam Image", use_column_width=True)

            # Predict seam class
            result = predict_seam_class(image)
            
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()

            end_time = time.time()
            elapsed_time = end_time - start_time
            st.success(f"The predicted seam class is: {result}")
            st.info(f"Cost time: {elapsed_time:.2f} seconds")

    else:
        uploaded_files = st.file_uploader("Choose an image...", accept_multiple_files=True, type=["jpg", "png", "jpeg"])
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                start_time = time.time()
                # Display the uploaded image
                image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(image, 1)
                st.image(image, caption="Seam Image", use_column_width=True)

                # Predict seam class
                result = predict_seam_class(image)
                
                progress_text = "Operation in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                time.sleep(1)
                my_bar.empty()

                end_time = time.time()
                elapsed_time = end_time - start_time
                st.success(f"The predicted seam class is: {result}")
                st.info(f"Cost time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()