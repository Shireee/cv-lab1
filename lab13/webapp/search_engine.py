import cv2
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# Extract SIFT features from an image
def extract_sift_features(img):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return descriptors

# Get descriptors from all images in the path
def get_descriptors(path):
    descriptors_list = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            descriptors = extract_sift_features(img)
            if descriptors is not None:
                descriptors_list.extend(descriptors)
    return descriptors_list


# Image to vector transformation
def image_to_vector(image, model):
    descriptors = extract_sift_features(image)
    if descriptors is not None:
        descriptors = descriptors.astype('float64')  # convert 
        predict_kmeans = model.predict(descriptors)
        histogram = np.bincount(predict_kmeans, minlength=model.n_clusters)
        return normalize(histogram.reshape(1, -1))
    else:
        return None

def search_image(image, K_Means_model, NN_model, df, k):
    # Convert image to vector
    vector = image_to_vector(image, K_Means_model)
    
    # Reshape vector to 2D array
    vector = vector.reshape(1, -1)
    
    # Find the k-nearest vectors
    distances, indices = NN_model.kneighbors(vector, n_neighbors=k)
    
    # Get the paths of the k-nearest images
    paths = df.loc[indices[0], 'path']
    
    return paths.tolist()
