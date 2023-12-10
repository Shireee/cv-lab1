from search_engine import *
import streamlit as st
import matplotlib.image as mpimg
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

#model type
N = 1024 

# Load models
with open(f'./models/kmeans_model{N}.pkl', 'rb') as f:
    K_Means_model = pickle.load(f)

with open(f'./models/NN_model{N}.pkl', 'rb') as f:
    NN_model = pickle.load(f)

df = pd.read_csv(f'./data/database{N}.csv')

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    image = np.array(image)

    # Input the number of similar images to find
    k = st.slider('Number of similar images to find', min_value=1, max_value=10)

    if st.button('Search'):

        # Search 
        similar_images = search_image(image, K_Means_model, NN_model, df, k)

        # Create plot
        fig = plt.figure(figsize=(20, 10))
        for i, path in enumerate(similar_images, 1):  
            # Remove the unwanted directory from the path
            path = path.replace('/webapp', '')

            img = mpimg.imread(path)
            ax = plt.subplot(k+1, 1, i) 
            plt.imshow(img)
            ax.axis('off')  
        st.pyplot(fig)