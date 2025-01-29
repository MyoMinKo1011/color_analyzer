import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


def get_dominant_colors(image, k=5):
  
    image = image.convert("RGB")
    image = image.resize((image.width // 10, image.height // 10))  
    
 
    image_data = np.array(image)
    
 
    image_data = image_data.reshape((-1, 3))
    
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(image_data)
    
 
    colors = kmeans.cluster_centers_.astype(int)
    
    return colors


st.title("Color Analyzer")


uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:

    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    dominant_colors = get_dominant_colors(image)

    st.subheader("Dominant Colors (Hex Code)")
    color_palette = np.zeros((100, 100 * len(dominant_colors), 3), dtype=int)
    for i, color in enumerate(dominant_colors):
        color_palette[:, i*100:(i+1)*100] = color
    
    st.image(color_palette, use_container_width=False)
    

    for i, color in enumerate(dominant_colors):
        hex_code = rgb_to_hex(color)
        st.write(f"Color {i+1} | Hex: {hex_code}")
