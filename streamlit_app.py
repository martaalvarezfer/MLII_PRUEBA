import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

# ========================================
# App Layout
# ========================================
st.set_page_config(layout="centered", page_title="Sube tu imagen")
st.title("📤 Sube tu propia imagen")

# ========================================
# Subida de imagen
# ========================================
uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("🔄 Cargando imagen...")

    # Abrimos la imagen
    img = Image.open(uploaded_file)

    # Convertimos a escala de grises
    img_gray = transforms.Grayscale()(img)

    # A tensor y normalización
    img_tensor = transforms.ToTensor()(img_gray)
    img_tensor = (img_tensor - torch.min(img_tensor)) / (torch.max(img_tensor) - torch.min(img_tensor))

    st.success("✅ Imagen cargada correctamente.")

    # Mostrar imagen
    fig, ax = plt.subplots()
    plt.imshow(img_tensor.squeeze(), cmap="gray")
    plt.axis("off")
    st.pyplot(fig)
else:
    st.info("📎 Por favor, sube una imagen en formato JPG o PNG.")

