
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="centered", page_title="Sube tu imagen")
st.title("📤 Sube tu propia imagen")

uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("🔄 Cargando imagen...")

    img = Image.open(uploaded_file)

    # Convertir a escala de grises
    img_gray = img.convert("L")

    # Convertir a array NumPy normalizado
    img_array = np.array(img_gray).astype(np.float32)
    img_array /= 255.0

    st.success("✅ Imagen cargada correctamente.")

    # Mostrar imagen
    fig, ax = plt.subplots()
    plt.imshow(img_array, cmap="gray")
    plt.axis("off")
    st.pyplot(fig)
else:
    st.info("📎 Por favor, sube una imagen en formato JPG o PNG.")
