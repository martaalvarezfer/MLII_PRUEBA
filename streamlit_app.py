import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random

# --- Variables y configuraciones ficticias ---
# Suponemos que no tenemos los modelos entrenados por el momento.
num_classes = 10
# Lista de nombres de clases ficticias (ajusta según tu problema)
classnames = [f"Clase {i}" for i in range(num_classes)]
Images_size = 224
Images_types = ['jpg', 'jpeg', 'png']
Disp_Models = ["Modelo A", "Modelo B"]  # Opciones dummy

# --- Dataset personalizado para la imagen cargada ---
class CustomImageDataset(Dataset):
    def __init__(self, image, transform=None):
        self.image = image
        self.transform = transform

    def __len__(self):
        return 1  # Solo una imagen

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.image)
        else:
            image = self.image
        # Etiqueta dummy (no se utiliza en la simulación)
        label = 0
        return image, label

# --- Función principal de la app ---
def main():
    # Configuración de la página
    st.set_page_config(page_title="ML2 - CNN", layout="centered")
    st.title("Clasificación de Imágenes con CNNs")
    
    # Mensaje de bienvenida y explicación
    with st.container():
        st.markdown("""
            ¡Bienvenido!  
            Actualmente, <span style='color:red;'>estamos en construcción</span> para clasificar imágenes.
            Pasos:
            1. Selecciona el modelo que deseas usar en la parte izquierda  
            2. Sube la imagen  
            3. Verás la predicción para ella
        """, unsafe_allow_html=True)

    # Configuraciones en la barra lateral
    with st.sidebar:
        st.header("Configuraciones")
        
        # Selector del modelo (por ahora solo dummy)
        model_option = st.selectbox(
            "Modelo a Utilizar:",
            Disp_Models,
            help="Selecciona el modelo de CNN (simulación)."
        )

    # Carga de la imagen
    with st.container():
        image_file = st.file_uploader("Cargar Imagen", type=Images_types)

    if image_file is not None:
        with st.spinner('Procesando imagen...'):
            # Cargar la imagen con PIL y asegurar RGB
            image = Image.open(image_file).convert("RGB")
            img_size = Images_size

            # Transformaciones: redimensionar y convertir a tensor
            streamlit_transforms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])

            # Crear un Dataset y DataLoader para la imagen cargada
            streamlit_data = CustomImageDataset(image, transform=streamlit_transforms)
            streamlit_loader = DataLoader(streamlit_data, batch_size=1, shuffle=False)

        # --- Simulación de la clasificación (dummy) ---
        with st.spinner('Simulando la clasificación...'):
            # Genera una salida aleatoria para 'num_classes'
            dummy_output = torch.rand(1, num_classes)
            # Obtén la clase con mayor "confianza" simulada
            _, top_class = torch.max(dummy_output, dim=1)
            
            predicted_label = top_class.item()
            class_name = classnames[predicted_label]
            prob = dummy_output[0][predicted_label].item()

        # Mostrar el resultado (solo top-1)
        st.success(f'### Clase predicha: {class_name} (Confianza simulada: {round(prob, 5)})')

        # Mostrar la imagen cargada
        st.image(image, caption='Imagen cargada', use_column_width=False)
    else:
        st.info("Por favor, carga una imagen para simular la clasificación.")

if __name__ == "__main__":
    main()
