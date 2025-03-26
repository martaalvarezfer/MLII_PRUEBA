import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
import random

# --- Variables y configuraciones ficticias ---
# Suponemos que no tenemos los modelos entrenados por el momento.
num_classes = 10
# Lista de nombres de clases ficticias (ajusta según tu problema)
classnames = [f"Clase {i}" for i in range(num_classes)]
Images_size = 224
Images_types = ['jpg', 'jpeg', 'png']
Disp_Models = ["Modelo A", "Modelo B"]  # Opciones dummy
Models_paths = []  # No se usarán ya que no tenemos modelo real
classification_models = ["resnext101_64x4d"]
extra_models = []
threshold = 0.1

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
            ¡Bienvenidos a la aplicación web Canonist.ia de clasificación de imágenes!
            Actualmente, <span style='color:red;'>no disponemos de modelos entrenados</span> para clasificar imágenes.
            La funcionalidad se simulará de manera dummy para mantener la estructura de la aplicación.
        """, unsafe_allow_html=True)

    # Configuraciones en la barra lateral
    with st.sidebar:
        st.header("Configuraciones")
        # Selector de modo de clasificación
        classification_mode = st.radio(
            "Modo de Clasificación:",
            ("Single-class", "Multi-class(Not available yet)"),
            help="Selecciona 'Single-class' para obtener una categoría."
        )
        
        # Selector del modelo (opción dummy)
        model_option = st.selectbox(
            "Modelo a Utilizar:",
            Disp_Models,
            help="Selecciona el modelo de CNN (actualmente dummy)."
        )
        
        st.warning("Actualmente no disponemos de modelos entrenados. La clasificación se simulará.")

    # Carga de la imagen
    with st.container():
        image_file = st.file_uploader("Cargar Imagen", type=Images_types)

    if image_file is not None:
        with st.spinner('Procesando imagen...'):
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
        
        # --- Simulación de la clasificación ---
        # Dado que no tenemos un modelo real, se simula una salida aleatoria
        with st.spinner('Simulando la clasificación...'):
            # Se genera una salida aleatoria (dummy) para num_classes
            dummy_output = torch.rand(1, num_classes)
            # Se obtienen las 2 clases con mayor "confianza" simulada
            top_probs, top_classes = torch.topk(dummy_output, k=2, dim=1)
            
            predicted_label_1 = top_classes[0][0].item()
            predicted_label_2 = top_classes[0][1].item()
            class_name_1 = classnames[predicted_label_1]
            class_name_2 = classnames[predicted_label_2]
            prob_1 = top_probs[0][0].item()
            prob_2 = top_probs[0][1].item()

            diff = prob_1 - prob_2

        # Mostrar resultados en función del modo de clasificación
        if classification_mode == "Single-class":
            st.success(f'### Clase predicha: {class_name_1} (Confianza: {round(prob_1, 5)})')
        else:
            if diff > threshold:
                st.success(f'### Clase predicha: {class_name_1} (Confianza: {round(prob_1, 5)})')
            else:
                st.success(f'### Clase 1: {class_name_1} (Confianza: {round(prob_1, 5)})')
                st.success(f'### Clase 2: {class_name_2} (Confianza: {round(prob_2, 5)})')

        st.image(image, caption='Imagen cargada', use_column_width=False)
    else:
        st.info("Por favor, carga una imagen para simular la clasificación.")

if __name__ == "__main__":
    main()
