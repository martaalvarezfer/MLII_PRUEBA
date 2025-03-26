import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F


# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path(__file__).resolve().parent.parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))

from src.utils.data_loader import num_classes, classnames
from src.utils.cnn import load_model_weights, CNN
from src.utils.local_functs import CustomImageDataset
from config.variables import Images_size, Images_types, Disp_Models, Models_paths, classification_models, extra_models, threshold


def main():
    # Configuración de la página
    st.set_page_config(page_title="ML2 - CNN", layout="centered")
    st.title("Clasificación de Imágenes con CNNs")
    
    # Mensaje de bienvenida
    with st.container():
        st.markdown("""
            ¡Bienvenidos a la aplicación web Canonist.ia de clasificación de imágenes del grupo compuesto por Alberto, Jorge, Nacho y Juan!
            Esta aplicación utiliza redes neuronales convolucionales (CNNs) para clasificar imágenes. Por favor, selecciona el modo de clasificación en la barra lateral y carga una imagen.
        """)

    # Configuraciones de la barra lateral
    with st.sidebar:
        st.header("Configuraciones")
        # Selector de modo de clasificación
        classification_mode = st.radio(
            "Modo de Clasificación:",
            ("Single-class", "Multi-class(Not available yet)"),
            help="Selecciona 'Single-class' si deseas que la imagen se clasifique en una sola categoría. Elige 'Multi-class' para obtener múltiples posibles categorías."
        )
        
        # Selector del modelo
        model_option = st.selectbox(
            "Modelo a Utilizar:",
            Disp_Models,
            help="Selecciona el modelo de CNN que deseas usar para clasificar tu imagen."
        )

        model_path = Models_paths[Disp_Models.index(model_option)]

        used_classes = num_classes
        
        # Cargar el modelo
        device = torch.device('cpu')
        model_weights = load_model_weights(model_path)#, map_location=device)
        model_name = model_path.split('\\')[-1].split("-")[0]


        # Change the model name according to the model used

        if model_name not in classification_models:
            print(f"Model {model_name} not found")
            print("Available models are:")
            print(classification_models.extend(extra_models))
            sys.exit()
        else:
            model_used = torchvision.models.__dict__[model_name](weights='DEFAULT')
        
        model = CNN(model_used, used_classes)
        
        model.load_state_dict(model_weights)
    
    # Carga de imagen y selección de modelo
    with st.container():
        image_file = st.file_uploader("Cargar Imagen", type=Images_types)
    
    if image_file is not None:
        with st.spinner('Procesando imagen...'):
            image = Image.open(image_file)

            img_size = Images_size

            streamlit_transforms = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.Grayscale(num_output_channels=3),  # Convertir a RGB si es necesario
                    transforms.ToTensor() 
                ])


            # Crea una instancia del Dataset personalizado
            streamlit_data = CustomImageDataset(image, transform=streamlit_transforms)

        # Crea un DataLoader con el Dataset
        streamlit_loader = DataLoader(streamlit_data, batch_size=1, shuffle=False)
        
        model.eval()
        for images, labels in streamlit_loader:
            output = model(images)
            top_probs, top_classes = torch.topk(output, k=2, dim=1)
            
        # print(top_probs)

        predicted_label_1 = top_classes[0][0].item()
        predicted_label_2 = top_classes[0][1].item()
        class_name_1 = classnames[predicted_label_1]
        class_name_2 = classnames[predicted_label_2]
        prob_1 = top_probs[0][0].item()
        prob_2 = top_probs[0][1].item()

        class_name_1 = classnames[predicted_label_1]
        class_name_2 = classnames[predicted_label_2]

        diff = prob_1 - prob_2

        if classification_mode == "Single-class":
            st.success(f'### Clase predicha: {class_name_1} (Confianza: {round(prob_1, 5)})')
        else:
            if diff > threshold:
                st.success(f'### Clase predicha: {class_name_1} (Confianza: {round(prob_1, 5)})')
            else:
                st.success(f'### Clase 1: {class_name_1} (Confianza: {round(prob_1, 5)})')
                st.success(f'### Clase 2: {class_name_2} (Confianza: {round(prob_2, 5)})')

        st.image(image_file, caption='Imagen cargada', use_column_width=False)
        

if __name__ == "__main__":
    main()
