############################################################################## important libraries ##########################################################################
import streamlit as st
from PIL import Image as ima
import os
import numpy as np
import cv2
import pandas as pd
import time

# ADDED: PyTorch imports
import torch
import torch.nn as nn
from torchvision import transforms, models
from openai import OpenAI

st.set_page_config(page_title="Sea Creatures 🌊", page_icon="🌊", layout="wide")

st.markdown("""
<div style="background-color: navy; width: 100%; padding: 30px; border-radius: 10px;">
   <h1 style="color: white; text-align: center;">Sea Creatures 🌊</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; font-family: Arial, sans-serif;">
   <h1 style="font-size: 40px; color: #333;">MARINE LIFE ENCYCLOPEDIA</h1>
   <p style="font-size: 18px; color: #0066cc;">
       Explore the Marine Life Encyclopedia to learn fun facts about marine animals.
   </p>
</div>
""", unsafe_allow_html=True)

##############################################################################################################################################################################################################
####################################################################################     the load_model and predict functions      ###########################################################################

# UPDATED: PyTorch model loading function
def load_model(model_path):
    """Load PyTorch model"""
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate the model architecture (EfficientNet-B0)
    model = models.efficientnet_b0(pretrained=False)
    
    # Get number of input features for the classifier
    num_features = model.classifier[1].in_features
    
    # Recreate the exact same classifier architecture we used during training
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, 20)  # 20 classes for sea creatures
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    return model, device

# UPDATED: PyTorch prediction function
def predict(image_path, model, device, class_names):
    """Predict using PyTorch model"""
    # Define the same transforms used during validation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    # Load and preprocess image using PIL (similar to training)
    from PIL import Image
    img = Image.open(image_path).convert('RGB')
    
    # Apply transforms
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Move to device
    img_tensor = img_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(probabilities, 1)
    
    # Get confidence score
    confidence = probabilities[0][predicted_idx].item()
    
    # Get class name
    predicted_class = class_names[predicted_idx.item()]
    
    return predicted_class, confidence

#################################################################################################################################################################################################
#################################################################################################################################################################################################
#################################################################################################################################################################################################

client = OpenAI()

def animal_info(animal):
    prompt = f"""
Tell me information about this animal in short lines in sperated format as a story.
Write ONLY the information  as a humen tell a excited info but don't be flatter.
Animal: {animal}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )

    return response.choices[0].message.content
#################################################################################################################################################################################################

def animal_quiz():
    prompt = f"""
Tell me a one quesion about any sea animal for ask a kid ,in the first line.
and the second line just write the name of this animal that you asked above and show as a funny 
dont type anything other than these two lines.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )

    return response.choices[0].message.content

#################################################################################################################################################################################################
#################################################################################################################################################################################################

#################################################################################################################################################################################################
with st.sidebar:
    st.title("Navigation")
    selected_page = st.radio("Go to:", ["Marine Life Encyclopedia", "Detection"])

########################################################################################################################################################################################
#                                                                                          Marine Life Encyclopedia
########################################################################################################################################################################################

if selected_page == "Marine Life Encyclopedia":
    
    image_directory = r"D:\college materials\3 semester 1\Deeplearing\section\pytorch\Sea Creatures project\samples"
    image_extensions = [".jpg", ".jpeg", ".png"]

    image_paths = [os.path.join(image_directory, f) for f in os.listdir(image_directory) 
                  
                if os.path.splitext(f)[1].lower() in image_extensions][:23]
    
    for i in range(0, len(image_paths), 4):
        cols = st.columns(4)
        for j, path in enumerate(image_paths[i:i + 4]):
            
            with cols[j]:
                try:
                    image = ima.open(path).resize((300, 300))
                    name = os.path.splitext(os.path.basename(path))[0]
                    st.image(image, use_container_width=True, caption=name)
                    
                    if st.button(f"See More about {name}", key=f"button_{i + j}"):
                        if name == 'Quiz':
                            text = animal_quiz()
                            for idx, line in enumerate(text.split("\n")):
                                if idx==0:
                                    st.markdown(f'<p style="color:black;font-weight:bold;">{line}</p>', 
                                            unsafe_allow_html=True)
                                else:
                                    import streamlit.components.v1 as components
                                    time.sleep(8)
                                    firework_html = """
                                    <div style="position:relative; text-align:center;">
                                        <span style="font-size:2.2em; color:blue; font-weight:bold; vertical-align:middle; text-shadow:0 0 8px #00aaff; text-transform:uppercase; letter-spacing:2px;"><sup>{}</sup></span>
                                    </div>
                                    """.format(line)
                                    st.markdown(firework_html, unsafe_allow_html=True)
                        else:
                            data=animal_info(name)
                            for value in data.split("\n"):
                                st.markdown(f'<p style="color:black;font-weight:bold;">{value}</p>', 
                                          unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error displaying image: {e}")

################################################################################################################################################################################
#                                                                                "Sea Creature Detection" 
################################################################################################################################################################################

elif selected_page == "Detection":
    st.title("Image Detection")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = ima.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # UPDATED: Class names to match our training (20 classes)
        class_names = ['Clams', 'Corals', 'Crab', 'Dolphin', 'Eel', 'Fish', 'Jelly Fish',
                      'Lobster', 'Nudibranchs', 'Octopus', 'Otter', 'Penguin', 'Puffer',
                      'Sea Ray', 'Sea Urchin', 'Sea Horse', 'Seal', 'Shark', 'Shrimp',
                      'Squid', 'Starfish', 'Sea Turtle', 'Whale']  # Note: You have 23 here but training has 20
        
        # IMPORTANT: You need to check which 20 classes you actually trained on
        # Use the class_names.txt file generated during training
        class_names_file = r"D:\college materials\3 semester 1\Deeplearing\section\pytorch\Sea Creatures project\sea_creatures_model_test1\class_names.txt"
        if os.path.exists(class_names_file):
            with open(class_names_file, 'r') as f:
                class_names = [line.strip() for line in f]
        
        # UPDATED: Load PyTorch model
        model_path = r"D:\college materials\3 semester 1\Deeplearing\section\pytorch\Sea Creatures project\sea_creatures_model_test1\best_model_sea_creatures.pth"
        
        if st.button("Predict"):
            try:
                # Load model
                model, device = load_model(model_path)
                
                # Save uploaded file temporarily
                temp_path = "temp_image.jpg"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Predict using PyTorch
                predicted_class, confidence = predict(temp_path, model, device, class_names)
                
                # Display results
                st.markdown(f'<p style="color:black;font-weight:bold;font-size:30px;">Predicted Class: {predicted_class}</p>',
                          unsafe_allow_html=True)
                st.markdown(f'<p style="color:green;font-weight:bold;font-size:20px;">Confidence: {confidence:.2%}</p>',
                          unsafe_allow_html=True)
                
                # Get animal information
                text = animal_info(predicted_class)

                if isinstance(text, str):
                    for line in text.strip().split("\n"):
                        st.markdown(
                            f'<p style="color:black;font-weight:bold;">{line}</p>',
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("No information available")

                # Clean up
                os.remove(temp_path)
                
            except FileNotFoundError:
                st.error(f"Model file not found at: {model_path}")
                st.info("Please make sure you have trained the model and saved it at the correct location.")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.error("Please ensure you have trained the model first by running the training script.")