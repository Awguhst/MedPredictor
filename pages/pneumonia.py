import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd

st.markdown("""
<style>
    /* Main font & colors */
    .main > div {padding-top: 2rem; padding-bottom: 2rem;}
    
    .stApp {
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    
    h1, h2, h3, .stButton {text-align: center;}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="X_Ray", layout="centered")

# Define the same model architecture as before
class CNN(nn.Module):
    def __init__(self, num_classes, dropout_rate):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# Load the model
def load_model(device):
    model = CNN(num_classes=2, dropout_rate=0.15)
    model.load_state_dict(torch.load('./models/model_weights.pth', map_location=device))  # Load saved weights
    model.to(device)  # Move the model to the correct device (GPU/CPU)
    model.eval()  # Set the model to evaluation mode
    return model

# Define the image transformation
def transform_image(image):
    # Convert the image to RGB if it's in grayscale
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    img_transforms = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
    ])
    image = img_transforms(image).unsqueeze(0)  # Add batch dimension
    return image

# Title and description
st.title("Pneumonia Detection")
st.markdown(
    """
    <p style="text-align:center; font-size:1.1rem;">
    Upload an image to predict whether the person has Pneumonia or not.
    </p>
    """,
    unsafe_allow_html=True,
)

# File uploader widget
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    # Get the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model to the appropriate device
    model = load_model(device)
    
    # Process the image
    image_tensor = transform_image(image)  # Apply transformations
    image_tensor = image_tensor.to(device)  # Move the image tensor to the correct device

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        
        # Ensure that the probabilities tensor has the correct shape
        print(probabilities.shape)  # Debug: Check the shape of probabilities tensor
        
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # Display the result in a result card
    class_names = ['Normal', 'Pneumonia']  # Adjust to your class names
    predicted_label = class_names[predicted_class]

    # Accessing probabilities safely
    normal_prob = probabilities[0, 0].item()  
    pneumonia_prob = probabilities[0, 1].item()  

        # Determine the majority class
    if normal_prob > pneumonia_prob:
        majority_class = "Normal"
        majority_prob = normal_prob
        border_color = "#00796b"
        text_color = "#00796b"
    else:
        majority_class = "Pneumonia"
        majority_prob = pneumonia_prob
        border_color = "#d32f2f"
        text_color = "#d32f2f"

    with st.container():
        st.markdown(
            f"""
            <div style="background-color:#171a21; padding:20px; border-radius:10px; box-shadow:0 4px 8px rgba(0,0,0,0.1); border: 2px solid {border_color}; text-align: center;">
                <h4 style="color: {text_color};">Predicted Condition: <strong>{majority_class}</strong></h4>
                <p style="font-size:18px; color: {text_color};">Probability: {majority_prob:.1%}</p>
            </div>
            """, unsafe_allow_html=True
        )


