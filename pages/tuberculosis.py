import torch
import torchvision
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
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
import os

st.markdown("""
<style>
    /* Main font & colors */
    .main > div {padding-top: 2rem; padding-bottom: 2rem;}
    
    .stApp {
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    
        /* Buttons */
    .stButton>button {
        background: linear-gradient(to right, #2a56a4, #3d6de7);
        color: white;
        border: none;
        height: 64px !important;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        margin-left: -6px; 
        margin-top: 4.8px; 
        transition: all 0.3s;
        box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

    div[data-baseweb="select"] > div {
        border-radius: 12px;
        border: 1.5px solid #2f3136;
        background: #11141a;
        min-height: 72px;
        padding: 1rem 1.5rem;
        margin-top: -40px;
    }

    div[data-baseweb="select"]:focus-within > div {
        border: 1px solid #3b82f6;    
        box-shadow: 0 0 0 1px rgba(59,130,246,.2);
    }

    div[data-baseweb="select"] .css-1n76uvr {
        color: inherit;
    }
    
    /* Hide ONLY the file info part (name + size), not the whole uploader */
    [data-testid="stFileUploader"] section + div {
        display: none !important;
    }
    /* Optional: also hide the drop label when a file is selected */
    [data-testid="stFileUploaderDropzone"] div small {
        opacity: 0.5;
    }
    
        details[class*="st-emotion-cache"] summary[class*="st-emotion-cache"] {
        background-color: #202229;
    }
    
    details[class*="st-emotion-cache"][open] summary[class*="st-emotion-cache"] {
        background-color: #262730;
    }
    
    button[data-testid="stTab"]
    div[data-testid="stMarkdownContainer"]
    p {
        font-family: 'Roboto', 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    h1, h2, h3, .stButton {text-align: center;}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="X_Ray_TB", layout="centered")

# Define the CNN model architecture for TB classification
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

# Load the model for TB detection
def load_model(device):
    model = CNN(num_classes=2, dropout_rate=0.15)
    model.load_state_dict(torch.load('./models/tuberculosis_weights.pth', map_location=device))  # Load TB detection weights
    model.to(device)  # Move the model to the correct device (GPU/CPU)
    model.eval()  # Set the model to evaluation mode
    return model

# Define the image transformation for input images
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

# Title and description for the app
st.title("Tuberculosis Detection")
st.markdown(
    """
    <p style="text-align:center; font-size:1.1rem;">
    Upload an X-ray image to predict whether the person has Tuberculosis or not.
    </p>
    """,
    unsafe_allow_html=True,
)

# Path to example image for testing
EXAMPLES_FOLDER = "examples"  
EXAMPLE_IMAGE_PATH = os.path.join(EXAMPLES_FOLDER, "Tuberculosis.png")  

# Ensure the folder and file exist
if not os.path.exists(EXAMPLE_IMAGE_PATH):
    st.error(f"Example image not found at {EXAMPLE_IMAGE_PATH}")
    st.stop()

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

with col2:
    st.markdown("<br>", unsafe_allow_html=True) 
    if st.button("Try Example", use_container_width=True, type="primary"):
        uploaded_file = open(EXAMPLE_IMAGE_PATH, "rb")  

if uploaded_file is not None:
    if isinstance(uploaded_file, str):  
        image = Image.open(uploaded_file)
    else:
        image = Image.open(uploaded_file)

    st.image(image, use_container_width=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)

    image_tensor = transform_image(image).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()

    class_names = ['Normal', 'Tuberculosis']
    normal_prob = probabilities[0].item()
    tb_prob = probabilities[1].item()

    majority_class = "Normal" if normal_prob > tb_prob else "Tuberculosis"
    majority_prob = max(normal_prob, tb_prob)
    border_color = "#00796b" if majority_class == "Normal" else "#d32f2f"
    text_color = border_color

    st.markdown(
        f"""
        <div style="background-color:#171a21;padding:20px;border-radius:10px;
                    border:2px solid {border_color};text-align:center;margin:20px 0;">
            <h3 style="color:{text_color};margin:0;">Predicted: <strong>{majority_class}</strong></h3>
            <p style="color:{text_color};font-size:18px;margin:8px 0 0;">
                Probability: {majority_prob:.1%}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    
    # Grad-CAM in Expander
    with st.expander("View Model Explanation (Grad-CAM Heatmap)", expanded=False):
        st.markdown("> **Heatmap shows regions the model focused on for its prediction.**")

        # Target the last convolutional layer
        target_layer = model.features[13]

        # Create GradCAM 
        cam = GradCAM(model=model, target_layers=[target_layer])

        # Generate heatmap for the predicted class
        targets = [ClassifierOutputTarget(predicted_class)]
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]

        # Prepare original image safely
        resized_image = image.resize((200, 200))
        rgb_img = np.array(resized_image)

        # Handle grayscale or RGBA
        if len(rgb_img.shape) == 2:  
            rgb_img = np.stack((rgb_img,) * 3, axis=-1)
        elif rgb_img.shape[2] == 4:  
            rgb_img = rgb_img[:, :, :3]

        rgb_img = rgb_img / 255.0 

        # Create overlay
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Display side by side
        col_left, col_right = st.columns(2)
        with col_left:
            st.image(resized_image, caption="Original Image", use_container_width=True)
        with col_right:
            st.image(visualization, caption=f"Grad-CAM Heatmap",
                     use_container_width=True)
else:
    st.info("Upload an image or click **Try Example** to see the model in action!")


