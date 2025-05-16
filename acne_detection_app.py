import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import requests
import json



# Define the acne types (25 classes)
ACNE_TYPES = [
    "blackhead", "whitehead", "papule", "pustule", "nodule", 
    "cyst", "milia", "comedonal", "hormonal", "cystic", 
    "inflammatory", "noninflammatory", "mild", "moderate", "severe",
    "fungal", "rosacea", "perioral", "steroid", "excoriated",
    "acne_vulgaris", "acne_conglobata", "acne_fulminans", "pomade_acne", "mechanical_acne"
]

# Path configurations
MODEL_DIR = "runs/detect/train6/weights"
DETECTION_MODEL_PATH = f"{MODEL_DIR}/best.pt"
CLASSIFICATION_MODEL_PATH = "runs/classify/train/weights/best.pt"

# LLM API Configuration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")  # Get from environment variable
USE_OFFLINE_MODE = not bool(TOGETHER_API_KEY)  # Only use offline mode if no API key is available

# Set page configuration 
st.set_page_config(
    page_title="Acne Analysis",
    page_icon="🔬",
    layout="wide"
)

# Apply minimal CSS
st.markdown("""
<style>
    /* Remove Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load the trained YOLOv8 detection model
@st.cache_resource
def load_detection_model():
    return YOLO(DETECTION_MODEL_PATH)

# Define classification model
class AcneClassifier(nn.Module):
    def __init__(self, num_classes=25):
        super(AcneClassifier, self).__init__()
        # Load pretrained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# Function to load classification model
@st.cache_resource
def load_classification_model(model_path=None):
    model = AcneClassifier(num_classes=len(ACNE_TYPES))
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Loaded classification model from {model_path}")
    else:
        print("Using untrained classification model")
    model.eval()
    return model

# Function to classify acne type from image
def classify_acne(image, model):
    if not has_classifier:
        return "Unknown", 0.0
    
    # Convert to PIL Image if it's a numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Apply transformation
    img_tensor = transform(image).unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    acne_type = ACNE_TYPES[predicted.item()]
    conf_score = confidence[predicted.item()].item()
    
    return acne_type, conf_score

# Function to blur face while preserving acne areas
def blur_face_except_acne(image, boxes, padding=10):
    # Create a copy of the image
    blurred = image.copy()
    
    # Apply heavy Gaussian blur to the entire image
    blurred = cv2.GaussianBlur(blurred, (51, 51), 30)
    
    # Create a mask for areas to keep unblurred (acne regions with padding)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # For each detected acne box, add to the mask
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        # Add padding but ensure within image bounds
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        # Add to mask
        mask[y1:y2, x1:x2] = 255
    
    # Create the inverse mask
    inv_mask = cv2.bitwise_not(mask)
    
    # Extract the regions of interest from the original image
    fg = cv2.bitwise_and(image, image, mask=mask)
    bg = cv2.bitwise_and(blurred, blurred, mask=inv_mask)
    
    # Combine the foreground and background
    result = cv2.add(fg, bg)
    
    return result

def get_llm_analysis(acne_type):
    try:
        # Format the acne type for better readability
        formatted_acne_type = acne_type.replace('_', ' ').title()
        
        # Try Together.ai API
        if TOGETHER_API_KEY:
            try:
                response = query_together_ai(formatted_acne_type)
                if response:
                    return response
                else:
                    st.error("Could not get response from LLM API. Please try again.")
                    return {
                        "reasoning": "Error: Could not get LLM analysis. Please try again.",
                        "recommendations": "• Please try again\n• Check your internet connection\n• Contact support if the issue persists"
                    }
            except Exception as e:
                st.error(f"API Error: {str(e)}")
                return {
                    "reasoning": "Error: Could not connect to LLM API. Please try again.",
                    "recommendations": "• Please try again\n• Check your internet connection\n• Contact support if the issue persists"
                }
        else:
            st.error("No API key configured. Please add your API key to use this feature.")
            return {
                "reasoning": "Error: No API key configured.",
                "recommendations": "• Please configure your API key to use this feature"
            }
            
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        return {
            "reasoning": "Error: An unexpected error occurred.",
            "recommendations": "• Please try again\n• Contact support if the issue persists"
        }

def query_together_ai(acne_type):
    """Query Together.ai API for acne analysis"""
    endpoint = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOGETHER_API_KEY}"
    }
    
    prompt = f"""As a dermatology expert, provide a detailed analysis of {acne_type} acne:

1. Explain in 2-3 sentences why this type of acne occurs (underlying causes, triggers, etc.)
2. Provide 5 specific, bullet-pointed recommendations for treatment and management

Format your response as a JSON with two fields: "reasoning" and "recommendations"
where "reasoning" is a paragraph and "recommendations" is a string with bullet points using "•" symbols."""
    
    data = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
            {
                "role": "system",
                "content": "You are a dermatology expert assistant that provides concise, accurate information about acne conditions. Always format your response as valid JSON."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    response = requests.post(endpoint, headers=headers, json=data, timeout=15)
    if response.status_code == 200:
        response_data = response.json()
        llm_response = response_data['choices'][0]['message']['content']
        
        try:
            return json.loads(llm_response)
        except:
            return parse_unstructured_response(llm_response)
    
    return None

def query_anthropic(acne_type):
    """Query Anthropic API for acne analysis (backup service)"""
    endpoint = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        # "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    
    prompt = f"""As a dermatology expert, provide a detailed analysis of {acne_type} acne:

1. Explain in 2-3 sentences why this type of acne occurs (underlying causes, triggers, etc.)
2. Provide 5 specific, bullet-pointed recommendations for treatment and management

Format your response as a JSON with two fields: "reasoning" and "recommendations"."""
    
    data = {
        "model": "claude-3-sonnet-20240229",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 500
    }
    
    response = requests.post(endpoint, headers=headers, json=data, timeout=15)
    if response.status_code == 200:
        response_data = response.json()
        llm_response = response_data['content'][0]['text']
        
        try:
            return json.loads(llm_response)
        except:
            return parse_unstructured_response(llm_response)
    
    return None

def parse_unstructured_response(text):
    """Parse unstructured LLM response into required format"""
    reasoning = ""
    recommendations = ""
    
    # Try to extract reasoning and recommendations
    if "reasoning" in text.lower():
        reasoning_start = text.lower().find("reasoning")
        reasoning_end = text.lower().find("recommendations")
        if reasoning_end > reasoning_start:
            reasoning = text[reasoning_start+10:reasoning_end].strip()
    
    if "recommendations" in text.lower():
        recommendations_start = text.lower().find("recommendations")
        recommendations = text[recommendations_start+15:].strip()
    
    if reasoning and recommendations:
        return {
            "reasoning": reasoning,
            "recommendations": recommendations
        }
    
    return None

# Try to load models
try:
    detection_model = load_detection_model()
    classification_model = load_classification_model(CLASSIFICATION_MODEL_PATH)
    has_classifier = True
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    detection_model = None
    classification_model = None
    has_classifier = False

# Image transformation for classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'blurred_image' not in st.session_state:
    st.session_state.blurred_image = None
if 'acne_classifications' not in st.session_state:
    st.session_state.acne_classifications = []
if 'acne_counts' not in st.session_state:
    st.session_state.acne_counts = {}
if 'primary_acne_type' not in st.session_state:
    st.session_state.primary_acne_type = None
if 'llm_analysis' not in st.session_state:
    st.session_state.llm_analysis = None

# Main acne analysis page
st.title("Acne Analysis")
st.write("Upload a clear photo of your face to get an accurate skin analysis")

# File upload area
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Process the uploaded image
if uploaded_file is not None:
    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display the uploaded image
    st.subheader("Your Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    # Create analysis button with columns for centering
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Skin")
    
    if analyze_button:
        # Check if models are loaded
        if not detection_model or not has_classifier:
            st.error("Models are not loaded correctly. Please check your installation.")
            st.stop()
            
        # Show progress bar for better UX
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Detecting acne
        status_text.text("Step 1/4: Detecting acne...")
        progress_bar.progress(25)
        
        # Run YOLOv8 detection
        results = detection_model(image)
        
        # Step 2: Processing results
        status_text.text("Step 2/4: Processing results...")
        progress_bar.progress(50)
        
        # Extract bounding boxes for blurring
        boxes = []
        acne_classifications = []
        acne_counts = {}
        
        if len(results[0].boxes) > 0:
            # For each detected acne, get coordinates
            for box in results[0].boxes.xyxy:
                boxes.append(box.cpu().numpy())
            
            # Apply blurring to face except acne areas
            blurred_image = blur_face_except_acne(image, boxes)
            
            # Step 3: Classifying acne types
            status_text.text("Step 3/4: Classifying acne types...")
            progress_bar.progress(75)
            
            # For each detected acne, crop and classify
            for i, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                acne_crop = image[y1:y2, x1:x2]
                
                if acne_crop.size > 0:  # Check if crop is valid
                    # Convert to PIL for classification
                    acne_crop_pil = Image.fromarray(cv2.cvtColor(acne_crop, cv2.COLOR_BGR2RGB))
                    
                    # Classify the cropped acne
                    if has_classifier:
                        acne_type, conf_score = classify_acne(acne_crop_pil, classification_model)
                        acne_classifications.append((acne_type, conf_score))
                        
                        # Count acne types
                        if acne_type in acne_counts:
                            acne_counts[acne_type] += 1
                        else:
                            acne_counts[acne_type] = 1
            
            # Get most common acne type for LLM analysis
            if acne_counts:
                most_common_acne = max(acne_counts.items(), key=lambda x: x[1])[0]
                st.session_state.primary_acne_type = most_common_acne
                
                # Step 4: Getting LLM analysis
                status_text.text(f"Step 4/4: Getting expert analysis for {most_common_acne.replace('_', ' ').title()} acne...")
                st.session_state.llm_analysis = get_llm_analysis(most_common_acne)
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.session_state.blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
            st.session_state.acne_classifications = acne_classifications
            st.session_state.acne_counts = acne_counts
            
            # Clear status
            status_text.empty()
            progress_bar.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display success message
            st.success("Analysis complete!")
        else:
            status_text.empty()
            progress_bar.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            st.warning("No acne detected in the image. Please upload a clearer image or try a different photo.")

# Display results if available
if st.session_state.results is not None:
    st.markdown('<div style="background-color: white; border-radius: 12px; padding: 2rem; margin: 2rem auto; max-width: 1000px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #1565C0; margin-bottom: 1.5rem;">Your Skin Analysis Results</h2>', unsafe_allow_html=True)
    
    # Display original and processed images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h3 style="text-align: center; color: #1976D2;">Original Image</h3>', unsafe_allow_html=True)
        st.image(st.session_state.processed_image, use_column_width=True)
    with col2:
        st.markdown('<h3 style="text-align: center; color: #1976D2;">Privacy Protected Image</h3>', unsafe_allow_html=True)
        st.image(st.session_state.blurred_image, use_column_width=True)
    
    # Display acne count summary with better visualization
    if st.session_state.acne_counts:
        st.markdown('<h3 style="text-align: center; color: #1565C0; margin-top: 2rem;">Acne Analysis Summary</h3>', unsafe_allow_html=True)
        
        # Total acne count
        total_acne = sum(st.session_state.acne_counts.values())
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 20px; background-color: #E3F2FD; padding: 15px; border-radius: 8px;">
            <div style="font-size: 1.2rem; color: #1565C0;">Total acne detected</div>
            <div style="font-size: 2.5rem; font-weight: bold; color: #1976D2;">{total_acne}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display acne types with counts using columns
        acne_types = list(st.session_state.acne_counts.items())
        for i in range(0, len(acne_types), 2):
            cols = st.columns(2)
            for j in range(2):
                if i+j < len(acne_types):
                    acne_type, count = acne_types[i+j]
                    percentage = (count / total_acne) * 100
                    with cols[j]:
                        st.markdown(f"""
                        <div style="background-color: white; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.1); display: flex; align-items: center; justify-content: space-between;">
                            <div>
                                <div style="font-weight: bold; color: #1976D2;">{acne_type.replace('_', ' ').title()}</div>
                                <div style="width: 100%; background-color: #E3F2FD; border-radius: 30px; height: 10px; margin: 0.5rem 0;">
                                    <div style="height: 100%; width: {percentage}%; background-color: #1976D2; border-radius: 30px;"></div>
                                </div>
                            </div>
                            <div style="margin-left: auto; background-color: #E3F2FD; color: #1976D2; font-weight: bold; padding: 0.3rem 0.8rem; border-radius: 20px;">{count}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Display LLM analysis for the primary acne type
        if st.session_state.primary_acne_type and st.session_state.llm_analysis:
            primary_acne = st.session_state.primary_acne_type.replace('_', ' ').title()
            llm_result = st.session_state.llm_analysis
            
            # Create two columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style='background-color: #E3F2FD; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                    <h3 style='color: #1565C0; margin-bottom: 15px;'>🔍 Understanding {primary_acne}</h3>
                    <div style='background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                        <p style='color: #34495E; line-height: 1.6;'>{llm_result["reasoning"]}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background-color: #E3F2FD; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                    <h3 style='color: #1565C0; margin-bottom: 15px;'>💡 Expert Recommendations</h3>
                    <div style='background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                        <div style='color: #34495E; line-height: 1.6;'>{llm_result["recommendations"]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add a warning message in a nice box
            st.markdown("""
            <div style='background-color: #FFF3E0; padding: 15px; border-radius: 8px; margin-top: 20px; border-left: 5px solid #FF9800;'>
                <p style='color: #E65100; margin: 0;'>
                    <strong>⚠️ Important Note:</strong> This is an automated analysis and does not replace professional medical advice. 
                    Please consult with a dermatologist for personalized treatment.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Reset analysis button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start New Analysis"):
            # Clear session state
            for key in ['results', 'processed_image', 'blurred_image', 'acne_classifications', 
                        'acne_counts', 'primary_acne_type', 'llm_analysis']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Add expander with information section at bottom
with st.expander("About This Application"):
    st.write("""
    This application uses computer vision and machine learning to:
    1. Detect acne lesions in an uploaded facial image
    2. Classify the type of acne present
    3. Generate personalized analysis and recommendations using an AI language model
    
    **Privacy Notice:** Your privacy is important to us. All processing is done within this application,
    and only anonymized acne types are sent to our AI model for analysis. Your images are never stored.
    
    **Medical Disclaimer:** This application is for informational purposes only and does not provide 
    medical advice. Always consult with a qualified healthcare provider for proper diagnosis and treatment.
    """)