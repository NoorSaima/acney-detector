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
import yaml

# Load class mappings from YAML
def load_class_mappings(model_dir):
    map_path = os.path.join(model_dir, "class_mapping.yaml")
    if os.path.exists(map_path):
        with open(map_path, 'r') as f:
            class_mapping = yaml.safe_load(f)
            return class_mapping['class_to_idx'], class_mapping['idx_to_class']
    return None, None

# Initialize class mappings
CLASS_TO_IDX, IDX_TO_CLASS = load_class_mappings("runs/classify/train")
ACNE_TYPES = list(CLASS_TO_IDX.keys()) if CLASS_TO_IDX else [
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
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

# Set page configuration
st.set_page_config(
    page_title="AI Dermatologist",
    page_icon="🔬",
    layout="wide"
)

# Apply custom CSS
st.markdown("""
<style>
    /* Remove Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}
    
    /* Global styling */
    body {
        font-family: 'Arial', sans-serif;
        color: #333;
        background-color: #f8f9fa;
    }
    
    /* Main container styling */
    .main-container {
        background-color: white;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.05);
        margin: 20px 0;
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #ddd;
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        margin: 30px 0;
        background-color: #fafafa;
        transition: all 0.3s;
    }
    
    .upload-area:hover {
        border-color: #0a0066;
        background-color: #f5f5ff;
    }
    
    /* Live detection container */
    .live-container {
        border: 2px solid #0a0066;
        border-radius: 10px;
        padding: 20px;
        margin: 30px 0;
        text-align: center;
        background-color: #f5f5ff;
    }
    
    /* Webcam feed styling */
    .webcam-feed {
        max-width: 640px;
        margin: 20px auto;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Results section */
    .results-container {
        background-color: white;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.05);
        margin-top: 40px;
    }
    
    /* Chat container styling */
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 15px;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 15px;
    }
    
    .user-message {
        background-color: #e6ecff;
        padding: 12px 15px;
        border-radius: 18px 18px 0 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .bot-message {
        background-color: white;
        padding: 12px 15px;
        border-radius: 18px 18px 18px 0;
        margin: 8px 0;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Buttons */
    .primary-button {
        background-color: #0a0066;
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 30px;
        font-size: 1rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s;
        display: inline-block;
        text-align: center;
        box-shadow: 0 4px 12px rgba(10, 0, 102, 0.2);
    }
    
    .primary-button:hover {
        background-color: #0a0099;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(10, 0, 102, 0.3);
    }
    
    /* Warning box */
    .warning-box {
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
        border-left: 5px solid #FF9800;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained YOLOv8 detection model
@st.cache_resource
def load_detection_model():
    try:
        model = YOLO(DETECTION_MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load detection model: {str(e)}")
        return None

# Define classification model
class AcneClassifier(nn.Module):
    def __init__(self, num_classes=25):
        super(AcneClassifier, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# Function to load classification model
@st.cache_resource
def load_classification_model(model_path=None):
    try:
        model = AcneClassifier(num_classes=len(ACNE_TYPES))
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load classification model: {str(e)}")
        return None

# Function to classify acne type from image
def classify_acne(image, model):
    if model is None:
        return "Unknown", 0.0
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    acne_type = ACNE_TYPES[predicted.item()]
    conf_score = confidence[predicted.item()].item()
    
    return acne_type, conf_score

# Function to blur face while preserving acne areas
def blur_face_except_acne(image, boxes, padding=10):
    blurred = image.copy()
    blurred = cv2.GaussianBlur(blurred, (51, 51), 30)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        mask[y1:y2, x1:x2] = 255
    
    inv_mask = cv2.bitwise_not(mask)
    fg = cv2.bitwise_and(image, image, mask=mask)
    bg = cv2.bitwise_and(blurred, blurred, mask=inv_mask)
    return cv2.add(fg, bg)

# Updated Live Detection Function
def detect_acne_live():
    st.session_state.live_mode = True
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button = st.button("Stop Live Detection", use_container_width=True)
    
    # Add confidence threshold slider
    conf_threshold = st.slider("Detection Confidence", 0.1, 0.9, 0.5, 0.05)
    
    # Face detection using OpenCV's Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Create compact layout for controls
    col1, col2 = st.columns(2)
    with col1:
        privacy_mode = st.checkbox("Privacy Mode (Blur non-acne areas)", value=False)
    with col2:
        show_details = st.checkbox("Show Detailed Analysis", value=False)
    
    # Acne type color mapping for all 25 types
    acne_colors = {}
    color_palette = [
        (255, 0, 0),     # Red
        (0, 255, 0),     # Green
        (0, 0, 255),     # Blue
        (255, 255, 0),   # Yellow
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Cyan
        (128, 0, 0),     # Maroon
        (0, 128, 0),     # Dark Green
        (0, 0, 128),     # Navy
        (128, 128, 0),   # Olive
        (128, 0, 128),   # Purple
        (0, 128, 128)    # Teal
    ]
    
    # Initialize color mapping for all 25 acne types
    for i, acne_type in enumerate(ACNE_TYPES):
        acne_colors[acne_type] = color_palette[i % len(color_palette)]
    
    # Display setup instructions
    st.info("📷 Position your face in center with good lighting. Keep still for accurate detection.")
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from webcam")
            break
        
        # Convert to RGB for detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Create a blurred version for privacy mode
        blurred_frame = cv2.GaussianBlur(frame_rgb, (25, 25), 15)
        
        # Final output frame
        output_frame = frame_rgb.copy()
        
        # Face detection results
        detected_face = False
        all_acne_detections = []
        acne_count_by_type = {}
        
        for (x, y, w, h) in faces:
            detected_face = True
            face_area = frame_rgb[y:y+h, x:x+w]
            
            # Draw rectangle around the face
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Apply privacy mode if enabled
            if privacy_mode:
                # Create mask for just the face area
                mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
                mask[y:y+h, x:x+w] = 255
                
                # Apply the mask
                inv_mask = cv2.bitwise_not(mask)
                fg = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask)
                bg = cv2.bitwise_and(blurred_frame, blurred_frame, mask=inv_mask)
                output_frame = cv2.add(fg, bg)
            
            # Process face area for acne detection
            results = detection_model(face_area, conf=conf_threshold)
            
            for box in results[0].boxes:
                box_coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = box_coords
                w_acne, h_acne = x2-x1, y2-y1
                
                # Filter for appropriate acne size
                if w_acne < w*0.15 and h_acne < h*0.15:
                    # Classify acne if classification model exists
                    acne_type = "acne"
                    if classification_model:
                        try:
                            acne_crop = face_area[int(y1):int(y2), int(x1):int(x2)]
                            if acne_crop.size > 0:
                                acne_crop_pil = Image.fromarray(acne_crop)
                                acne_type, conf_score = classify_acne(acne_crop_pil, classification_model)
                        except Exception:
                            pass  # Silent fail on classification errors
                    
                    # Add face offset to box coordinates for global frame
                    global_x1, global_y1 = x1 + x, y1 + y
                    global_x2, global_y2 = x2 + x, y2 + y
                    
                    # Store detection
                    all_acne_detections.append((global_x1, global_y1, global_x2, global_y2, acne_type))
                    
                    # Count by type
                    acne_count_by_type[acne_type] = acne_count_by_type.get(acne_type, 0) + 1
        
        # Draw acne boxes
        for x1, y1, x2, y2, acne_type in all_acne_detections:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = acne_colors.get(acne_type, (255, 0, 0))
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        
        # Display statistics overlay
        if detected_face:
            # Only show basic stats by default
            cv2.putText(output_frame, f"Detected: {len(all_acne_detections)}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show detailed analysis if enabled
            if show_details and acne_count_by_type:
                y_pos = 55
                # Get top 3 acne types
                top_types = sorted(acne_count_by_type.items(), key=lambda x: x[1], reverse=True)[:3]
                for acne_type, count in top_types:
                    display_type = acne_type.replace('_', ' ')
                    if len(display_type) > 10:
                        display_type = display_type[:10]  # Truncate long names
                    cv2.putText(output_frame, f"{display_type}: {count}", 
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_pos += 20
        else:
            # Face detection guidance
            cv2.putText(output_frame, "Position your face in the center", 
                      (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        frame_placeholder.image(output_frame, channels="RGB", use_container_width=True)
        
        if stop_button:
            break
    
    cap.release()
    st.session_state.live_mode = False
    frame_placeholder.empty()
    st.success("Analysis completed")  # More minimal success message

def get_offline_analysis(acne_type):
    """Fallback responses when API isn't available"""
    offline_responses = {
        "blackhead": {
            "reasoning": "Blackheads form when pores become clogged with excess oil and dead skin cells.",
            "recommendations": "• Use salicylic acid cleansers\n• Try retinoid creams\n• Avoid pore-clogging products\n• Get professional extractions\n• Use non-comedogenic moisturizers"
        },
        "whitehead": {
            "reasoning": "Whiteheads occur when pores are completely blocked by oil and dead skin cells.",
            "recommendations": "• Use benzoyl peroxide treatments\n• Apply gentle exfoliants\n• Avoid picking at lesions\n• Use oil-free products\n• Consider professional facials"
        },
        "papule": {
            "reasoning": "Papules are small, inflamed bumps without visible pus, caused by bacterial infection and inflammation.",
            "recommendations": "• Apply topical antibiotics\n• Use anti-inflammatory creams\n• Avoid touching your face\n• Try tea tree oil treatments\n• Use gentle skincare products"
        },
        "pustule": {
            "reasoning": "Pustules are inflamed pimples filled with pus, often caused by bacterial infection.",
            "recommendations": "• Apply spot treatments with benzoyl peroxide\n• Use warm compresses\n• Avoid popping pustules\n• Keep skin clean\n• See a dermatologist for persistent cases"
        },
        "cyst": {
            "reasoning": "Cystic acne forms deep within the skin when pores become blocked and infected.",
            "recommendations": "• Consult a dermatologist\n• Consider prescription medications\n• Apply warm compresses\n• Avoid picking or squeezing\n• Use gentle, non-irritating products"
        }
    }
    return offline_responses.get(acne_type, {
        "reasoning": f"Consult a dermatologist for proper diagnosis and treatment of {acne_type.replace('_', ' ')}.",
        "recommendations": "• Maintain a consistent skincare routine\n• Avoid picking at acne\n• Use non-comedogenic products\n• Stay hydrated\n• See a dermatologist for persistent cases"
    })

def get_llm_analysis(acne_type):
    try:
        formatted_acne_type = acne_type.replace('_', ' ').title()
        
        if not TOGETHER_API_KEY:
            st.error("API key not configured. Using offline analysis.")
            return get_offline_analysis(acne_type)
        
        # Try to get response from API
        response = query_together_ai(formatted_acne_type)
        
        if response:
            return response
        else:
            st.error("Could not get response from API. Using offline analysis.")
            return get_offline_analysis(acne_type)
            
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        return get_offline_analysis(acne_type)

def query_together_ai(acne_type):
    """Improved API query function with better error handling"""
    endpoint = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOGETHER_API_KEY}"
    }
    
    prompt = f"""As a dermatology expert, provide analysis of {acne_type} acne in this EXACT JSON format:
{{
    "reasoning": "2-3 sentence explanation of causes",
    "recommendations": "5 bullet-pointed recommendations starting with •"
}}"""
    
    data = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
            {
                "role": "system", 
                "content": "You are a dermatology expert. Provide responses in perfect JSON format only."
            },
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(endpoint, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        
        content = response.json()['choices'][0]['message']['content']
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            st.error("API returned malformed JSON. Using offline analysis.")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None

def chat_with_llm(user_query, context=None):
    """Function to handle chat interactions with LLM"""
    try:
        if not TOGETHER_API_KEY:
            return "I'm currently operating in offline mode. For more detailed answers, please configure the API key."
            
        endpoint = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOGETHER_API_KEY}"
        }
        
        # Build the prompt with context if available
        messages = [
            {
                "role": "system",
                "content": "You are a dermatology expert assistant. Answer questions about acne clearly and concisely."
            }
        ]
        
        if context:
            messages.append({
                "role": "system",
                "content": f"Context about the user's acne condition:\n{context}"
            })
        
        messages.append({
            "role": "user",
            "content": user_query
        })
        
        data = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        response = requests.post(endpoint, headers=headers, json=data, timeout=15)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"I'm having trouble connecting to the expert system. Please try again later. (Error {response.status_code})"
    
    except Exception as e:
        return f"I'm experiencing technical difficulties. Please try again later. (Error: {str(e)})"

# Initialize models
detection_model = load_detection_model()
classification_model = load_classification_model(CLASSIFICATION_MODEL_PATH)
has_classifier = classification_model is not None

# Initialize session state
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
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'live_mode' not in st.session_state:
    st.session_state.live_mode = False

# Main page layout
st.title("AI Dermatologist")
st.write("Upload a photo or use live detection to analyze your acne condition")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Upload Image", "Live Detection"])

with tab1:
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width=400)
        
        if st.button("Analyze Skin", key="analyze_button", use_container_width=True):
            if detection_model is None:
                st.error("Detection model failed to load")
                st.stop()
                
            with st.status("Analyzing your skin...", expanded=True) as status:
                status.update(label="Detecting acne...", state="running")
                results = detection_model(image)
                
                if len(results[0].boxes) > 0:
                    boxes = [box.cpu().numpy() for box in results[0].boxes.xyxy]
                    blurred_image = blur_face_except_acne(image, boxes)
                    
                    status.update(label="Classifying acne types...", state="running")
                    acne_classifications = []
                    acne_counts = {}
                    
                    for box in results[0].boxes.xyxy:
                        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                        acne_crop = image[y1:y2, x1:x2]
                        
                        if acne_crop.size > 0:
                            acne_crop_pil = Image.fromarray(cv2.cvtColor(acne_crop, cv2.COLOR_BGR2RGB))
                            if has_classifier:
                                acne_type, conf_score = classify_acne(acne_crop_pil, classification_model)
                                acne_classifications.append((acne_type, conf_score))
                                acne_counts[acne_type] = acne_counts.get(acne_type, 0) + 1
                    
                    if acne_counts:
                        most_common_acne = max(acne_counts.items(), key=lambda x: x[1])[0]
                        st.session_state.primary_acne_type = most_common_acne
                    
                    st.session_state.results = results
                    st.session_state.processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.session_state.blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
                    st.session_state.acne_classifications = acne_classifications
                    st.session_state.acne_counts = acne_counts
                    
                    status.update(label="Analysis complete!", state="complete")
                    st.rerun()
                else:
                    status.update(label="No acne detected", state="complete")
                    st.warning("No acne detected in the image")

with tab2:
    st.markdown('<div class="live-container">', unsafe_allow_html=True)
    st.write("Real-time acne detection using your webcam")
    
    if st.button("Start Live Detection", key="live_button", use_container_width=True):
        if detection_model is None:
            st.error("Detection model not loaded")
        else:
            detect_acne_live()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Display results if available
if st.session_state.results is not None:
    with st.container():
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown('<h2 style="text-align: center;">Your Skin Analysis Results</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<h3 style="text-align: center;">Original Image</h3>', unsafe_allow_html=True)
            st.image(st.session_state.processed_image, width=300)
        with col2:
            st.markdown('<h3 style="text-align: center;">Privacy Protected</h3>', unsafe_allow_html=True)
            st.image(st.session_state.blurred_image, width=300)
        
        if st.session_state.acne_counts:
            total_acne = sum(st.session_state.acne_counts.values())
            st.markdown(f"""
            <div style="text-align: center; margin: 20px 0; background-color: #E3F2FD; padding: 15px; border-radius: 8px;">
                <div style="font-size: 1.2rem;">Total acne detected</div>
                <div style="font-size: 2.5rem; font-weight: bold;">{total_acne}</div>
            </div>
            """, unsafe_allow_html=True)
            
            for acne_type, count in st.session_state.acne_counts.items():
                percentage = (count / total_acne) * 100
                st.markdown(f"""
                <div style="background-color: white; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="font-weight: bold;">{acne_type.replace('_', ' ').title()}</div>
                        <div style="background-color: #E3F2FD; padding: 0.3rem 0.8rem; border-radius: 20px;">{count}</div>
                    </div>
                    <div style="width: 100%; background-color: #E3F2FD; border-radius: 30px; height: 10px; margin-top: 0.5rem;">
                        <div style="height: 100%; width: {percentage}%; background-color: #0a0066; border-radius: 30px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.session_state.primary_acne_type:
                primary_acne = st.session_state.primary_acne_type.replace('_', ' ').title()
                
                if st.button("Get Expert Analysis", key="expert_analysis"):
                    with st.spinner("Getting expert analysis..."):
                        st.session_state.llm_analysis = get_llm_analysis(st.session_state.primary_acne_type)
                    st.rerun()
                
                if st.session_state.llm_analysis:
                    llm_result = st.session_state.llm_analysis
                    
                    st.markdown(f"""
                    <div style='background-color: #E3F2FD; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                        <h3 style='margin-bottom: 15px;'>Understanding {primary_acne}</h3>
                        <div style='background-color: white; padding: 15px; border-radius: 8px;'>
                            <p>{llm_result["reasoning"]}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style='background-color: #E3F2FD; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                        <h3 style='margin-bottom: 15px;'>Expert Recommendations</h3>
                        <div style='background-color: white; padding: 15px; border-radius: 8px;'>
                            <div>{llm_result["recommendations"]}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <p><strong>⚠️ Important Note:</strong> This is an automated analysis and does not replace professional medical advice.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Start New Analysis", use_container_width=True):
            for key in ['results', 'processed_image', 'blurred_image', 'acne_classifications', 
                       'acne_counts', 'primary_acne_type', 'llm_analysis']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Chatbot sidebar

# Chatbot sidebar
with st.sidebar:
    st.title("Acne Expert Chat")
    st.write("Ask questions about your acne condition")
    
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    user_input = st.chat_input("Ask about your acne...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        context = None
        if st.session_state.primary_acne_type and st.session_state.acne_counts:
            context = f"User has {st.session_state.primary_acne_type.replace('_', ' ')} acne. Detected: {', '.join([f'{k.replace('_', ' ')} ({v})' for k,v in st.session_state.acne_counts.items()])}"
        
        with st.spinner("Thinking..."):
            response = chat_with_llm(user_input, context)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
# About section
with st.expander("About This App"):
    st.write("""
    This AI dermatologist helps analyze acne conditions using computer vision and machine learning.
    
    **Features:**
    - Upload images or use live webcam detection
    - Identifies and classifies different acne types
    - Provides expert recommendations
    
    **Privacy:** Your images are processed locally and never stored.
    
    **Disclaimer:** For informational purposes only. Consult a dermatologist for medical advice.
    """)
