from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import base64
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torchvision.transforms as transforms
import os
import requests
import json
import yaml
from datetime import datetime

app = Flask(__name__, static_folder='static')

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# LLM API Configuration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "your_api_key_here")
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Model Paths
DETECTION_MODEL_PATH = "runs/detect/train6/weights/best.pt"
CLASSIFICATION_MODEL_PATH = "runs/classify/train/weights/best.pt"
CLASS_MAPPING_PATH = "runs/classify/train/class_mapping.yaml"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def query_together_ai(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are a dermatology expert assistant. Provide detailed, professional advice in JSON format."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=45
        )
        
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            try:
                # Try to extract JSON from markdown code block if present
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0]
                return json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, return as reasoning
                return {"reasoning": content}
        return None
    except Exception as e:
        print(f"API Error: {str(e)}")
        return None

def load_models():
    try:
        # Load detection model
        if not os.path.exists(DETECTION_MODEL_PATH):
            raise FileNotFoundError(f"Detection model not found at {DETECTION_MODEL_PATH}")
        detection_model = YOLO(DETECTION_MODEL_PATH)
        
        # Load classification model and mappings
        if not os.path.exists(CLASSIFICATION_MODEL_PATH):
            raise FileNotFoundError(f"Classification model not found at {CLASSIFICATION_MODEL_PATH}")
            
        with open(CLASS_MAPPING_PATH) as f:
            class_mapping = yaml.safe_load(f)
            CLASS_TO_IDX = class_mapping['class_to_idx']
            IDX_TO_CLASS = {int(k): v for k, v in class_mapping['idx_to_class'].items()}
        
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_TO_IDX))
        model.load_state_dict(torch.load(CLASSIFICATION_MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        
        print("✅ Models loaded successfully")
        return detection_model, model, IDX_TO_CLASS
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        return None, None, None

# Initialize models
detection_model, classification_model, IDX_TO_CLASS = load_models()

def classify_acne(image, model, idx_to_class):
    if model is None:
        return "Unknown", 0.0, {}
    
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Get top prediction
            _, predicted = torch.max(outputs, 1)
            top_class = idx_to_class.get(predicted.item(), "Unknown")
            top_conf = probabilities[predicted.item()].item()
            
            # Get all class probabilities
            all_probs = {}
            for idx, prob in enumerate(probabilities):
                if idx in idx_to_class and prob.item() > 0.05:
                    all_probs[idx_to_class[idx]] = prob.item()
            
            return top_class, top_conf, all_probs
        
    except Exception as e:
        print(f"Classification error: {str(e)}")
        return "Unknown", 0.0, {}

def blur_face_except_acne(image, boxes, padding=15):
    try:
        blurred = cv2.GaussianBlur(image.copy(), (51, 51), 30)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1-padding), max(0, y1-padding)
            x2, y2 = min(image.shape[1], x2+padding), min(image.shape[0], y2+padding)
            mask[y1:y2, x1:x2] = 255
        
        return cv2.bitwise_and(image, image, mask=mask) + cv2.bitwise_and(blurred, blurred, mask=~mask)
    except Exception as e:
        print(f"Blur error: {str(e)}")
        return image

def draw_acne_annotations(image, boxes, acne_types, confidences):
    annotated_img = image.copy()
    color_map = {
        "Hormonal": (0, 0, 255),    # Red
        "Mild": (0, 255, 0),        # Green
        "Moderate": (255, 0, 0),    # Blue
        "Severe": (0, 255, 255),    # Yellow
        "Cystic": (255, 0, 255),    # Magenta
        "Unknown": (128, 128, 128)  # Gray
    }
    
    for i, (box, acne_type, conf) in enumerate(zip(boxes, acne_types, confidences)):
        x1, y1, x2, y2 = map(int, box)
        color = color_map.get(acne_type, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        
        # Create label with ID and acne type
        label = f"#{i+1}: {acne_type} ({conf:.1f})"
        
        # Calculate text size and position
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_img, (x1, y1-20), (x1+label_width, y1), color, -1)
        cv2.putText(annotated_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add legend
    legend_y = annotated_img.shape[0] - 30
    for i, (acne_type, color) in enumerate(color_map.items()):
        if acne_type == "Unknown":
            continue
        x_pos = 10 + i * 150
        cv2.rectangle(annotated_img, (x_pos, legend_y), (x_pos+15, legend_y+15), color, -1)
        cv2.putText(annotated_img, acne_type, (x_pos+20, legend_y+12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return annotated_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/project')
def project():
    return render_template('project.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/analyze_acne', methods=['POST'])
def analyze_acne():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Process image
        img_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        if detection_model is None or classification_model is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        # Detect acne with confidence threshold
        results = detection_model(image, conf=0.3)
        boxes = []
        valid_detections = 0
        
        for box in results[0].boxes:
            coords = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            if ((coords[2]-coords[0]) > 5 and 
                (coords[3]-coords[1]) > 5 and 
                conf > 0.3):
                boxes.append(coords.tolist())
                valid_detections += 1
        
        # Analyze detected acne
        acne_counts = {}
        acne_types_list = []
        confidences_list = []
        detailed_acne_info = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            acne_crop = image[y1:y2, x1:x2]
            
            if acne_crop.size > 0:
                acne_type, conf, all_probs = classify_acne(acne_crop, classification_model, IDX_TO_CLASS)
                
                # Store results
                acne_counts[acne_type] = acne_counts.get(acne_type, 0) + 1
                acne_types_list.append(acne_type)
                confidences_list.append(conf)
                
                detailed_acne_info.append({
                    "id": i + 1,
                    "type": acne_type,
                    "confidence": conf,
                    "location": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "alternative_classifications": all_probs
                })
        
        # Draw annotations
        annotated_img = draw_acne_annotations(image, boxes, acne_types_list, confidences_list)
        
        # Generate blurred image
        blurred_img = blur_face_except_acne(image, boxes)
        
        # Convert images to base64
        _, buffer_blurred = cv2.imencode('.jpg', blurred_img)
        blurred_base64 = base64.b64encode(buffer_blurred).decode('utf-8')
        
        _, buffer_annotated = cv2.imencode('.jpg', annotated_img)
        annotated_base64 = base64.b64encode(buffer_annotated).decode('utf-8')
        
        # Determine primary type
        primary_type = max(acne_counts.items(), key=lambda x: x[1])[0] if acne_counts else "None Detected"
        
        # Generate AI analysis prompt
        prompt = f"""As a dermatology expert, analyze this acne case and provide:
1. Professional assessment (reasoning)
2. Recommended products (product_recommendations)
3. Lifestyle changes (lifestyle_recommendations)
4. When to see a doctor (when_to_see_doctor)

Patient's acne analysis:
- Total lesions: {valid_detections}
- Type breakdown: {', '.join([f"{count} {type}" for type, count in acne_counts.items()])}
- Primary type: {primary_type}

Respond in this exact JSON format:
{{
    "reasoning": "Detailed analysis...",
    "product_recommendations": ["item1", "item2"],
    "lifestyle_recommendations": ["advice1", "advice2"],
    "when_to_see_doctor": "When condition..."
}}"""

        # Get AI analysis
        ai_response = query_together_ai(prompt) or {}
        
        return jsonify({
            'success': True,
            'acne_count': valid_detections,
            'primary_type': primary_type,
            'acne_types': acne_counts,
            'blurred_image': blurred_base64,
            'annotated_image': annotated_base64,
            'detailed_acne_info': detailed_acne_info,
            'reasoning': ai_response.get('reasoning', 'No detailed analysis available'),
            'product_recommendations': ai_response.get('product_recommendations', []),
            'lifestyle_recommendations': ai_response.get('lifestyle_recommendations', []),
            'when_to_see_doctor': ai_response.get('when_to_see_doctor', 'Consult a dermatologist if condition persists')
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        context = data.get('context', '')
        
        prompt = f"""As a dermatology chatbot, answer this patient query based on {context}.
        
        Patient Question: {message}
        
        Provide a detailed, professional response in plain text (no JSON)."""
        
        response = query_together_ai(prompt)
        if isinstance(response, dict):
            response = response.get('reasoning', "I couldn't process that request. Please try again.")
        
        return jsonify({
            'response': response or "I couldn't generate a response. Please try again."
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)