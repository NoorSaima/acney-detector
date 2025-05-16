import os
import yaml
import shutil
import argparse
from pathlib import Path
import random
from PIL import Image
import cv2
import numpy as np

def download_from_roboflow(api_key=None, workspace='claude-yjecd', project='acne-classification-7wwrm', version=1):
    """
    Download a dataset from Roboflow. This requires the roboflow Python package.
    
    Args:
        api_key: Your Roboflow API key
        workspace: Roboflow workspace name
        project: Roboflow project name
        version: Dataset version
    
    Returns:
        Path to the downloaded dataset
    """
    try:
        from roboflow import Roboflow
        
        if api_key is None:
            # Check if API key is in environment variable
            api_key = os.environ.get('ROBOFLOW_API_KEY')
            if api_key is None:
                raise ValueError("API key not provided and not found in environment variables")
        
        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)
        
        # Get the project
        project = rf.workspace(workspace).project(project)
        
        # Download the dataset
        dataset = project.version(version).download("folder")
        
        return dataset.location
    except ImportError:
        print("Roboflow package not installed. Install with: pip install roboflow")
        return None
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def organize_dataset(data_dir, output_dir):
    """
    Organize dataset into classes based on YOLOv8 classification format or Roboflow format.
    
    Args:
        data_dir: Path to downloaded dataset
        output_dir: Path to save organized dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if this is a YOLO format dataset
    if os.path.exists(os.path.join(data_dir, 'train', 'labels')):
        print("Detected YOLO format dataset")
        organize_yolo_dataset(data_dir, output_dir)
    else:
        print("Checking for image classification format")
        organize_classification_dataset(data_dir, output_dir)

def organize_yolo_dataset(data_dir, output_dir):
    """
    Organize a YOLO format dataset (object detection) into a classification dataset.
    
    Args:
        data_dir: Path to YOLO dataset
        output_dir: Path to save organized dataset
    """
    # Check for YAML file
    yaml_files = list(Path(data_dir).glob('*.yaml'))
    class_names = None
    
    if yaml_files:
        with open(yaml_files[0], 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data:
                class_names = data['names']
                print(f"Found classes: {class_names}")
    
    # If no class names found, look for labels
    if class_names is None:
        print("No class names found in YAML. Attempting to infer from labels.")
        class_names = {}
        label_dir = os.path.join(data_dir, 'train', 'labels')
        if os.path.exists(label_dir):
            for label_file in os.listdir(label_dir):
                if label_file.endswith('.txt'):
                    with open(os.path.join(label_dir, label_file), 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                class_names[class_id] = f"class_{class_id}"
    
    # Process the dataset
    for split in ['train', 'valid', 'test']:
        if not os.path.exists(os.path.join(data_dir, split)):
            continue
        
        # Create output directories for this split
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Process images
        img_dir = os.path.join(data_dir, split, 'images')
        label_dir = os.path.join(data_dir, split, 'labels')
        
        if os.path.exists(img_dir) and os.path.exists(label_dir):
            for img_file in os.listdir(img_dir):
                base_name = os.path.splitext(img_file)[0]
                label_file = f"{base_name}.txt"
                label_path = os.path.join(label_dir, label_file)
                
                if os.path.exists(label_path):
                    # Read the label file to get class ID
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                if class_id in class_names:
                                    class_name = class_names[class_id]
                                else:
                                    class_name = f"class_{class_id}"
                                
                                # Create class directory if it doesn't exist
                                class_dir = os.path.join(split_output_dir, class_name)
                                os.makedirs(class_dir, exist_ok=True)
                                
                                # Copy image to class directory
                                img_path = os.path.join(img_dir, img_file)
                                shutil.copy(img_path, os.path.join(class_dir, img_file))

def organize_classification_dataset(data_dir, output_dir):
    """
    Organize a classification dataset.
    
    Args:
        data_dir: Path to classification dataset
        output_dir: Path to save organized dataset
    """
    # Check if the dataset is already organized into classes
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        # Look for class directories
        has_class_dirs = False
        for item in os.listdir(split_dir):
            if os.path.isdir(os.path.join(split_dir, item)):
                has_class_dirs = True
                break
        
        if has_class_dirs:
            print(f"Found existing class directories in {split} split")
            # Copy the structure as is
            split_output_dir = os.path.join(output_dir, split)
            os.makedirs(split_output_dir, exist_ok=True)
            
            for class_name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, class_name)
                if os.path.isdir(class_dir):
                    class_output_dir = os.path.join(split_output_dir, class_name)
                    os.makedirs(class_output_dir, exist_ok=True)
                    
                    for img_file in os.listdir(class_dir):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_dir, img_file)
                            shutil.copy(img_path, os.path.join(class_output_dir, img_file))
        else:
            # Check for a corresponding labels file
            label_file = os.path.join(data_dir, f"{split}.txt")
            if os.path.exists(label_file):
                print(f"Found label file for {split} split")
                split_output_dir = os.path.join(output_dir, split)
                os.makedirs(split_output_dir, exist_ok=True)
                
                # Read labels
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            img_path = os.path.join(data_dir, split, parts[0])
                            if os.path.exists(img_path):
                                try:
                                    class_name = parts[1]
                                    # Handle numeric class labels
                                    if class_name.isdigit():
                                        class_name = f"class_{class_name}"
                                    
                                    class_output_dir = os.path.join(split_output_dir, class_name)
                                    os.makedirs(class_output_dir, exist_ok=True)
                                    
                                    shutil.copy(img_path, os.path.join(class_output_dir, os.path.basename(img_path)))
                                except Exception as e:
                                    print(f"Error processing {img_path}: {e}")
            else:
                print(f"Warning: Could not find class structure or labels for {split} split")

def create_dummy_dataset(output_dir, num_classes=25, num_images_per_class=10):
    """
    Create a dummy dataset for testing when no labeled data is available.
    
    Args:
        output_dir: Path to save the dummy dataset
        num_classes: Number of classes to create
        num_images_per_class: Number of images to generate per class
    """
    print(f"Creating dummy dataset with {num_classes} classes, {num_images_per_class} images per class")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create train directory
    train_dir = os.path.join(output_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    
    # Define class names (these should match your expected classes)
    class_names = [
        "blackhead", "whitehead", "papule", "pustule", "nodule", 
        "cyst", "milia", "comedonal", "hormonal", "cystic", 
        "inflammatory", "noninflammatory", "mild", "moderate", "severe",
        "fungal", "rosacea", "perioral", "steroid", "excoriated",
        "acne_vulgaris", "acne_conglobata", "acne_fulminans", "pomade_acne", "mechanical_acne"
    ]
    
    # Ensure we have enough class names
    while len(class_names) < num_classes:
        class_names.append(f"class_{len(class_names)}")
    
    # Use only the number of classes we need
    class_names = class_names[:num_classes]
    
    # Generate dummy images for each class
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(train_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for img_idx in range(num_images_per_class):
            # Create a colored image with some random patterns
            img = np.ones((224, 224, 3), dtype=np.uint8) * (class_idx * 10 + 50)
            
            # Add some random circles to simulate acne
            for _ in range(random.randint(5, 20)):
                center = (random.randint(0, 224), random.randint(0, 224))
                radius = random.randint(5, 30)
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
                cv2.circle(img, center, radius, color, -1)
            
            # Save the image
            img_path = os.path.join(class_dir, f"{class_name}_{img_idx}.jpg")
            cv2.imwrite(img_path, img)
    
    print(f"Dummy dataset created at {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Prepare Acne Classification Dataset')
    parser.add_argument('--download', action='store_true', help='Download dataset from Roboflow')
    parser.add_argument('--api-key', type=str, help='Roboflow API key')
    parser.add_argument('--workspace', type=str, default='claude-yjecd', help='Roboflow workspace')
    parser.add_argument('--project', type=str, default='acne-classification-7wwrm', help='Roboflow project')
    parser.add_argument('--version', type=int, default=1, help='Dataset version')
    parser.add_argument('--data-dir', type=str, help='Path to existing dataset directory')
    parser.add_argument('--output-dir', type=str, default='datasets/acne_classification', help='Output directory')
    parser.add_argument('--create-dummy', action='store_true', help='Create dummy dataset if no labeled data available')
    parser.add_argument('--num-classes', type=int, default=25, help='Number of classes for dummy dataset')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    data_dir = None
    
    # Download dataset if requested
    if args.download:
        print("Downloading dataset from Roboflow...")
        data_dir = download_from_roboflow(
            api_key=args.api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version
        )
        if data_dir:
            print(f"Dataset downloaded to {data_dir}")
    elif args.data_dir:
        data_dir = args.data_dir
    
    # Organize dataset
    if data_dir and os.path.exists(data_dir):
        print(f"Organizing dataset from {data_dir} to {args.output_dir}")
        organize_dataset(data_dir, args.output_dir)
    elif args.create_dummy:
        print("No dataset provided. Creating dummy dataset for testing...")
        create_dummy_dataset(args.output_dir, args.num_classes)
    else:
        print("No dataset provided and dummy dataset creation not requested. Exiting.")

if __name__ == '__main__':
    main()