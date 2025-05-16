import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import yaml
import argparse
from pathlib import Path

# Define the acne types (25 classes)
ACNE_TYPES = [
    "blackhead", "whitehead", "papule", "pustule", "nodule", 
    "cyst", "milia", "comedonal", "hormonal", "cystic", 
    "inflammatory", "noninflammatory", "mild", "moderate", "severe",
    "fungal", "rosacea", "perioral", "steroid", "excoriated",
    "acne_vulgaris", "acne_conglobata", "acne_fulminans", "pomade_acne", "mechanical_acne"
]

# Create class to ID mapping
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(ACNE_TYPES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(ACNE_TYPES)}

# Custom dataset for acne classification
class AcneDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = os.path.join(data_dir, split) if split else data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.csv_data = None
        
        # Check for CSV file (based on your file structure in screenshot)
        csv_path = os.path.join(data_dir, f"{split}.csv") if split else None
        if csv_path and os.path.exists(csv_path):
            self._load_from_csv(csv_path)
        # Check if the dataset has class folders
        elif os.path.exists(self.data_dir) and os.path.isdir(self.data_dir):
            self._load_from_folders()
        else:
            print(f"Warning: Could not find valid data in {self.data_dir}")
    
    def _load_from_folders(self):
        for class_name in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                # If class name is in our predefined types, use that index
                if class_name in CLASS_TO_IDX:
                    class_idx = CLASS_TO_IDX[class_name]
                # Otherwise, try to parse as integer or assign a new index
                else:
                    try:
                        class_idx = int(class_name.replace('class_', ''))
                    except ValueError:
                        # If we can't parse it, just assign a new index
                        class_idx = len(CLASS_TO_IDX)
                        CLASS_TO_IDX[class_name] = class_idx
                        IDX_TO_CLASS[class_idx] = class_name
                
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_idx)
    
    def _load_from_csv(self, csv_path):
        import pandas as pd
        try:
            self.csv_data = pd.read_csv(csv_path)
            
            # Try to find columns for filename and class
            filename_col = None
            for col in self.csv_data.columns:
                if 'file' in col.lower() or 'name' in col.lower() or 'image' in col.lower():
                    filename_col = col
                    break
            
            if not filename_col:
                # Assume first column is filename
                filename_col = self.csv_data.columns[0]
            
            # For each row, try to find a class
            for _, row in self.csv_data.iterrows():
                img_name = row[filename_col]
                img_path = os.path.join(self.data_dir, img_name)
                
                # Try to determine class from other columns
                # This is based on the pattern observed in your CSV (one-hot encoded classes)
                class_found = False
                for col in self.csv_data.columns[1:]:  # Skip filename column
                    if row[col] == 1:
                        # Found the class
                        class_name = col
                        if class_name in CLASS_TO_IDX:
                            class_idx = CLASS_TO_IDX[class_name]
                        else:
                            # Try to extract class name from column name
                            for acne_type in ACNE_TYPES:
                                if acne_type in class_name.lower():
                                    class_idx = CLASS_TO_IDX[acne_type]
                                    class_found = True
                                    break
                            
                            if not class_found:
                                class_idx = len(CLASS_TO_IDX)
                                CLASS_TO_IDX[class_name] = class_idx
                                IDX_TO_CLASS[class_idx] = class_name
                        
                        if os.path.exists(img_path):
                            self.image_paths.append(img_path)
                            self.labels.append(class_idx)
                        break
            
            print(f"Loaded {len(self.image_paths)} images from CSV {csv_path}")
            
        except Exception as e:
            print(f"Error loading from CSV: {e}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder to avoid breaking the dataloader
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, 0

# Function to create data loaders
def create_data_loaders(data_dir, batch_size=8, val_split=0.2):
    # Data transformations
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check if we have predefined splits
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        # Use predefined splits
        train_dataset = AcneDataset(data_dir, transform=transform_train, split='train')
        val_dataset = AcneDataset(data_dir, transform=transform_val, split='val')
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, len(train_dataset), len(val_dataset)
    else:
        # Load the dataset and do a random split
        full_dataset = AcneDataset(data_dir, transform=transform_train, split=None)
        
        # Split into train and validation
        dataset_size = len(full_dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        
        if dataset_size == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, train_size, val_size

# Function to train the model
def train_model(data_dir, epochs=5, batch_size=8, learning_rate=0.001, save_dir='runs/classify/train'):
    # Create data loaders
    try:
        train_loader, val_loader, train_size, val_size = create_data_loaders(data_dir, batch_size)
        print(f"Training with {train_size} images, validating with {val_size} images")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return None, None
    
    # Create the model (ResNet50 with pretrained weights)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(ACNE_TYPES))
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # Create save directory
    os.makedirs(f"{save_dir}/weights", exist_ok=True)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total if total > 0 else 0})
        
        train_loss = running_loss / train_size if train_size > 0 else 0
        train_acc = correct / total if total > 0 else 0
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'loss': loss.item(), 'acc': correct/total if total > 0 else 0})
        
        val_loss = running_loss / val_size if val_size > 0 else 0
        val_acc = correct / total if total > 0 else 0
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Adjust learning rate
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/weights/best.pt")
            print(f"Saved best model with validation accuracy: {val_acc:.4f}")
    
    # Save the final model
    torch.save(model.state_dict(), f"{save_dir}/weights/last.pt")
    
    # Save training history
    plot_training_history(history, save_dir)
    
    # Save class mapping
    with open(f"{save_dir}/class_mapping.yaml", 'w') as f:
        yaml.dump({
            'idx_to_class': IDX_TO_CLASS,
            'class_to_idx': CLASS_TO_IDX
        }, f)
    
    print(f"Training complete. Model saved to {save_dir}/weights/")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return model, history

# Function to plot training history
def plot_training_history(history, save_dir):
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_history.png")
    plt.close()

# Function to train YOLOv8 detection model
def train_yolo_model(data_yaml, epochs=5, save_dir='runs/detect/train'):
    try:
        from ultralytics import YOLO
        
        # Load a pretrained YOLOv8 model
        model = YOLO('yolov8n.pt')
        
        # Train the model
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=8,
            patience=3,
            project='runs/detect',
            name='train'
        )
        
        print(f"YOLOv8 training complete. Best model saved at: {save_dir}/weights/best.pt")
        return True
    except Exception as e:
        print(f"Error training YOLOv8 model: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Acne Classification and Detection Models')
    parser.add_argument('--data-classify', type=str, default='datasets/acne_classification', 
                        help='Path to classification dataset directory')
    parser.add_argument('--data-detect', type=str, default='data.yaml', 
                        help='Path to detection dataset YAML file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (fewer as requested)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--train-detection', action='store_true', help='Train YOLOv8 detection model')
    parser.add_argument('--train-classification', action='store_true', help='Train classification model')
    
    args = parser.parse_args()
    
    # Train the models
    if args.train_detection:
        print("Training YOLOv8 detection model...")
        train_yolo_model(args.data_detect, epochs=args.epochs)
    
    if args.train_classification:
        print("Training classification model...")
        train_model(
            data_dir=args.data_classify,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    if not args.train_detection and not args.train_classification:
        print("Training both models with default settings...")
        train_yolo_model(args.data_detect, epochs=args.epochs)
        train_model(
            data_dir=args.data_classify,
            epochs=args.epochs,
            batch_size=args.batch_size
        )