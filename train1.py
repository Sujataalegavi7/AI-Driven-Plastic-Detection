import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(images_dir, labels_dir, img_size=(128, 128)):
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    images = []
    labels = []

    # Load images and labels
    for filename in os.listdir(images_dir):
        img_path = os.path.join(images_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)  # Resize image
        img = img / 255.0  # Normalize image
        images.append(img)

        # Load corresponding label
        label_path = os.path.join(labels_dir, filename.replace('.jpg', '.txt'))  # Assuming labels are in .txt format
        with open(label_path, 'r') as file:
            label_lines = file.readlines()  # Read all lines
            label_data = []
            for line in label_lines:
                parts = line.strip().split()
                if len(parts) < 5:  # Ensure we have at least class_id and 4 bounding box values
                    print(f"Invalid data line for {filename}: {line.strip()}")
                    continue
                
                class_id = int(parts[0])  # The first value is the class ID
                bounding_box = list(map(float, parts[1:5]))  # Convert only the first four values to float
                
                if len(bounding_box) == 4:  # Ensure we only take 4 bounding box values
                    label_data.append((class_id, bounding_box))
                else:
                    print(f"Invalid bounding box data for {filename}: {parts[1:]}")  # Handle invalid data

            labels.append(label_data)

    return np.array(images), labels

def prepare_labels(labels, num_classes):
    y = []
    for label_data in labels:
        current_label = np.zeros((num_classes + 4, 1))  # Assuming num_classes + 4 for bounding boxes
        for class_id, bounding_box in label_data:
            current_label[class_id, 0] = 1  # Set class label
            current_label[num_classes:num_classes + 4, 0] = bounding_box  # Set bounding box values
        y.append(current_label)
    
    return np.array(y)

def create_model(input_shape=(128, 128, 3), num_classes=2):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes + 4, activation='sigmoid')  # Change to 'sigmoid' for bounding boxes
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])  # Change loss for bounding box regression
    return model

# Load training and validation data
train_images_dir = 'D:/AI- Driven plastic bottle Detection/dataset/train/images'
train_labels_dir = 'D:/AI- Driven plastic bottle Detection/dataset/train/label'
val_images_dir = 'D:/AI- Driven plastic bottle Detection/dataset/valid/images'
val_labels_dir = 'D:/AI- Driven plastic bottle Detection/dataset/valid/label'

train_images, train_labels = load_data(train_images_dir, train_labels_dir)
val_images, val_labels = load_data(val_images_dir, val_labels_dir)

num_classes = 2  # Adjust based on your dataset
train_labels_prepared = prepare_labels(train_labels, num_classes)
val_labels_prepared = prepare_labels(val_labels, num_classes)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create and train the model
model = create_model(num_classes=num_classes)

# Monitor the best model during training
checkpoint = ModelCheckpoint('best_bottle_detection_model.keras', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Train the model
history = model.fit(datagen.flow(train_images, train_labels_prepared, batch_size=8),
                    epochs=50,
                    validation_data=(val_images, val_labels_prepared),
                    callbacks=[checkpoint, early_stopping, reduce_lr])

# Save the final model
model.save('bottle_detection_model.keras')
