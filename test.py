import os
import cv2
import numpy as np
import tensorflow as tf

def load_test_data(test_images_dir, img_size=(128, 128)):
    if not os.path.exists(test_images_dir):
        raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")

    test_images = []
    test_filenames = []

    # Load images from test directory
    for filename in os.listdir(test_images_dir):
        img_path = os.path.join(test_images_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)  # Resize image
        img = img / 255.0  # Normalize image
        test_images.append(img)
        test_filenames.append(filename)

    return np.array(test_images), test_filenames

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def predict(model, test_images):
    predictions = model.predict(test_images)
    return predictions

def process_predictions(predictions, num_classes):
    results = []
    for prediction in predictions:
        class_probs = prediction[:num_classes]  # Class probabilities
        bounding_box = prediction[num_classes:]  # Bounding box coordinates
        class_id = np.argmax(class_probs)  # Get the class with the highest probability
        results.append((class_id, bounding_box))
    return results

def draw_predictions(test_images, results, test_filenames):
    for i, (filename, (class_id, bounding_box)) in enumerate(zip(test_filenames, results)):
        # Rescale bounding box coordinates back to original image dimensions
        img = test_images[i]
        h, w = img.shape[0], img.shape[1]

        x_center = int(bounding_box[0] * w)
        y_center = int(bounding_box[1] * h)
        width = int(bounding_box[2] * w)
        height = int(bounding_box[3] * h)

        # Calculate top-left corner of the bounding box
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw rectangle around the detected object
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle

        # Add "Bottle Detected" text instead of Class ID
        if class_id == 0:  # Assuming class 0 represents a bottle
            print("Bottle Detected")
        else:
            print("Bottle not detected")

        # Display the image
        cv2.imshow(f'Result - {filename}', img)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

# Define paths
base_dir = 'D:/AI- Driven plastic bottle Detection'  # Base directory
test_images_dir = os.path.join(base_dir, 'test/images')  # Test images directory
model_path = os.path.join(base_dir, 'best_bottle_detection_model.keras')  # Model path

# Load test data and model
test_images, test_filenames = load_test_data(test_images_dir)
model = load_model(model_path)

# Make predictions
predictions = predict(model, test_images)

# Process predictions
num_classes = 2  # Adjust based on your dataset
results = process_predictions(predictions, num_classes)

# Draw predictions on images
draw_predictions(test_images, results, test_filenames)
