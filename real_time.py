import cv2
import numpy as np
import tensorflow as tf

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def process_predictions(predictions, num_classes):
    results = []
    for prediction in predictions:
        class_probs = prediction[:num_classes]  # Class probabilities
        bounding_box = prediction[num_classes:]  # Bounding box coordinates
        class_id = np.argmax(class_probs)  # Get the class with the highest probability
        results.append((class_id, bounding_box))
    return results

def draw_predictions(frame, results, img_size=(128, 128)):
    for (class_id, bounding_box) in results:
        h, w = frame.shape[0], frame.shape[1]

        # Rescale bounding box coordinates to the original frame size
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
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle

        # Add "Bottle Detected" text
        if class_id == 0:  # Assuming class 0 represents a bottle
            cv2.putText(frame, 'Bottle Detected', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        else:
            cv2.putText(frame, 'Unknown Object', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return frame

def real_time_bottle_detection(model_path, num_classes=2, img_size=(128, 128)):
    model = load_model(model_path)

    # Initialize the camera feed
    cap = cv2.VideoCapture(0)  # 0 for default camera, change index if using an external camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting real-time bottle detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame.")
            break

        # Preprocess frame for model
        input_img = cv2.resize(frame, img_size)  # Resize to model's input size
        input_img = input_img / 255.0  # Normalize
        input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(input_img)

        # Process predictions
        results = process_predictions(predictions, num_classes)

        # Draw predictions on the frame
        output_frame = draw_predictions(frame, results)

        # Display the frame
        cv2.imshow('Real-Time Bottle Detection', output_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

# Path to your trained model
model_path = 'D:/AI- Driven plastic bottle Detection/best_bottle_detection_model.keras'

# Start real-time detection
real_time_bottle_detection(model_path)
