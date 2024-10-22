import cv2
import boto3
import time
import os
import threading
import signal
import sys
from datetime import datetime  # Import datetime module

# AWS S3 Configuration
S3_BUCKET = 'belt-model'
ACCESS_KEY = 'ADDYOURKEY'  # Optional if you have aws configure
SECRET_KEY = 'ADDYOURKEY'  # Optional if you have aws configure
REGION = 'ap-south-1'

# Initialize S3 client
s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)

# Create temp directory if it doesn't exist
temp_dir = "/home/host123/temp"
os.makedirs(temp_dir, exist_ok=True)

# Set up video capture (USB camera is usually /dev/video0)
camera = cv2.VideoCapture(0)

# Set properties for frame capture
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

is_recording = True
frames_per_second = 2  # Capture 2 frames per second

# Function to upload and delete images
def upload_and_delete(file_path):
    image_filename = os.path.basename(file_path)
    try:
        print(f"Uploading {image_filename} to S3...")
        s3.upload_file(file_path, S3_BUCKET, f"images/{image_filename}")
        print(f"Image successfully uploaded to S3: images/{image_filename}")

        # Delete the local file after a successful upload
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Local image file '{image_filename}' removed.")
    except Exception as e:
        # If there's an error, do not delete the local file and log the error
        print(f"Error uploading image '{image_filename}': {str(e)}. File will not be deleted locally.")

# Start capturing images
def start_capturing():
    global is_recording
    print("Starting image capture (2 frames per second)...")

    while is_recording:
        frame_delay = 1 / frames_per_second  # Time between each frame capture
        capture_start_time = time.time()

        for _ in range(frames_per_second):  # Capture 2 frames per second
            ret, frame = camera.read()
            if ret:
                # Get current timestamp for the image filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save each frame as an image file in the temp directory with timestamp
                image_file = os.path.join(temp_dir, f"image_capture_{timestamp}.jpg")
                cv2.imwrite(image_file, frame)

                # Upload and delete the image if successful
                upload_thread = threading.Thread(target=upload_and_delete, args=(image_file,))
                upload_thread.start()

                # Show the captured frame (optional)
                cv2.imshow('Capturing Frames...', frame)

                # Stop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopping capture by pressing 'q'...")
                    is_recording = False
                    break
            else:
                print("Failed to capture frame")
                break

        # Wait for the rest of the second to complete the frame delay
        time.sleep(max(0, frame_delay - (time.time() - capture_start_time)))

    # Gracefully exit
    camera.release()
    cv2.destroyAllWindows()

# Start capturing images
start_capturing()
