import os
import sys
import time
import json
import base64
from pathlib import Path
import warnings

import requests
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# --- Configuration & Paths ---

def get_app_paths():
    """
    Determines application paths for both script and PyInstaller bundle execution.
    Returns (base_dir, bundle_dir)
    """
    if getattr(sys, 'frozen', False):
        # Running as a PyInstaller bundle
        base_dir = Path(sys.executable).resolve().parent
        bundle_dir = Path(sys._MEIPASS)
    else:
        # Running as a script
        base_dir = Path(__file__).resolve().parent
        bundle_dir = base_dir
    return base_dir, bundle_dir

BASE_DIR, BUNDLE_DIR = get_app_paths()

# Load environment variables from .env file located next to the executable or script
load_dotenv(dotenv_path=(BASE_DIR / ".env"))

AILAB_KEY = os.getenv("AILABAPI_KEY")
AGE_API_URL = 'https://www.ailabapi.com/api/portrait/effects/face-attribute-editing'

# Define paths for I/O and data files
INPUT_DIR = BASE_DIR / "input_images"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BUNDLE_DIR / "model"
MODEL_PATH = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"
DB_PATH = BUNDLE_DIR / "db.json"

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# --- Global Variables ---
model = None
class_names = []
mbti_database = {}

# --- Core Functions ---

def load_all_models_and_data():
    """Load the Keras model, labels, and MBTI database."""
    global model, class_names, mbti_database
    
    print("Loading resources...")
    
    # 1. Load Keras Model
    try:
        from tensorflow.keras.models import load_model
        # Disable experimental features that might cause issues on some systems
        model = load_model(MODEL_PATH, compile=False)
        print("Keras model loaded successfully.")
    except Exception as e:
        print(f"Error loading Keras model: {e}", file=sys.stderr)
        print("Please ensure TensorFlow is installed correctly ('pip install tensorflow-cpu') and the model file exists.", file=sys.stderr)
        sys.exit(1)
        
    # 2. Load Labels
    try:
        with open(LABELS_PATH, "r") as f:
            class_names = [line.strip().split(' ', 1)[1] for line in f]
        print("Labels loaded successfully.")
    except Exception as e:
        print(f"Error loading labels.txt: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 3. Load MBTI Database
    try:
        with open(DB_PATH, 'r', encoding='utf-8') as f:
            mbti_database = json.load(f)
        print("MBTI database loaded successfully.")
    except Exception as e:
        print(f"Error loading or parsing db.json: {e}", file=sys.stderr)
        sys.exit(1)

    print("All resources loaded. Ready to process images.")

def analyze_mbti(image_path):
    """
    Analyze the image to predict MBTI type using the loaded Keras model.
    """
    if not model:
        return None
        
    try:
        # Create the array of the right shape to feed into the keras model
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        
        image = Image.open(image_path).convert("RGB")
        size = (224, 224)
        image = image.resize(size)

        # Turn the image into a numpy array
        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Get MBTI info from the database
        mbti_info = mbti_database.get("한국어", {}).get(class_name, {})
        
        return {
            "type": class_name,
            "confidence": confidence_score,
            "description": mbti_info.get("성격", "정보 없음"),
            "partner": mbti_info.get("배우자", "정보 없음"),
            "famous": mbti_info.get("유명인", "정보 없음"),
        }
    except Exception as e:
        print(f"Error during MBTI analysis for {image_path}: {e}", file=sys.stderr)
        return None

def get_aged_face(image_path):
    """
    Send the image to the AI LAB API to get an aged version.
    """
    try:
        with open(image_path, "rb") as image_file:
            files = {
                'image': (image_path.name, image_file, 'application/octet-stream'),
                'action_type': (None, 'TO_OLD'),
            }
            headers = {'ailabapi-api-key': AILAB_KEY}
            
            response = requests.post(AGE_API_URL, files=files, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            data = response.json()

            if data.get("error_code") and data["error_code"] != 0:
                print(f"AI LAB API Error: {data.get('error_msg', 'Unknown error')}", file=sys.stderr)
                return None
            
            return data.get("result", {}).get("image") # Base64 encoded image string

    except requests.exceptions.RequestException as e:
        print(f"Error calling AI LAB API: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred in get_aged_face: {e}", file=sys.stderr)
        return None


def process_image(image_path):
    """
    Main processing function for a new image.
    """
    print(f"\n--- New image detected: {image_path.name} ---")
    time.sleep(1) # Wait a moment to ensure file is fully written

    base_filename = image_path.stem

    # 1. MBTI Analysis
    print("1. Starting MBTI analysis...")
    mbti_result = analyze_mbti(image_path)
    if mbti_result:
        mbti_output_path = OUTPUT_DIR / f"{base_filename}_mbti.txt"
        output_content = (
            f"MBTI 예측 결과: {mbti_result['type']} (정확도: {mbti_result['confidence']:.2%})\n\n"
            f"[성격]\n{mbti_result['description']}\n\n"
            f"[추천하는 배우자]\n{mbti_result['partner']}\n\n"
            f"[유명인]\n{mbti_result['famous']}"
        )
        with open(mbti_output_path, "w", encoding="utf-8") as f:
            f.write(output_content)
        print(f"   -> MBTI result saved to: {mbti_output_path.name}")
    else:
        print("   -> MBTI analysis failed.")

    # 2. Face Aging Analysis
    print("2. Starting face aging analysis...")
    aged_face_b64 = get_aged_face(image_path)
    if aged_face_b64:
        aged_face_path = OUTPUT_DIR / f"{base_filename}_aged.jpg"
        try:
            img_data = base64.b64decode(aged_face_b64)
            with open(aged_face_path, 'wb') as f:
                f.write(img_data)
            print(f"   -> Aged face image saved to: {aged_face_path.name}")
        except Exception as e:
            print(f"   -> Failed to decode or save aged face image: {e}", file=sys.stderr)
    else:
        print("   -> Face aging analysis failed.")
    
    print(f"--- Processing complete for {image_path.name} ---")


# --- File System Watcher ---

class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            # Check for common image file extensions
            if event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                process_image(Path(event.src_path))

def start_watcher():
    """Start watching the input directory for new files."""
    # Ensure directories exist
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, str(INPUT_DIR), recursive=False)
    observer.start()
    
    print(f"Watching for new images in: {INPUT_DIR}")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# --- Main Execution ---

if __name__ == "__main__":
    if not AILAB_KEY:
        print("Error: AILABAPI_KEY not found in .env file.", file=sys.stderr)
        sys.exit(1)
        
    load_all_models_and_data()
    start_watcher()
