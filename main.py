import cv2

# Load pre-trained model
vehicle_cascade = cv2.CascadeClassifier('vehicle_cascade.xml')

def classify_vehicle(shape):
    # This is a simplified example, in a real-world application you would
    # use a machine learning model to classify the vehicles based on their shape
    if shape < 500:
        return 'motorcycle'
    elif shape < 1000:
        return 'car'
    elif shape < 1500:
        return 'minibus'
    elif shape < 2000:
        return 'bus'
    else:
        return 'truck'

def count_vehicles(video_path):
    # Open video file
    video = cv2.VideoCapture(video_path)

    vehicle_counts = {
        'motorcycle': 0,
        'car': 0,
        'minibus': 0,
        'bus': 0,
        'truck': 0
    }

    while True:
        # Read video frame
        ret, frame = video.read()

        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect vehicles
        vehicles = vehicle_cascade.detectMultiScale(gray, 1.1, 4)

        # Classify and count vehicles
        for (x, y, w, h) in vehicles:
            vehicle_type = classify_vehicle(w * h)
            vehicle_counts[vehicle_type] += 1

    # Release video file
    video.release()

    return vehicle_counts

if __name__ == "__main__":
    video_path = 'vehicles.mp4'
    vehicle_counts = count_vehicles(video_path)
    print(vehicle_counts)