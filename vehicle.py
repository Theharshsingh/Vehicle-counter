import cv2 as cv
import numpy as np
import time

# Web camera/video input
cap = cv.VideoCapture('video.mp4')

count_line_position = 550  # position of the counting line
min_width_react = 80       # minimum width of the rectangle to be considered a vehicle
min_height_react = 80      # minimum height of the rectangle to be considered a vehicle

# Speed detection parameters
PIXELS_PER_METER = 12.0    # Calibrate this based on your camera distance (pixels per meter)
CALIBRATION_DISTANCE = 10  # Distance between two lines in meters (line1 to line2)
line1_position = 550       # First speed detection line (before counting line)
line2_position = 650       # Second speed detection line (after counting line)

# Initialize background subtractor
algo = cv.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Data structures for tracking
detect = []
tracked_vehicles = {}  # {center: {'id': int, 'line1_time': float, 'line2_time': float, 'speed': float}}
vehicle_id_counter = 0
offset = 6
counter = 0

def calculate_speed(line1_time, line2_time):
    """Calculate speed in km/h"""
    time_diff = line2_time - line1_time
    if time_diff > 0:
        speed_mps = CALIBRATION_DISTANCE / time_diff  # meters per second
        speed_kmh = speed_mps * 3.6  # convert to km/h
        return round(speed_kmh, 1)
    return 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width = frame.shape[:2]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3,3), 5)
    
    # Background subtraction and processing
    img_sub = algo.apply(blur)
    dilated = cv.dilate(img_sub, np.ones((5,5)))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    dilatada = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)
    dilatada = cv.morphologyEx(dilatada, cv.MORPH_CLOSE, kernel)
    counterShape, _ = cv.findContours(dilatada, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    current_time = time.time()
    
    # Draw speed detection lines
    cv.line(frame, (25, line1_position), (1200, line1_position), (0, 255, 255), 3)  # Line 1 (Yellow)
    cv.line(frame, (25, line2_position), (1200, line2_position), (255, 0, 255), 3)  # Line 2 (Magenta)
    cv.line(frame, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)  # Counting line (Orange)
    
    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv.boundingRect(c)
        validate_counter = (w >= min_width_react) and (h >= min_height_react)
        
        if not validate_counter:
            continue
        
        center = center_handle(x, y, w, h)
        
        # Check if vehicle is already tracked
        vehicle_detected = False
        for existing_center, data in list(tracked_vehicles.items()):
            # Distance threshold for matching same vehicle
            dist = np.sqrt((center[0] - existing_center[0])**2 + (center[1] - existing_center[1])**2)
            if dist < 50:  # Same vehicle within 50 pixels
                tracked_vehicles[center] = tracked_vehicles.pop(existing_center)
                vehicle_detected = True
                break
        
        if not vehicle_detected:
            tracked_vehicles[center] = {
                'id': vehicle_id_counter,
                'line1_time': None,
                'line2_time': None,
                'speed': 0
            }
            vehicle_id_counter += 1
        
        # Draw bounding box and center
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.circle(frame, center, 4, (0, 0, 255), -1)
        
        # Speed detection logic
        vehicle_data = tracked_vehicles[center]
        vehicle_id = vehicle_data['id']
        
        # Check Line 1 crossing
        if (vehicle_data['line1_time'] is None and 
            line1_position - offset < center[1] < line1_position + offset):
            vehicle_data['line1_time'] = current_time
        
        # Check Line 2 crossing
        elif (vehicle_data['line2_time'] is None and 
              line2_position - offset < center[1] < line2_position + offset):
            vehicle_data['line2_time'] = current_time
        
        # Calculate speed when both lines are crossed
        if vehicle_data['line1_time'] is not None and vehicle_data['line2_time'] is not None:
            vehicle_data['speed'] = calculate_speed(vehicle_data['line1_time'], vehicle_data['line2_time'])
        
        # Vehicle counting at counting line
        if (center[1] < (count_line_position + offset) and 
            center[1] > (count_line_position - offset) and 
            center not in detect):
            counter += 1
            cv.line(frame, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
            print(f"Vehicle #{counter} - Speed: {vehicle_data['speed']} km/h")
            detect.append(center)
        
        # Display vehicle ID and speed
        label = f"ID:{vehicle_id} S:{vehicle_data['speed']}kmh"
        cv.putText(frame, label, (x, y-20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Cleanup old detections (vehicles that moved out of frame)
    to_remove = []
    for center, data in tracked_vehicles.items():
        if center[1] > height or data['speed'] > 0:  # Remove if out of frame or speed calculated
            to_remove.append(center)
    for center in to_remove:
        del tracked_vehicles[center]
    
    # Display statistics
    cv.putText(frame, f"VEHICLE COUNT: {counter}", (450, 70), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv.putText(frame, f"TOTAL VEHICLES TRACKED: {vehicle_id_counter}", (450, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv.imshow('Vehicle Speed Detection', frame)
    
    if cv.waitKey(33) == 27:  # ESC to exit
        break

cv.destroyAllWindows()
cap.release()
print(f"\nFinal Results:")
print(f"Total Vehicles Counted: {counter}")
print(f"Total Vehicles Tracked: {vehicle_id_counter}")