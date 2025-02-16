import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import time
from collections import Counter
import math

# Load YOLOv5 model dynamically
st.sidebar.title('Traffic Signal Controls')
st.title('Dynamic Traffic Management System')

# Paths for traffic light images
red_light = 'red.png'
yellow_light = 'yellow.png'
green_light = 'green.png'

# Model loading section
path_model_file = st.sidebar.text_input('Path to YOLOv5 Model:', 'path/to/best.pt')
confidence = st.sidebar.slider('Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

if st.sidebar.checkbox('Load Model'):
    model = torch.hub.load('.', 'custom', path=path_model_file, source='local', force_reload=True)
    class_labels = model.names

    # Upload video files for each lane
    lane_1_file = st.sidebar.file_uploader("Upload Lane 1 Video", type=['mp4', 'avi', 'mkv'])
    lane_2_file = st.sidebar.file_uploader("Upload Lane 2 Video", type=['mp4', 'avi', 'mkv'])
    lane_3_file = st.sidebar.file_uploader("Upload Lane 3 Video", type=['mp4', 'avi', 'mkv'])

    # Temporary files for the videos
    if lane_1_file and lane_2_file and lane_3_file:
        tfile1 = tempfile.NamedTemporaryFile(delete=False)
        tfile1.write(lane_1_file.read())
        lane_1_cap = cv2.VideoCapture(tfile1.name)

        tfile2 = tempfile.NamedTemporaryFile(delete=False)
        tfile2.write(lane_2_file.read())
        lane_2_cap = cv2.VideoCapture(tfile2.name)

        tfile3 = tempfile.NamedTemporaryFile(delete=False)
        tfile3.write(lane_3_file.read())
        lane_3_cap = cv2.VideoCapture(tfile3.name)

        # Function to calculate green light duration based on vehicle count
        def calculate_green_time(vehicle_count):
            if vehicle_count >= 30:
                return 30
            elif 10 <= vehicle_count < 30:
                return 20
            else:
                return 10

        # Detect vehicles and handle each lane
        def detect_vehicles(cap, process_time=3):
            vehicle_count = 0
            ambulance_detected = False
            start_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform inference with YOLOv5
                results = model(frame)

                # Render the bounding boxes on the frame
                frame_with_boxes = np.array(results.render()[0])

                df = results.pandas().xyxy[0]  # YOLOv5 results in a dataframe format

                # Count vehicles and check for ambulance
                vehicle_count += df[df['name'].isin(['car', 'truck', 'bus', 'bike', 'auto'])].shape[0]
                ambulance_detected = ambulance_detected or ('ambulance' in df['name'].values)

                # Stop processing after the specified duration (3 seconds)
                if time.time() - start_time > process_time:
                    break

            # Return the frame with bounding boxes and vehicle/ambulance info
            return frame_with_boxes, vehicle_count / (process_time * 3), ambulance_detected

        # Process each lane and update signals with traffic light animations
        def update_traffic_signals():
            col1, col2, col3 = st.columns(3)

            traffic_light1 = col1.empty()
            traffic_light2 = col2.empty()
            traffic_light3 = col3.empty()

            video_placeholder1 = col1.empty()
            video_placeholder2 = col2.empty()
            video_placeholder3 = col3.empty()

            traffic_placeholder = st.empty()  # Placeholder for traffic updates

            while True:
                # Lane 1 traffic signal and video feed
                with col1:
                    traffic_light1.image(red_light, width=100)  # Display traffic light above the video feed
                    frame_1, lane_1_count, ambulance_1 = detect_vehicles(lane_1_cap)
                    video_placeholder1.image(frame_1, caption="Lane 1 Video Feed", use_column_width=True)
                    st.write(f"Lane 1 vehicle count: {math.ceil(lane_1_count)}")

                # Lane 2 traffic signal and video feed
                with col2:
                    traffic_light2.image(red_light, width=100)  # Display traffic light above the video feed
                    frame_2, lane_2_count, ambulance_2 = detect_vehicles(lane_2_cap)
                    video_placeholder2.image(frame_2, caption="Lane 2 Video Feed", use_column_width=True)
                    st.write(f"Lane 2 vehicle count: {math.ceil(lane_2_count)}")

                # Lane 3 traffic signal and video feed
                with col3:
                    traffic_light3.image(red_light, width=100)  # Display traffic light above the video feed
                    frame_3, lane_3_count, ambulance_3 = detect_vehicles(lane_3_cap)
                    video_placeholder3.image(frame_3, caption="Lane 3 Video Feed", use_column_width=True)
                    st.write(f"Lane 3 vehicle count: {math.ceil(lane_3_count)}")

                # Check for ambulances
                if ambulance_1 or ambulance_2 or ambulance_3:
                    with traffic_placeholder.container():
                        if ambulance_1:
                            st.write("Ambulance detected in Lane 1!")
                            traffic_light1.image(green_light, width=100)  # Set green light for Lane 1
                        if ambulance_2:
                            st.write("Ambulance detected in Lane 2!")
                            traffic_light2.image(green_light, width=100)  # Set green light for Lane 2
                        if ambulance_3:
                            st.write("Ambulance detected in Lane 3!")
                            traffic_light3.image(green_light, width=100)  # Set green light for Lane 3
                    time.sleep(30)  # Give time for the ambulance to pass
                else:
                    # Determine which lane gets the green light based on vehicle count
                    lane_counts = {"Lane 1": lane_1_count, "Lane 2": lane_2_count, "Lane 3": lane_3_count}
                    sorted_lanes = sorted(lane_counts.items(), key=lambda x: x[1], reverse=True)

                    with traffic_placeholder.container():
                        for lane, count in sorted_lanes:
                            green_time = calculate_green_time(count)

                            # Set appropriate traffic light for the lane with green signal
                            if lane == "Lane 1":
                                traffic_light1.image(green_light, width=100)  # Set green light for Lane 1
                                traffic_light2.image(red_light, width=100)
                                traffic_light3.image(red_light, width=100)
                            elif lane == "Lane 2":
                                traffic_light1.image(red_light, width=100)
                                traffic_light2.image(green_light, width=100)  # Set green light for Lane 2
                                traffic_light3.image(red_light, width=100)
                            elif lane == "Lane 3":
                                traffic_light1.image(red_light, width=100)
                                traffic_light2.image(red_light, width=100)
                                traffic_light3.image(green_light, width=100)  # Set green light for Lane 3

                            st.write(f"{lane} is green for {green_time} seconds (Vehicle Count: {math.ceil(count)})")
                            time.sleep(green_time)

        update_traffic_signals()