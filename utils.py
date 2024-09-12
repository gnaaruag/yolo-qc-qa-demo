import json
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import replicate
import streamlit as st
from PIL import Image
import boto3
from botocore.exceptions import NoCredentialsError
from datetime import timedelta


def get_init_frame(video_path, output_path):
	# Create the output directory if it doesn't exist
	os.makedirs(os.path.join(output_path), exist_ok=True)

	# Open the video file
	video = cv2.VideoCapture(video_path)
	fps = video.get(cv2.CAP_PROP_FPS)
	print(f"Frames per second: {fps}")

	# Initialize frame counter
	frame_count = 0

	# Read frames from the video
	while True:
		# Read the next frame
		ret, frame = video.read()
		
		# Break the loop if no more frames are available
		if not ret:
			break

		output_file = os.path.join(output_path, f"{frame_count:05d}.jpg")
		cv2.imwrite(output_file, frame)
		
		# Increment the frame counter
		frame_count += 1

	# Release the video file
	video.release()
	return fps

def get_coords(coords):
	data = coords
	glove_instances = []
	for key, value in data.items():
		if value['cls'] == 'gloves':
			x0, y0, x1, y1 = value['x0'], value['y0'], value['x1'], value['y1']
			center_x = (x0 + x1) // 2
			center_y = (y0 + y1) // 2
			glove_instances.append([center_x, center_y])
	glove_instances = np.array(glove_instances)
	# st.write(glove_instances)
	return glove_instances

def show_plot(frame_path, coords):
	plt.figure(figsize=(12, 8))
	plt.title(f"{frame_path}")
	plt.imshow(Image.open(frame_path))
	if coords.ndim == 1:
		coords = coords.reshape(-1, 2)
	plt.scatter(coords[:, 0], coords[:, 1], marker='*', color='red')
	st.pyplot(plt)
 
def upload_to_s3(local_file, bucket, s3_file):
	s3 = boto3.client('s3')
	try:
		s3.upload_file(local_file, bucket, s3_file)
		file_url = f"https://{bucket}.s3.ap-south-1.amazonaws.com/{s3_file}"
		print(f"File successfully uploaded to {file_url}")
		return file_url
	except FileNotFoundError:
		print("The file was not found.")
		return None
	except NoCredentialsError:
		print("Credentials not available.")
		return None
	

def yolo_inference(video_path, output_path, input_text, input_media):
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.join(output_path), exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Initialize frame counter
    print(input_media)
    input = {
		"input_media": input_media,
		"class_names": input_text,
		"return_json": True,
		"score_thr": 0.3
	}
    
    output = replicate.run(
		"zsxkib/yolo-world:d232445620610b78671a7f288f37bf3baec831537503e9064afcf0bfd0f0a151",
		input=input
	)
    print(output)
    return output


def merge_time_ranges(timestamps):
    # Convert "MM:SS" to total seconds
    def time_to_seconds(time_str):
        minutes, seconds = map(int, time_str.split(":"))
        return minutes * 60 + seconds

    # Convert total seconds back to "MM:SS"
    def seconds_to_time(seconds):
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02}:{seconds:02}"

    # Merge overlapping or continuous time ranges
    def merge_ranges(ranges):
        # Sort ranges by their start time
        ranges.sort()

        # Initialize the merged list
        merged = [ranges[0]]

        for current in ranges[1:]:
            prev_start, prev_end = merged[-1]
            current_start, current_end = current

            # Merge if overlapping or continuous
            if current_start <= prev_end + 1:
                merged[-1] = (prev_start, max(prev_end, current_end))
            else:
                merged.append(current)

        return merged

    # Step 1: Convert all timestamp ranges to seconds
    time_ranges = []
    for t in timestamps:
        if "-" in t:
            start, end = t.split("-")
            time_ranges.append((time_to_seconds(start), time_to_seconds(end)))
        else:
            # Single timestamps are treated as a range from (t, t)
            time_in_seconds = time_to_seconds(t)
            time_ranges.append((time_in_seconds, time_in_seconds))

    # Step 2: Merge the ranges
    merged_ranges = merge_ranges(time_ranges)

    # Step 3: Convert the merged ranges back to "MM:SS"
    merged_timestamps = [f"{seconds_to_time(start)}-{seconds_to_time(end)}" for start, end in merged_ranges]

    return merged_timestamps


