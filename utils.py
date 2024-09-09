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

def get_init_frame(video_path, output_path):
	# Create the output directory if it doesn't exist
	os.makedirs(os.path.join(output_path), exist_ok=True)

	# Open the video file
	video = cv2.VideoCapture(video_path)

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
	st.write(glove_instances)
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
    
	
		
    
    
    
    return output