import streamlit as st
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import replicate
import json
import numpy as np
import subprocess
from utils import get_coords
from utils import show_plot
from utils import get_init_frame
from utils import upload_to_s3
from utils import yolo_inference
import toml

# print(st.secrets)

os.environ['REPLICATE_API_TOKEN'] = st.secrets["REPLICATE_API_TOKEN"]

# print(os.environ['REPLICATE_API_TOKEN'])

def process_data(file, input_text):

	if file is not None:
		file_path = os.path.join("uploads", file.name)
		with open(file_path, "wb") as f:
			f.write(file.read())
		st.write(f"File uploaded and stored at: {file_path}")
	
	get_init_frame(file_path, os.path.join("uploads", "output_frames"))
	url = upload_to_s3(file_path, "qc-qa-demo", file_path.split("\\")[-1])
	parse = yolo_inference(file_path, os.path.join("uploads", "output_frames"), input_text, url)
	all, object = 0, 0
	# st.write(parse["json_str"])
	data = json.loads(parse["json_str"])
	stride = 0
	for frame, det in data.items():
		all += 1
		if 'gloves' in str(det):
			object += 1
		if stride % 10 == 0:
			plot = get_coords(det)
			show_plot(os.path.join("uploads", "output_frames", f"{stride:05d}.jpg"), plot)
		stride += 1

	percentage = object / all * 100
	st.write(percentage)
	if percentage > 80:
		st.success(f"{input_text} adequately present; Object percentage: {percentage}%")
	else:
		st.error(f"{input_text} not adequately present; Object percentage: {percentage}%", icon="🚨")
  
def main():
	file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])
 
	input_text = st.text_input("Enter your text")
	
	if st.button("Submit"):
		process_data(file, input_text)

if __name__ == "__main__":
	main()