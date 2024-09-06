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
from utils import yolo_inference
import toml

with open(".streamlit/secrets.toml", "r") as f:
    secrets = toml.load(f)

os.environ['REPLICATE_API_TOKEN'] = secrets["REPLICATE_API_TOKEN"]


def process_data(checkbox1, checkbox2, file, input_text):
	if checkbox1:
		st.write("Checkbox 1 is selected")
	if checkbox2:
		st.write("Checkbox 2 is selected")
	if file is not None:
		file_path = os.path.join("uploads", file.name)
		with open(file_path, "wb") as f:
			f.write(file.read())
		st.write(f"File uploaded and stored at: {file_path}")
	
	get_init_frame(file_path, os.path.join("uploads", "output_frames"))
	parse = yolo_inference(file_path, os.path.join("uploads", "output_frames"), input_text)
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
		st.success(f"Object percentage: {percentage}%")
	else:
		st.warning(f"Object percentage: {percentage}%")
  
def main():
	# Your existing code here
	# ...
	file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])
	
	checkbox1 = st.checkbox("Check for gloves")
	checkbox2 = st.checkbox("Check for overall cleanliness")
 
	input_text = st.text_input("Enter your text")
	
	if st.button("Submit"):
		process_data(checkbox1, checkbox2, file, input_text)

if __name__ == "__main__":
	main()