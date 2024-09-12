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
from utils import merge_time_ranges

# print(st.secrets)

os.environ['REPLICATE_API_TOKEN'] = st.secrets["REPLICATE_API_TOKEN"]

# print(os.environ['REPLICATE_API_TOKEN'])


def process_data(file, input_text):

	if file is not None:
		file_path = os.path.join("uploads", file.name)
		with open(file_path, "wb") as f:
			f.write(file.read())
		st.write(f"File uploaded and stored at: {file_path}")
	
	fps = get_init_frame(file_path, os.path.join("uploads", "output_frames"))
	url = upload_to_s3(file_path, "qc-qa-demo", file_path.split("\\")[-1])
	parse = yolo_inference(file_path, os.path.join("uploads", "output_frames"), input_text, url)
	all, object = 0, {}
	# st.write(parse["json_str"])
	data = json.loads(parse["json_str"])
	stride = 0
	violations = {}
 
	for frame, det in data.items():
		all += 1
		for text in input_text.split(', '):
			if text in str(det):
				print(text)
				print(str(det))
				if text not in object:
					object[text] = 1
				else:
					object[text] += 1
			else:
				fno = int(frame.split("-")[1])
				if text not in violations:
					violations[text] = [fno]
				else:
					violations[text].append(fno)
		# else:
			# plot = get_coords(det)
			# violations.append(frame)
			# show_plot(os.path.join("uploads", "output_frames", f"{stride:05d}.jpg"), plot)

		stride += 1
	# st.write(object)
	percentage = {key: value / all * 100 for key, value in object.items()}
	# st.write(percentage)
	# st.write(violations)
	ranges = []
	start = None
	for key, value in violations.items():
		start = None
		ranges = []
		for i in range(len(value)):
			if start is None:
				start = value[i]
			if i + 1 < len(value) and value[i + 1] == value[i] + 1:
				continue
			else:
				if start == value[i]:
					ranges.append("{:02d}:{:02d}".format(int(start/fps/60), int(start/fps%60)))
				else:
					ranges.append("{:02d}:{:02d}-{:02d}:{:02d}".format(int(start/fps/60), int(start/fps%60), int(value[i]/fps/60), int(value[i]/fps%60)))
				start = None
		# st.write(ranges)
		warn = f"Violation ranges for {key}: {', '.join(merge_time_ranges(ranges))}"
		st.warning(warn)
	for key, value in percentage.items():
		if value > 60:
			st.success(f"{key} adequately present; Object percentage: {value:.2f}%", icon="✅")
		else:
			st.error(f"{key} not adequately present; Object percentage: {value:.2f}%", icon="🚨")

	# st.write(percentage)
	# if percentage > 80:
	# 	st.success(f"{input_text} adequately present; Object percentage: {percentage}%")
	# else:
	# 	st.error(f"{input_text} not adequately present; Object percentage: {percentage}%", icon="🚨")
  
def main():
	st.title("Your AI Quality Assurance Agent")
	file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])

	gloves = st.checkbox("Gloves")
	hat = st.checkbox("Hat")
	shoes = st.checkbox("Shoes")
	masks = st.checkbox("Masks")
	watch = st.checkbox("Watch")

	input_text = ""
	if gloves:
		input_text += "gloves, "
	if hat:
		input_text += "hat, "
	if shoes:
		input_text += "shoes, "
	if masks:
		input_text += "masks, "
	if watch:
		input_text += "watch, "
  
	submit_button = st.button("Submit")

	input_text = input_text.rstrip(", ")
 
	if file is not None and input_text != "":
		with st.spinner("checking for policy breaches..."):
			if submit_button:
				process_data(file, input_text)
	

if __name__ == "__main__":
	main()