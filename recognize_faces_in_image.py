#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:19:26 2020

@author: princematt
"""


import face_recognition
import argparse
import pickle
import cv2

#  I'm going to use command line arguments
#To run this code use :  python recognize_faces_in_image.py --encodings encodings.pickle --image examples/example_04.png 
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("Loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load the input image and convert it from BGR to RGB
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y)-coordinates of the bounding boxes corresponding
print("Recognizing faces...")
boxes = face_recognition.face_locations(rgb,
	model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# initialize the list of names for each face detected
names = []

for encoding in encodings:
	# matching each face in the input image to our knownencodings
	matches = face_recognition.compare_faces(data["encodings"],
		encoding)
	name = "Unknown"

	# check to see if we have found a match
	if True in matches:
		# finding the total number of times each image was matched in the dict
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}

		# looping over the matched indexes and maintain a count for each recognized face 
		for i in matchedIdxs:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1

		# getting the number of the highest recognition 
		name = max(counts, key=counts.get)
	
	# update the list of names
	names.append(name)

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
	# draw the predicted face name on the image
	cv2.rectangle(image, (left, top), (right, bottom), (255, 25, 2), 2)
	y = top - 10 if top - 10 > 15 else top + 10
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (25, 255, 70), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)