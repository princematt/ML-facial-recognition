#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:19:26 2020

@author: princematt
"""
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

#  i'm going to use command line arguments
#To run this code use :  "python encode_faces.py --dataset dataset --encodings encodings.pickle"
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")#choosing deep learning (cnn) for better accuracy
args = vars(ap.parse_args())

print("Quantifying faces...")
Paths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

for (i, Path) in enumerate(Paths):
	# extract the person name from the image path
	print("Processing image {}/{}".format(i + 2,
		len(Paths)))
	name = Path.split(os.path.sep)[-2]
    #dlib uses rgb. Therefore, we have to convert to RGB
	image = cv2.imread(Path)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# detect the (x, y)-coordinates of the bounding boxes
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	# compute the facial embedding for the face
	encodings =  face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for i in encodings:
		# add each encoding + name to our set of known names and encodings
		knownEncodings.append(i)
		knownNames.append(name)

# dump the facial encodings + names to disk
print("Serializing.")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()