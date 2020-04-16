
# python yolo.py --image images/img.jpg --yolo file_model


import numpy as np
import argparse
import time
import cv2
import os
import sys

class Object(object):
	def __init__(self, point1, point2):
		self.point1 = point1
		self.point2 = point2


class ObjectDetect(object):
	
	def __init__ (self):
		config_yolo= "/home/vsmart/data/10/labelImg/model_YOLO/yolov3.cfg"
		weight_yolo=  "/home/vsmart/data/10/labelImg/model_YOLO/yolov3.weights"
		self.labelsPath= "/home/vsmart/data/10/labelImg/model_YOLO/coco.names"
		self.net = cv2.dnn.readNetFromDarknet(config_yolo,weight_yolo)
		self.ln = self.net.getLayerNames()
		self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

	def detect(self,preFilePath):
		image= cv2.imread(preFilePath)
		LABELS= open(self.labelsPath).read().strip().split("\n") 
		res = []
		(H, W) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
		self.net.setInput(blob)
		layerOutputs = self.net.forward(self.ln)
		boxes = []
		confidences = []
		classIDs = []

		for output in layerOutputs:
			
			for detection in output:
				
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				
				if confidence > 0.5:
					
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					
					classIDs.append(classID)
					
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
		if len(idxs) > 0:
			# print (len(idxs))
			for i in idxs.flatten():
				print(i, classIDs)
				if LABELS[classIDs[i]]!="person":
					continue
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				obj = Object((x, y ), (w , h ))
				res.append(obj)
				# print ("xxxxx")
		return res

if __name__ == '__main__':
	input_image = "/home/pvmuoi/Downloads/labelImg/demo/00008.png"
	detectObject = ObjectDetect()
	res = detectObject.detect(input_image)
	print (res[0].point1, res[0].point2)
