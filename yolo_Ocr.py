
# python yolo.py --image images/img.jpg --yolo file_model


import numpy as np
import argparse
import time
import cv2
import os
import sys

class Object(object):
	def __init__(self, text ,point1, point2):
		self.text = text 
		self.point1 = point1
		self.point2 = point2
		
		

class ObjectDetect(object):
	
	def __init__ (self):
		# config_yolo= "/home/vsmart/10/labelImg/model_Gun/yolov3.cfg"
		# weight_yolo=  "/home/vsmart/10/labelImg/model_Gun/yolov3_70000.weights"
		# self.labelsPath= "/home/vsmart/10/labelImg/model_Gun/full_gkcmp.names"

		config_yolo= "/home/vsmart/Downloads/traffic_detector/yolov3.cfg"
		weight_yolo=  "/home/vsmart/Downloads/traffic_detector/yolov3.weights"
		self.labelsPath= "/home/vsmart/Downloads/traffic_detector/obj.names"

		self.net = cv2.dnn.readNetFromDarknet(config_yolo,weight_yolo)
		
		self.ln = self.net.getLayerNames()
		self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

	def detect(self,preFilePath):
		image= cv2.imread(preFilePath)

		LABELS= open(self.labelsPath).read().strip().split("\n") 
		res = []
		(H, W) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (608, 608), swapRB=True, crop=False)
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

				
				if confidence > 0.44:
					
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					print ("Conf", LABELS[classID], confidence)
					classIDs.append(classID)

					
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.44, 0.5)
		print (len(classIDs))
		if len(idxs) > 0:
			# print (len(idxs))
			
			for i in idxs.flatten():
				
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				text = "{}".format(LABELS[classIDs[i]])
				print ("xxxyyy", text)

				obj = Object(text, (x, y ), (w , h ))
				res.append(obj)
				# print ("xxxxx")
		return res

if __name__ == '__main__':
	input_image = "/home/vsmart/VSM/alpr_train/labelImg/square/0_0_lp.png"
	detectObject = ObjectDetect()
	res = detectObject.detect(input_image)
	print (len(res), res[1].text, res[0].point1, res[0].point2)