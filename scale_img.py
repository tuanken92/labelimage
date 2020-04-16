import numpy as np
import os, sys, cv2
import glob
import xml.etree.ElementTree as ET

def ShowGroundTruth(img_path1, anno_path,im):
	colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],[0, 255, 85], [255, 85, 0], [85, 255, 255],[150, 0, 0], [0, 150, 0], [0, 0, 150]]
	# Reading annotation infor
	print (anno_path)
	cls='bienxe'
	font = cv2.FONT_HERSHEY_SIMPLEX    	
	tree = ET.parse(anno_path)
	root = tree.getroot()
	for size_imgae in root.findall('size'):
		size_w = size_imgae.find('width').text = str(1350)
		size_h = size_imgae.find('height').text = str(450)
	for obj in root.findall('object'):			
		cls = obj.find('name').text
		#print name
		bndbox= obj.find('bndbox')
					
		xmin = int(bndbox.find('xmin').text) 
		ymin = int(bndbox.find('ymin').text) 
		xmax = int(bndbox.find('xmax').text) 
		ymax = int(bndbox.find('ymax').text)
		bndbox.find('xmin').text = str(xmin)
		bndbox.find('ymin').text = str(ymin)
		bndbox.find('xmax').text = str(xmax)
		bndbox.find('ymax').text = str(ymax)

		print ("box: (({},{}),({},{}))".format(xmin,ymin,xmax,ymax))
		#cv2.rectangle(im,(xmin, ymin), (xmax, ymax), [255,255,2555], 2)
		#cv2.rectangle(im,(xmin, ymin), (xmax, ymax), [0,255,0], 2)
	
		#cv2.putText(im,cls,(xmin, ymin-2), font, 0.5, [255,255,255], 1) 	
	# if (cls==1 or cls==2):
	# 	return True
	# else:
	# 	return False

if __name__ == '__main__':    
	anno_dir="/home/vsmart/Works/labelImg/oto/image/image1/"	
	frame_dir="/home/vsmart/Works/labelImg/oto/image/image1/"
	#cv2.namedWindow("preview")
	anno_paths=glob.iglob(os.path.join(anno_dir, "*.xml"))
	anno_paths=list(anno_paths)
	np=len(anno_paths)
	print("Number of samples: ",np)
	i=0;
	deleted=0;
	while i <np:		
		anno_path=anno_paths[i]	
		if not os.path.isfile(anno_path):
			i=i+1
			continue		  
		anno, ext = os.path.splitext(os.path.basename(anno_path))			
		print ('Processing {:d}/{:d} {:s}...'.format(i,np,anno))
		inFile = open(anno_path, 'r')
		data = inFile.readlines()
		inFile.close()
		outFile = open("/home/vsmart/Works/Yolo-detect/1/frame%d.xml" % i, 'w')
		outFile.writelines(data)
		outFile.close()		
		img_path=frame_dir+ anno +'.jpg'   
		print (img_path)           
		frame = cv2.imread(img_path) 
		
		r=ShowGroundTruth(img_path, anno_path, frame)
		cv2.imwrite("/home/vsmart/Works/Yolo-detect/1/frame%d.png" % i, frame)            
		#cv2.imshow("preview", frame_crop)
			           
		if r==True:
			key = cv2.waitKey()
		else:
			key = cv2.waitKey(1)

		if key == 27: # exit on ESC				
			break
		if key==100: # press d to remove bad sample
			print('Remove {:s}'.format(anno))
			os.remove(anno_path)
			os.remove(img_path)
			deleted=deleted+1;
		if key==98:  # Back
			if i>0:
				i=i-1							
			continue
		i=i+1            
	#cv2.destroyWindow("preview")