#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 20:00:29 2021

@author: Vinayaka Vivekananda Malgi <vmalgi@deakin.edu.au>

Deakin University, Melbourne, Victoria - Australia

"""

# Check for GPU

# Uncomment the below comment block if you are using Apple Mac (For Apple M1, please install separate version of tensorflow).

'''

import tensorflow as tf
if tf.test.gpu_device_name():
   print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

#Set ML compute to GPU in case of Mac OS

#https://github.com/apple/tensorflow_macos

#Import mlcompute module to use the optional set_mlc_device API for device selection with ML Compute.
from tensorflow.python.compiler.mlcompute import mlcompute

# Select CPU device.
mlcompute.set_mlc_device(device_name='gpu') # Available options are 'cpu', 'gpu', and ‘any'.

'''


# Importing the libraries

import numpy as np
import tensorflow as tf
import cv2

# Reading the class labels from coco.names because Yolo is trained on COCO dataset

with open('/home/vinayaka/Downloads/YOLO_model/Config and weights/coco.names') as f:
    class_labels = [line.strip() for line in f]

    
# Initializing writer to none
writer = None

#Readding the video file from the file location

videostream = cv2.VideoCapture('/home/vinayaka/Downloads/YOLO_model/Test images/mumbai-traffic.mp4')

while (videostream.isOpened()):
    #getting the current frames from the video stream
    ret,current_frame = videostream.read()
    #break if no frame is retrieved 
    if not ret:
        break
    #we get the current frame instead of image
    img_to_detect = current_frame
    img_height = img_to_detect.shape[0]
    img_width = img_to_detect.shape[1]
    # Converting the image to blob to pass the blob image into the model
    img_blob = cv2.dnn.blobFromImage(img_to_detect,1/255.0,(416,416),swapRB=True,crop=False)

    #Declare list of colors as an array
    #Green, Red, Blue, Cyan, Yellow, Purple
    #Splitting is based on , and for every split convert it into an integer
    #Converting that to a numpy array to apply color mask to image numpy array
    
    class_colors = ["0,255,0","0,0,255","255,0,0","0,0,255","255,255,0","0,255,255"]
    class_colors = [np.array(every_color.split(",")).astype('int') for every_color in class_colors]
    class_colors = np.array(class_colors)                        
    class_colors = np.tile(class_colors,(16,1))
    
    #Loading the pretrained model
    #Input the processed blob into the model and pass through the model
    #Obtain the detection predictions by using the forward() method
    
    yolo_model = cv2.dnn.readNetFromDarknet('/home/vinayaka/Downloads/YOLO_model/Config and weights/yolov3.cfg','/home/vinayaka/Downloads/YOLO_model/Config and weights/yolov3.weights')
    
    # Getting all the layers of the pre-trained yolo model
    # Find the last layer of the yolo model
    
    yolo_layers = yolo_model.getLayerNames()
    yolo_output_layer = [yolo_layers[yolo_layer[0] - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]
    
    #Input the pre processed blob into the model and pass through the model
    
    yolo_model.setInput(img_blob)
    
    #Obtain the detection layers by passing it through till the output layer
    
    obj_detection_layers = yolo_model.forward(yolo_output_layer)
    
    #NMS suppression to remove the less confidence bounding boxes
    #initialization for non-max supression
    #declare the list for class_id,bounding boxes and confidence scores list
    
    class_ids_list = []
    boxes_list = []
    confidences_list = []
    
    #loop over each of the layer outputs
    for object_detection_layer in obj_detection_layers:
        #loop over object detection layer
        for object_detection in object_detection_layer:
            all_scores = object_detection[5:] #object detection[1:4] will two center points box width and box height
            predicted_class_id = np.argmax(all_scores) #will have scores for all objects within bounding boxes
            predicted_confidence = all_scores[predicted_class_id]
            
            #taking only the predictions with confidence score > 20%
            if(predicted_confidence > 0.20):
                #getting the predicted class label
                predicted_class_label = class_labels[predicted_class_id]
                #obtaining the bounding boxes for actual image from the predicted images
                bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype('int')
                start_x_pt = int(box_center_x_pt - (box_width / 2))
                start_y_pt = int(box_center_y_pt - (box_height / 2))
                
                #Saving class_id, starting x and y, width and height,confidences in a list for nms processing
                #Passing confidence as float whereas height and width are integers
                
                class_ids_list.append(predicted_class_id)
                confidences_list.append(float(predicted_confidence))
                boxes_list.append([start_x_pt,start_y_pt, int(box_width), int(box_height)])
                
        #Applying the NMS will return only the selected max ids while suppressing the weak overlappings
        #Non maxima suppression confidence set as 0.5 and max suppression threshold for NMS as 0.4(the values can be adjusted)
        max_values_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
        
        #Looping through the final set of detections remaining after NMS and draw bounding box and write text
        for max_valueid in max_values_ids:
            max_class_id = max_valueid[0]
            box = boxes_list[max_class_id]
            start_x_pt = box[0]
            start_y_pt = box[1]
            box_width = box[2]
            box_height = box[3]
            
            #getting the predicted class id and the label
            predicted_class_id = class_ids_list[max_class_id]
            predicted_class_label = class_labels[predicted_class_id]
            predicted_confidence = confidences_list[max_class_id]
                
            end_x_pt = start_x_pt + box_width
            end_y_pt = start_y_pt + box_height
    
            #get a random mask color from numpy array of colors
            box_colors = class_colors[predicted_class_id]
    
            #convert the color numpy array as a list and apply to text boxes
            box_color = [int(c) for c in box_colors]
    
            #print the prediction in console
            predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, predicted_confidence*100)
            print("Predicted Object {}".format(predicted_class_label))
    
            # draw rectangle and text in the image
            cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
            cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5),cv2.FONT_HERSHEY_COMPLEX, 0.7, box_color, 2)
        
        cv2.imshow("Detection Output", img_to_detect)
        
        #### Saving the processed video to a directory
        
        if writer is None:
            
            # Constructing code of the codec
            # to be used in the function VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Writing current processed frame into the video file
        
            writer = cv2.VideoWriter('/home/vinayaka/Downloads/mumbai_traffic_processed_1.mp4', fourcc, 30,
                                 (current_frame.shape[1], current_frame.shape[0]), True)

        # Write processed current frame to the file
        writer.write(current_frame)
        
        #terminate the while loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
#Releasing the video stream and the camera
#closing all the opencv() windows
videostream.release()
writer.release()
