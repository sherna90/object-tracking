# import necessary libraries
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import numpy as np
from utils import Rectangle, Detection

class FasterRCNN:
    def __init__(self, gpu = 0):
        # get the pretrained model from torchvision.models
        # Note: pretrained=True will get the pretrained weights for the model.
        # model.eval() to use the model for inference
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        # Class labels from official PyTorch documentation for the pretrained model
        # Note that there are some N/A's 
        # for complete list check https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        # we will use the same list for this notebook
        self.CLASSES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.INTEREST_CLASSES = ['__background__', 'person','bicycle']

    def get_prediction(self,img, threshold):
        """
        get_prediction
            parameters:
            - img_path - path of the input image
            - threshold - threshold value for prediction score
            method:
            - Image is obtained from the image path
            - the image is converted to image tensor using PyTorch's Transforms
            - image is passed through the model to get the predictions
            - class, box coordinates are obtained, but only prediction score > threshold
                are chosen.
            
        """
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        pred = self.model([img])
        pred_class = [self.CLASSES[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        pred_score = pred_score[:pred_t+1]
        ind=[i for i,x in enumerate(pred_class) if x in self.INTEREST_CLASSES]
        pred_class = [pred_class[i] for i in ind]
        pred_boxes = [pred_boxes[i] for i in ind]
        pred_score = [pred_score[i] for i in ind]
        return pred_boxes, pred_class,pred_score


    def detect(self, im,  threshold=0.5, verbose = False):
        pred_boxes, pred_class,pred_score = self.get_prediction(im,  threshold)
        #all_boxes = [[] for _ in self.CLASSES]
        detections = []
        for i in range(len(pred_boxes)):
            (x1,y1) = pred_boxes[i][0] 
            (x2,y2) = pred_boxes[i][1]
            score = pred_score[i]
            new_det = Detection(int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1)), score)
            detections.append(new_det)
            if verbose:
                format_str = ('%d,%d,%d,%d,%.3f')
                print(format_str % (x1,y1,x2-x1,y2-y1,score))
        return detections