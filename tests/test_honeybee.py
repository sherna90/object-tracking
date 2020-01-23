import numpy as np
import cv2
import argparse as ap
from context import tracking
from tracking.models.gmphd_filter import GMPHDFilter
from tracking.utils.utils import *

def read_detections():
    entries = os.listdir('rec1_detections/')
    detections=dict()
    for i,e in enumerate(entries):
        f=open('rec1_detections/'+e,'r')
        detections[i]=[]
        for line in f:
            dat=line.split(',')
            detections[i].append([int(dat[0]),int(dat[1])])
        f.close()
    return detections

def find_foreground_objects(background_model):
    thresh = cv2.threshold(background_model,50, 255, cv2.THRESH_BINARY)[1]
    #thresh = cv2.dilate(thresh, None, iterations=1)
    #thresh = cv2.erode(thresh, None, iterations=2)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def detect(background_model,frame):
    mask = background_model.apply(frame)
    foreground_objects = find_foreground_objects(mask)
    detections=[]
    for c in foreground_objects:
        area=cv2.contourArea(c)
        if area<50 or area>5000:
            continue
        (x,y,width,height) = cv2.boundingRect(c)
        det=Detection(x, y, width, height, conf = None, feature = None)
        detections.append(det)
    return detections

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    group = parser.add_argument_group()
    group.add_argument('-v', '--video', help = 'Path to video file', required = True)
    args = vars(parser.parse_args())
    if args['video']:
        verbose = False
        draw = True  
        if draw:
            cv2.namedWindow('MTT', cv2.WINDOW_NORMAL)
        
        filter = GMPHDFilter(verbose)
        cap = cv2.VideoCapture(args['video'])
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        ret, img = cap.read()
        background_model = cv2.createBackgroundSubtractorMOG2()
        idx = 1
        while(cap.isOpened()):
            ret, img = cap.read()
            detections=detect(background_model,img)
            estimates = []
            if not filter.is_initialized():
                filter.initialize(img, detections)
                estimates = filter.estimate(img, draw = draw)
            else:
                filter.predict()
                filter.update(img, detections, verbose = verbose)
                estimates = filter.estimate(img, draw = draw)
            if not(verbose):
                if estimates is not None:
                    for e in estimates:
                        print(str(idx) + ',' + str(e.label) + ',' + str(e.bbox.x) + ',' + str(e.bbox.y) + ','\
                        + str(e.bbox.width) + ',' + str(e.bbox.height)\
                        + ',1,-1,-1,-1')
                        #print type(e.bbox.width)
                        #if (e.bbox.width == 0) or (e.bbox.height == 0):
                        #    exit()
                idx+=1
            else:
                for det in detections:
                    #cv2.rectangle(img, (det.bbox.x, det.bbox.y), (det.bbox.x + det.bbox.width, det.bbox.y + det.bbox.height), (255,0,0), 3)
                    print(str(idx) + ','  + str(det.bbox.x) + ',' + str(det.bbox.y) + ','\
                        + str(det.bbox.width) + ',' + str(det.bbox.height)\
                        + ',1,-1,-1,-1')
                idx+=1
            if draw:
                cv2.imshow('MTT', img)
                cv2.waitKey(1)
        
    #cv2.destroyWindow('MTT')