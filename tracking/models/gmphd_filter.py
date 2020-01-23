from utils import Target, Rectangle, cost_matrix, nms
from tracking.dpp.dpp import DPP
from tracking.models.kalman_filter import KalmanFilter
#from resnet import Resnet
import scipy.stats as stats
from scipy.optimize import linear_sum_assignment
from lapsolver import solve_dense
import random
import numpy as np
import cv2

class GMPHDFilter:
    DIM = 4
    POS_STD_X = 3.0
    POS_STD_Y = 3.0
    SCALE_STD_WIDTH = 3.0
    SCALE_STD_HEIGHT = 3.0
    THRESHOLD = 100.0
    SURVIVAL_RATE = 1.0
    SURVIVAL_DECAY = 0.7
    #CLUTTER_RATE = 2.0
    BIRTH_RATE = 0.8
    #DETECTION_RATE = 0.5
    #POSITION_LIKELIHOOD_STD = 30.0
    verbose = False
    initialized = False

    def __init__(self, verbose = False):
        self.verbose = verbose
        self.tracks = []
        self.labels = []
        self.birth_model = []


    def is_initialized(self):
        return self.initialized

    def reinitialize(self):
        self.initialized = False

    def initialize(self, img, detections = None, calcHist = False):
        (self.img_height, self.img_width, self.n_channels) = img.shape
        self.tracks = []
        if len(detections) > 0 and detections:
            for idx, det in enumerate(detections):
                target = Target(det.bbox, idx, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), det.conf, self.SURVIVAL_RATE, det.feature)
                self.tracks.append(target)
                self.labels.append(idx)
            self.initialized = True
        else:
            self.initialized = False

    def predict(self):
        if self.is_initialized():
            predicted_tracks = []
            for track in self.tracks:
                track.predict()
                if np.random.uniform() < track.survival_rate:
                    predicted_tracks.append(track)
            for track in self.birth_model:
                predicted_tracks.append(track)
            self.tracks = predicted_tracks

    def update(self, img, detections = None, verbose = False, calcHist = False):
        self.birth_model = []
        if self.is_initialized() and len(detections) > 0 and detections:
            new_detections = []
            for idx, det in enumerate(detections):
                target = Target(bbox = det.bbox, color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)),\
                    conf = det.conf, survival_rate = self.SURVIVAL_RATE, feature = det.feature)
                new_detections.append(target)
            new_tracks = []
            if len(self.tracks) > 0:
                diagonal = np.sqrt( np.power(self.img_height, 2) + np.power(self.img_width, 2) )
                area = self.img_height * self.img_width
                cost = cost_matrix(self.tracks, new_detections, diagonal, area, False)
                #tracks_ind, new_dets_ind = linear_sum_assignment(cost)
                tracks_ind, new_dets_ind = solve_dense(cost)
                dets_high_cost = set()
                for idxTrack, idxNewDet in zip(tracks_ind, new_dets_ind):
                    if cost[idxTrack, idxNewDet] < self.THRESHOLD:
                        new_detections[idxNewDet].label = self.tracks[idxTrack].label
                        new_detections[idxNewDet].color = self.tracks[idxTrack].color
                        self.tracks[idxTrack].update(new_detections[idxNewDet].bbox)
                        new_tracks.append(self.tracks[idxTrack])
                    else:
                        self.tracks[idxTrack].survival_rate = np.exp(self.SURVIVAL_DECAY * (-1.0 + self.tracks[idxTrack].survival_rate * 0.9))
                        new_tracks.append(self.tracks[idxTrack])
                        dets_high_cost.add(idxNewDet)
                
                tracks_no_selected = set(np.arange(len(self.tracks))) - set(tracks_ind)
                for idxTrack in tracks_no_selected:
                    #print str(new_tracks[idxTrack].bbox.x) + ',' + str(new_tracks[idxTrack].bbox.y) + ',' + str(new_tracks[idxTrack].bbox.width) + ',' + str(new_tracks[idxTrack].bbox.height)
                    self.tracks[idxTrack].survival_rate = np.exp(self.SURVIVAL_DECAY * (-1.0 + self.tracks[idxTrack].survival_rate * 0.9))
                    new_tracks.append(self.tracks[idxTrack])
                #print '###################'
                
                new_detections_no_selected = set(np.arange(len(new_detections))) - set(new_dets_ind)
                new_detections_no_selected = new_detections_no_selected | dets_high_cost
                for idxNewDet in new_detections_no_selected:
                    #print str(new_detections[idxNewDet].bbox.x) + ',' + str(new_detections[idxNewDet].bbox.y) + ',' + str(new_detections[idxNewDet].bbox.width) + ',' + str(new_detections[idxNewDet].bbox.height)
                    if np.random.uniform() > self.BIRTH_RATE:
                        new_label = max(self.labels) + 1
                        new_detections[idxNewDet].label = new_label
                        self.birth_model.append(new_detections[idxNewDet])
                        self.labels.append(new_label)
                #print '###################'
            else:
                for idxNewDet, det in enumerate(new_detections):
                    if np.random.uniform() > self.BIRTH_RATE:
                        new_label = max(self.labels) + 1
                        new_detections[idxNewDet].label = new_label
                        self.birth_model.append(new_detections[idxNewDet])
                        self.labels.append(new_label)

            #dpp = DPP()
            #self.tracks = dpp.run(boxes = new_tracks, img_size = (self.img_width, self.img_height),features=None)
            #self.tracks = nms(new_tracks, 0.7, 0, 0.5)
            self.tracks = new_tracks


    def estimate(self, img = None, draw = False, color = (0, 255, 0)):
        if self.initialized:
            if draw:
                for track in self.tracks:
                    #track.bbox.print()
                    cv2.rectangle(img, (track.bbox.x, track.bbox.y), (track.bbox.x + track.bbox.width, track.bbox.y + track.bbox.height), track.color, 3)
                    y = track.bbox.y - 15 if track.bbox.y - 15 > 15 else track.bbox.y + 15
                    cv2.putText(img, str(track.label), (track.bbox.x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, track.color, 2)
            return self.tracks