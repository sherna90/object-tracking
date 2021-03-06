import numpy as np
import cv2
import argparse as ap
from context import tracking
from tracking.utils.image_generator import MTTImageGenerator as ImageGenerator
from tracking.models.gmphd_filter import GMPHDFilter
from frcnn_pytorch import FasterRCNN

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    group = parser.add_argument_group()
    group.add_argument('-i', '--images', help = 'Path to images folder', required = True)
    group.add_argument('-g', '--groundtruth', help = 'Path to groundtruth file', required = True)
    group.add_argument('-d', '--detections', help = 'Path to detections file')
    args = vars(parser.parse_args())
    if args['images']:
    
        if args['detections']:
            generator = ImageGenerator(args['images'], args['groundtruth'], args['detections'])
        else:
            generator = ImageGenerator(args['images'], args['groundtruth'])
        
        verbose = False
        draw = True
        
        if draw:
            cv2.namedWindow('MTT', cv2.WINDOW_NORMAL)
        
        filter = GMPHDFilter(verbose)

        if not(args['detections']):
            detector = FasterRCNN()
        idx = 1
        for i in range(generator.get_sequences_len()):
            img = generator.get_frame(i)
            gt = generator.get_groundtruth(i)
            
            if args['detections']:
                detections = generator.get_detections(i)
                #features = generator.get_features(i)
            else:
                detections = detector.detect(img,threshold=0.5)
            
            estimates = []
            if not filter.is_initialized():
                filter.initialize(img, detections)
                estimates = filter.estimate(img, draw = draw)
                #filter.draw_particles(img)
            else:
                filter.predict()
                filter.update(img, detections, verbose = verbose,calcHist=True)
                estimates = filter.estimate(img, draw = draw)
                #filter.draw_particles(img)
            
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