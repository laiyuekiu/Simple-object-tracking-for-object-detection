# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    : Q
    @Time      : Mar 2021
    @Detail    : Car tracking and counting
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
import cv2
from vidgear.gears import VideoGear, WriteGear 
import datetime
import pytz
import os

"""hyper parameters"""
use_cuda = True


def detect_cv2_camera(cfgfile, weightfile):    
    global obj_id, count_car_list, class_names
    args = get_args()
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    stream = VideoGear(source=args.video).start()    ### use VideoGear to get stream

    ### set ffmpeg output parameter here, refer to WriteGear for details
    #output_params = {"-vcodec": "mjpeg", "-crf": 51, "-q:v": 31, "-preset": "veryslow", "-update": 1, "-vf": "scale=1024:768,fps=30/60"} ###-update: 1 must as to replace the file
    #gear_writer = WriteGear(output_filename='./detect_result/cap.jpg', logging=True, **output_params)

    gear_writer = WriteGear(output_filename=args.output, logging=True)
    print("Starting the YOLO loop...")

    class_names = load_class_names(args.namefile)

    obj_id = 0   #### start count at 1, 0 for detecting the 1st frame
    track_bbox_list = []
    count_car_list = list(range(len(class_names)))
    count_car_list = [i*0 for i in count_car_list]

    while True:
        img = stream.read()

        if args.stream_on:
            if img is None:      ### in order to restart input stream when receving a broken/empty frame from stream. Video does not have this issue
                stream = VideoGear(source=args.video).start()
                img = stream.read()

        img_h, img_w, _ = img.shape
        #img = img[720:(720+abs(img_h-720)), 500:(500+abs(img_w-500))]  ### crop away those "far away" cars, only detect near cars, plz adjust your cutting
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))
        
        if len(boxes[0]) > 0:
            result_img, track_bbox_list = object_tracking(boxes[0], track_bbox_list, img)   ### doing tracking and counting
        else:
            result_img = img      ### avoid nth in detection, none bounding box (bbox). if nth, directly use the current frame

        gear_writer.write(result_img)


        if args.show_img:
            cv2.imshow('Yolo demo', result_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stream.stop()
                cv2.destroyAllWindows()
                gear_writer.close()
                break  



def object_tracking(bbox, track_bbox_list, image):
    global obj_id, count_car_list
    img_h, img_w, _ = image.shape
    result_img = image   ### avoid nth in bbox as those bbox may remove by below if statement, so copy input image
    temp_track_list = []
    temp_match_list = []
    temp_used_list = []
    temp_bbox_list = []

    for num in range(len(bbox)):
        bbox[num][0] = int(bbox[num][0] * img_w)     ### convert to normal int x1y1x2y2 coordinate
        bbox[num][1] = int(bbox[num][1] * img_h)
        bbox[num][2] = int(bbox[num][2] * img_w)
        bbox[num][3] = int(bbox[num][3] * img_h)


    temp_bbox_list = bbox.copy()
    for index, m in enumerate(bbox):
        if m[0]-0 <= 40 or m[1]-0 <= 40 or img_w-m[2] <= 100 or img_h-m[3] <= 100:   ### remove those bbox that are near the image top and bottom boundary as it will not track any more
            temp_bbox_list.remove(m)                                                 ### you may adjust those (40), (100) coordinate to set your boundary
    bbox = temp_bbox_list.copy()

    if obj_id == 0:    ### to skip the first bbox, as no track at first
        if len(bbox) > 0:
            track_bbox_list = bbox.copy()
            for num in range(len(track_bbox_list)):
                obj_id += 1 
                track_bbox_list[num].append(obj_id)
            count_car_list = count_vehicle(bbox, count_car_list)       ### start count at 1st frame too
        return image, track_bbox_list


    for num_i, i in enumerate(bbox):
        for num_n, n in enumerate(track_bbox_list):
            ### adjust the below minus coordinate(150), for fast moving object the minus shd larger
            if abs(i[0] - n[0]) <= 150 and abs(i[1] - n[1]) <= 150 and abs(i[2] - n[2]) <= 150 and abs(i[3] - n[3]) <= 150:  ### use abs() as the new bbox may larger a bit than old one
                if temp_used_list.count(num_i) == 0:
                    min_aera = abs(i[0] - n[0]) + abs(i[1] - n[1]) + abs(i[2] - n[2]) +abs(i[3] - n[3])
                    temp_match_list.append([num_n, min_aera, i])        ### temp_match_list: [-index of track_bbox_list-, -min_aera-, [-bbox coord-]]
                    temp_used_list.append(num_i)                        ### avoid track_bbox_list double use bbox coord, one bbox for one track_bbox_list use only

    ################
    ### This object track is based on calculate bbox coordinate only, will not care about the object detection class_name/class_id
    ################


    for i in temp_match_list:
        temp_track_list.append(i[2])                               ### update the latest bbox coordinate
        temp_track_list[-1].append(track_bbox_list[i[0]][-1])      ### get the index from "temp_match_list[i][0]" for finding its obj_id and append its obj_id
        bbox.remove(i[2])                                          ### remove the used bbox coordinate


    for n in bbox:
        temp_track_list.append(n)              ### append the remain bbox coord which is new detection
        obj_id += 1                            ### as the patten is add one to obj_id first before append()
        temp_track_list[-1].append(obj_id)     ### append obj_id to the latest append bbox coord

    count_car_list = count_vehicle(bbox, count_car_list)     ### as only count those new detection
             
    track_bbox_list = temp_track_list.copy()

    for track_bbox in track_bbox_list:
        x1 = track_bbox[0]
        y1 = track_bbox[1]
        x2 = track_bbox[2]
        y2 = track_bbox[3]
        result_img = cv2.putText(image, str(track_bbox[-1])+' '+class_names[track_bbox[6]], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2) ### the track_bbox[] last index is the obj_id
        result_img = cv2.rectangle(result_img, (x1, y1), (x2, y2), (0,255,0), 2)   ### set the detection bbox color

    result_img = cv2.putText(result_img, str(count_car_list), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)   ### show car types counting

    return result_img, track_bbox_list
    

def count_vehicle (bbox, count_car_list):
    if len(bbox) > 0:                                                     ## count_car_list format is based on your model.names (class_id)
        for i in bbox:                                                    ## e.g. 2 class_id, count_car_list = [-class_0: private_car-, -class_1: truck-] (total len is 2 and with counting number only)
            for num_m, m in enumerate(count_car_list):
                if i[6] == num_m:
                    count_car_list[num_m] += 1
    return count_car_list


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-video', type=str, default='Input your video/stream path',
                        help='path of your video file or stream url', dest='video')
    parser.add_argument('-show_img', action='store_true',
                        help='show frames during streamming/video', dest='show_img')
    parser.add_argument('-namefile', type=str,
                        default='your darknet name file, ./data/x.names',
                        help='path of your darknet name file', dest='namefile')
    parser.add_argument('-output', type=str, default='Path to save image, need put "/" at the end to be directory',                      
                        help='path of saving your image file. need put "/" at the end to be directory', dest='output')
    parser.add_argument('-store_log', action='store_true',
                        help='Output detect log txt.', dest='store_log')
    parser.add_argument('-stream_on', action='store_true',
                        help='For streaming video/camera, reconnect stream when receving broken frame', dest='stream_on')


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    detect_cv2_camera(args.cfgfile, args.weightfile)
