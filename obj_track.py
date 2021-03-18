import cv2

### e.g. result_img, track_bbox_list = object_tracking(-your object detection result-, track_bbox_list, img)
### your object detection result format should be at least the first 4 index is the bounding box coordinate (top-left, bottom-right), [xmin,ymin,xmax,ymax,.....]

def object_tracking(bbox, track_bbox_list, image):          ### bbox = current frame detection result, track_bbox_list = previous detection result
    global obj_id             ### plz declare a 'obj_id = 0' before calling this method
    img_h, img_w, _ = image.shape
    result_img = image        ### avoid nth in bbox as those bbox may remove by below if statement, so copy input image
    temp_bbox_list = []
    temp_track_list = []
    temp_match_list = []
    temp_used_list = []
    
    ### uncomment these if you are using yolo
    #for num in range(len(bbox)):                          
    #    bbox[num][0] = int(bbox[num][0] * img_w)          ### convert yolo normalization format to normal int coordinate
    #    bbox[num][1] = int(bbox[num][1] * img_h)
    #    bbox[num][2] = int(bbox[num][2] * img_w)
    #    bbox[num][3] = int(bbox[num][3] * img_h)

    temp_bbox_list = bbox.copy()
    for index, m in enumerate(bbox):
        if m[0]-0 <= 40 or m[1]-0 <= 40 or img_w-m[2] <= 100 or img_h-m[3] <= 100:   ### remove those bbox that are near the image top and bottom boundary as it will not track any more
            temp_bbox_list.remove(m)                                                 ### you may adjust those (40), (100) coordinate to set your boundary
    bbox = temp_bbox_list.copy()


    if obj_id == 0:     ### to skip the first bbox as no tracking at first
        if len(bbox) > 0:
            track_bbox_list = bbox.copy()                        ### copy all the bbox for next frame tracking
            for num in range(len(track_bbox_list)):
                obj_id += 1 
                track_bbox_list[num].append(obj_id)
        return image, track_bbox_list


    for num_i, i in enumerate(bbox):
        for num_n, n in enumerate(track_bbox_list):
            ### adjust the below minus coordinate(150), for fast moving object the minus shd larger
            if abs(i[0] - n[0]) <= 150 and abs(i[1] - n[1]) <= 150 and abs(i[2] - n[2]) <= 150 and abs(i[3] - n[3]) <= 150:  ### use abs() as the new bbox may larger or smaller than old one
                if temp_used_list.count(num_i) == 0:
                    min_aera = abs(i[0] - n[0]) + abs(i[1] - n[1]) + abs(i[2] - n[2]) +abs(i[3] - n[3])
                    temp_match_list.append([num_n, min_aera, i])        ### temp_match_list: [-index of track_bbox_list-, -min_aera-, [-bbox coord-]]
                    temp_used_list.append(num_i)                        ### avoid track_bbox_list double use bbox coordinate, one bbox for one track_bbox_list use only


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
        temp_track_list[-1].append(obj_id)     ### append obj_id to the latest append bbox coordinate

    track_bbox_list = temp_track_list.copy()


    for track_bbox in track_bbox_list:
        x1 = track_bbox[0]
        y1 = track_bbox[1]
        x2 = track_bbox[2]
        y2 = track_bbox[3]
        result_img = cv2.putText(image, str(track_bbox[-1]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2) ### the track_bbox[] last index is the obj_id
        result_img = cv2.rectangle(result_img, (x1, y1), (x2, y2), (0,255,0), 2)    ### set the detection bbox color

    return result_img, track_bbox_list
