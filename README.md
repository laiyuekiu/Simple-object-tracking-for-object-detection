# Simple object tracking for object detection
An object tracking for different object detection models like YOLO

Base on object detection model detection results to do the tracking, tracking by the bounding box coordinate

Comparing the previous and current frame detection result to track the objects.
This object tracking works on multi classes but it tracks object coordinate only that it will keep tracking even the class name suddenly changed.
<br/>
<br/>

----------------------------

obj_track.py is a python module that can be directly import to use.

obj_track.py requires 3 input, bbox, track_bbox_list, image:
1. bbox: a list of all current detection result (bounding box coordinate), the list should following in this format: [[xmin,ymin,xmax,ymax, ...], [xmin,ymin,xmax,ymax, ...], ...] <br/>
(obj_id will be assigned to be the last index)
2. track_bbox_list: a list of previous detection result (bbox), just input a empty list, this module will do the rest of it
3. image: input the image you want to draw the result on

The bounding box coordinate should be in pixel unit and the point of top-left and the point of bottom-right.
<br/>
<br/>
Can refer to below tracking cars demo to see the implementation.

----------------------------
A full demo of tracking cars by using Yolov4 Pytorch model, car_track_demo.py

The model of Yolov4 Pytorch is developed by @Tianxiaomo and this demo combines the Yolov4 Pytorch and the object tracking
https://github.com/Tianxiaomo/pytorch-YOLOv4

_Car types counting is included_

**Installation:**
```
1. git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git then "pip install -r requirements.txt"
2. Install Vidgear (for retrieving video)
3. prepare your yolov4 weights
4. copy car_track_demo.py to the yolov4 pytorch directory
```
[You may follow VidGear to pip install](https://abhitronix.github.io/vidgear/latest/installation/pip_install/)\
[You may follow AlexeyAB yolov4 training instruction to train your own weights](https://github.com/AlexeyAB/darknet)


**Usage:**
```
python car_track_demo.py [-cfgfile ___your_cfg_file___] [-namefile ___your_name_file___] [-weightfile ___your_weight_file___] 
[-video ___your_input_video___] [-output ___your_output_file___] [-show_img] [-stream_on]

### e.g. python car_track_demo.py -cfgfile ./cfg/car_type.cfg -namefile ./data/car_type.names -weightfile ./data/car_type_best.weights -video ./data/car_type/test_car.mp4 -output ./car_detected.mp4
### -show_img, it will prompt a opencv window (cv2.imshow()) to show the detecting 
### -stream_on, it is for real-time streaming (RTSP/HTTP) or streaming video to reconnect the stream when receiving broken frame
```

**Customization:**

You may adjust the tracking accepting range. <br/>
In car_track_demo.py **Line 130**
```
if abs(i[0] - n[0]) <= 150 and abs(i[1] - n[1]) <= 150 and abs(i[2] - n[2]) <= 150 and abs(i[3] - n[3]) <= 150: 
```
Adjust the minus result(150). if your object moves faster, you need a larger value. A slower object need a smaller value.
<br/>
<br/>
<br/>
You may adjust the image boundary for stop tracking when the object is near the boundary. <br/>
In car_track_demo.py **Line 113**
```
if m[0]-0 <= 40 or m[1]-0 <= 40 or img_w-m[2] <= 100 or img_h-m[3] <= 100:
```
----------------------------------------------
If you need a better object tracking, you can have a look Deep SORT / SORT \
[Deep SORT](https://github.com/nwojke/deep_sort)\
[SORT](https://github.com/abewley/sort)
