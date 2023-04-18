## Computer Vision Notes 03 | Apply the YOLOv8 algorithm for object detection and segmentation in pictures

> This is the 3rd article in the 「Computer Vision Notes」 series, which records how to apply the YOLOv8 algorithm for object detection and segmentation in pictures, and implement the drawing of Masks and Bounding Boxes through the OpenCV library.
> - Video:
> - Code:
### 0. Classify, Detect and Segment Objects
![Classfication, Detection, Segmentation](https://files.mdnice.com/user/1474/4ec68f91-55b9-4790-ad84-7ed11d99c1eb.png)

*Image classification* involves classifying an entire image into one of a set of predefined classes.
- **Output:** a *single class label* and a *confidence score*.  

*Object detection* involves identifying the location and class of objects in an image or video stream.
- **Output:** a *set of bounding boxes* that enclose the objects in the image, along with *class labels* and *confidence scores* for each box

*Instance segmentation* goes a step further than object detection and involves **identifying individual objects in an image** and **segmenting them from the rest of the image**.
- **Output:** a *set of masks or contours that outline each object* in the image, along with *class labels* and *confidence scores* for each object.


### Step 1. Import Libraries, Model and Image
Make sure we have updated the latest version of `ultralytics` 
- in this note, we use `ultralytics 8.0.81` that was released in Apr 16, 2023
```bash
$ pip install ultralytics==8.0.81
```

![](https://files.mdnice.com/user/1474/1f53f560-ca14-4ee0-91ce-24122548d4fc.png)

```py
# Step 1. Import Libraries, Model and Image
from ultralytics import YOLO
import cv2
import numpy as np

# load a pretrained model
model = YOLO("yolov8n-seg.pt") 

image = cv2.imread("images/mj.jpg")
img = image.copy()

```
![Models download automatically from the latest Ultralytics release on first use.](https://files.mdnice.com/user/1474/17b54a93-2e25-488a-ad0a-33625b29f3a7.png)

### Step 2. Prediction
`model.predict()` accepts **multiple arguments** that control the prediction operation

![All Supported Arguments](https://files.mdnice.com/user/1474/074e46fe-b822-4ab4-b506-4717cda039b5.png)

```py
# Step 2. Prediction
results = model.predict(img, 
                        conf=0.5, 
                        show=True, 
                        save=True, 
                        save_crop=True 
                        )
```
![](https://files.mdnice.com/user/1474/924765c0-069f-4e29-b8c0-01bdb9c2abb0.png)


### Step 3. Working with Results
Each `result` object contains the components such as:
- `result.boxes` with properties and methods for manipulating **bounding boxes**
- `result.masks` for indexing masks or getting segment coordinates, where each mask is a binary image
- `result.probs` containing class probabilities


![Source Code](https://files.mdnice.com/user/1474/57c3668e-89c5-491d-ad59-cdd78533fc8e.png)


Each `result` is composed of a `torch.Tensor` by default, which allows to be converted to `numpy` for easy manipulation

```python
# Boxes
# xyxy: points for (left, top) and (right, bottom, )
boxes = np.array(result.boxes.xyxy.cpu(), dtype="int") 
# N boxes with xyxy format, (N, 4)
classes = np.array(result.boxes.cls.cpu(), dtype="int") 
# N classes for N boxes, (N, 1)
scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2) 
# N confidence scores, (N, 1)

# Masks
masks = []

for mask in result.masks.xy: # N masks
    # convert each mask from float to int
    mask = np.array(mask, dtype=np.int32)
    masks.append(mask)
```


### 4. Plot Masks and Boxes by OpenCV
#### Masks
```python
# Plot Masks on Image
colors = np.random.randint(0,255, (len(result.names),3))
print(colors)
for mask, class_id in zip(masks, classes):
    color = colors[class_id]
    
    cv2.polylines(img, [mask], True, (int(color[0]),int(color[1]),int(color[2])), 4)
    cv2.fillPoly(img, [mask], (int(color[1]),int(color[2]),int(color[0])))
# Plot Boxes on Image
# ...
# Show the Result
res_img = np.concatenate((image,img), axis=1)
cv2.imshow("img", res_img)
```

![](https://files.mdnice.com/user/1474/7701f773-bcd4-48a5-b760-d5e41c6eb99b.png)

#### Boxes
```python
# Plot Boxes on Image
for box, class_id, score in zip(boxes, classes, scores):
    (left, top, right, bottom) = box # xyxy
    color = colors[class_id]
    cv2.rectangle(img = img, 
                pt1 = (left, top), 
                pt2 = (right, bottom), 
                color = (int(color[2]),int(color[1]),int(color[0])), 
                thickness = 2)


    cv2.putText(img = img, 
                text = str(result.names[class_id]) + " " + str(score), 
                org = (left + 10, top - 5), 
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = 0.3, 
                color = (int(color[0]),int(color[2]),int(color[1])), 
                thickness = 1)
# Show the Result
res_img = np.concatenate((image,img), axis=1)
cv2.imshow("img", res_img)
```

![](https://files.mdnice.com/user/1474/5eb92e0a-d9ec-4fcb-a976-d100e97b549e.png)

#### Add Transparency on Masks
```python
overlay = cv2.addWeighted(img, 0.6, image, 0.4, 0)
res_img = np.concatenate((image,overlay), axis=1)
cv2.imshow("img", res_img)
```

![](https://files.mdnice.com/user/1474/a3498dd3-dfbd-407b-846b-737351220231.png)
### Step 4. Stop the Pipeline
```python
cv2.waitKey(0)
cv2.destroyAllWindows()
```
### Resources
- https://docs.ultralytics.com/tasks/segment/
- https://docs.ultralytics.com/modes/predict/
- https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/engine/results.py
- https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html
- https://github.com/yiyangd/Computer-Vision-Notes/tree/main/03_YOLOv8_Detection_Segmentation
