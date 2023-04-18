# Step 1. Import Library and Model
from ultralytics import YOLO
import cv2
import numpy as np
# load a pretrained model
model = YOLO("yolov8n-seg.pt") 

image = cv2.imread("images/mug2.png")
img = image.copy()
height, width, channels = img.shape # (540,720,3)

# Step 2. Prediction
results = model.predict(img, 
                        conf=0.25, 
                        show=True, 
                        save=False, 
                        save_crop=False 
                        )

# Step 3. Working with Results
result = results[0]
res_plotted = result.plot()
cv2.imshow("result", res_plotted)
# Boxes(convert Tensor type to numpy)
boxes = np.array(result.boxes.xyxy.cpu(), dtype="int") # boxes with xyxy format, (N, 4)
print(boxes)
classes = np.array(result.boxes.cls.cpu(), dtype="int") # classes for N boxes, (N, 1)
print(classes)
scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2) # confidence score, (N, 1)
print(scores)
# Masks
masks = []

for mask in result.masks.xy: # A list of contour pixels
    # convert array type from float to int
    mask = np.array(mask, dtype=np.int32)
    masks.append(mask)

'''
print(masks[0])
sorted_list = sorted(masks[0], key=lambda x: (x[1], x[0]))
print(sorted_list)
my_list = [list(item) for item in sorted_list]

print(my_list)
'''
# Plot Masks on Image
colors = np.random.randint(0,255, (len(result.names),3))
print(colors)
for mask, class_id in zip(masks, classes):

    
    color = colors[class_id]

    cv2.polylines(img, [mask], True, (int(color[0]),int(color[1]),int(color[2])), 4)
    cv2.fillPoly(img, [mask], (int(color[1]),int(color[2]),int(color[0])))

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

overlay = cv2.addWeighted(img, 0.6, image, 0.4, 0)
res_img = np.concatenate((image,overlay), axis=1)
cv2.imshow("img", res_img)

cv2.waitKey(0)
cv2.destroyAllWindows()