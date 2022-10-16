import cv2


model = "E:/PyCharm/PRLM_Lab_47138/MobileNetSSD_deploy.caffemodel"
txt = "E:/PyCharm/PRLM_Lab_47138/MobileNetSSD_deploy.prototxt.txt"
# specify the path of the pre-trained MobileNet SSD model
# use this model to detect the objects in a new image
net = cv2.dnn.readNet(model,txt)

# the pre-trained model can detect a list of object classes,
# so we define those classes in a dictionary and a list
categories = {0: 'background', 1: 'airplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
              9: 'chair', 10: 'cow', 11: 'dining-table', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
              16: 'potted-plant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tv-monitor'}

# defined in list also
classes = ["background", "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "dining-table",  "dog", "horse", "motorbike", "person", "potted-plant", "sheep", "sofa", "train", "tv-monitor"]

print("Done importing : The pretrained model \n The classess and categories\n")