import numpy as np
import cv2
import os

from LoadModel import net,categories,classes

file = "E:/PyCharm/PRLM_Lab_47138/Datasets/Trained/"
image_counter = 0
every_image = []

for get_image_from_folder in os.listdir(file):
    image = cv2.imread(os.path.join(file, get_image_from_folder))  # inserting every images in the list
    if image is not None:  # if there is an image append the image in the list
        every_image.append(image)

for image in every_image:
    (h, w) = image.shape[:2]

    # MobileNet requires fixed dimensions for all input images, so first i resize
    # the image to 300x300 pixels and then normalize it
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)  # feeding the scaled image to the network
    detections = net.forward() # finally we will get the name of the detected object with confidence scores

    colors = np.random.uniform(255, 0, size=(len(categories), 3))  # select random colors for the bounding boxes

    # iterating over all the detection results and discard
    # any output whose confidence/probability is less than 0.2
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # getting the confidence score

        if confidence > 0.2:  # checking if the confidence is less than 0.2
            idx = int(detections[0, 0, i, 1])  # get the index of the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # locate the position of detected object in an image
            print("... class: ", classes[idx], ", confidence score: ", confidence)

            (startX, startY, endX, endY) = box.astype("int")  # get the coordinate of bounding box
            label = "{}: {:.2f}%".format(classes[idx], confidence * 100)  # set label and confidence score
            cv2.rectangle(image, (startX, startY), (endX, endY), colors[idx], 4)  # create a rectangular box around the object

            y = startY - 15 if startY - 15 > 15 else startY + 15  # set position of text which is written on bounding box
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)  # write label of the detected box
    # displaying the input image with detected objects
    cv2.imshow("Output", image)
    cv2.waitKey(2000)
    cv2.imwrite("E:/PyCharm/PRLM_Lab_47138/Datasets/{:.0f}.jpg".format(image_counter), image)
    image_counter += 1

cv2.destroyAllWindows()
