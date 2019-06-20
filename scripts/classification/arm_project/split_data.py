from imutils import paths
import argparse
import requests
import os.path
from pathlib import Path
import csv
import shutil
import random
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True,
                help="path to file containing data")
ap.add_argument("-tr", "--train", required=True,
                help="path to folder containing training images")
ap.add_argument("-te", "--test", required=True,
                help="path to file containing testing images")
ap.add_argument("-n", "--numb", required=True,
                help="number of training images")
args = vars(ap.parse_args())

resize_shape=(1000,1000)
data = open(args["data"] + "data.txt", "w")
imageDataPaths = list(paths.list_images(args["data"]))
data_list = []
for imageDataPath in imageDataPaths:
    print(imageDataPath)
    label0 = imageDataPath.split(os.path.sep)[-1].replace(args["data"], "").replace(".jpg", " ")
    # print(label0)
    data_list.append(label0.replace(" ",""))
    data.write("{}\n".format(label0))
data.close()

random.seed(42)
training_list = random.sample(data_list, k=int(args["numb"]))
testing_list = []
for i in range(len(data_list)):
    if data_list[i] not in training_list:
        testing_list.append(data_list[i])

print("Start copy training images\n")
for i in range(len(training_list)):
    url_src_data = args["data"] + str(training_list[i]) + str(".jpg")
    url_dst_train = args["train"] + str(training_list[i]) + str(".jpg")
    print(url_src_data)
    print(url_dst_train)
    shutil.copy(url_src_data, url_dst_train)
    origin = cv2.imread(url_dst_train)
    resize = cv2.resize(origin, resize_shape, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(url_dst_train, resize)
train = open(args["train"] + "train.txt", "w")
imageTrainPaths = list(paths.list_images(args["train"]))
for imageTrainPath in imageTrainPaths:
    label0 = imageTrainPath.split(os.path.sep)[0].replace(args["train"], "").replace(".jpg", " ")
    train.write("{}\n".format(label0))
train.close()

print("Start copy testing images\n")
for i in range(len(testing_list)):
    url_src_data = args["data"] + str(testing_list[i]) + str(".jpg")
    url_dst_test = args["test"] + str(testing_list[i]) + str(".jpg")
    print(url_src_data)
    print(url_dst_test)
    shutil.copy(url_src_data, url_dst_test)
    origin = cv2.imread(url_dst_test)
    resize = cv2.resize(origin, resize_shape, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(url_dst_test, resize)
test = open(args["test"] + "test.txt", "w")
imageTestPaths = list(paths.list_images(args["test"]))
for imageTestPath in imageTestPaths:
    label0 = imageTestPath.split(os.path.sep)[0].replace(args["test"], "").replace(".jpg", " ")
    test.write("{}\n".format(label0))
test.close()