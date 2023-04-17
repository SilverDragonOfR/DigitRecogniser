from PIL import Image
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import mnist
import time
import cv2
import joblib

#importing model using joblib
model = joblib.load("./Hmodel.joblib")

#opening our image
img = cv2.imread("./model_test.png",0)
img = img.reshape((-1,28*28))
img = img/256

#prediction
[prediction] = model.predict(img)
print("The ans is : ",prediction)
