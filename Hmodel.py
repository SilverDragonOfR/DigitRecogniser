from PIL import Image
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import mnist
import joblib

#training
X_train = mnist.train_images()
y_train = mnist.train_labels()

#testing
X_test = mnist.test_images()
y_test = mnist.test_labels()

#reshaping each image to one list of pixels
X_train = X_train.reshape((-1,28*28))
X_test = X_test.reshape((-1,28*28))

#normalizing to 0-1 range
X_train = X_train/256
X_test = X_test/256

#making the model
model = MLPClassifier(solver="adam",activation="relu",hidden_layer_sizes=(500,500))

#training the model with time taken
model.fit(X_train,y_train)

#saving model
joblib.dump(model,"Hmodel.joblib")
print("done")
