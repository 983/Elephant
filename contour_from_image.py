import cv2
import numpy as np

path = "fancy_elephant.png"

image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, image = cv2.threshold(image, 127, 255, 0)

im2, contours, hierarchy = cv2.findContours(
    image,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)

x = contours[0][:, 0, 0]*1.0
y = contours[0][:, 0, 1]*1.0
x -= np.mean(x)
y -= np.mean(y)
x /= np.std(x)
y /= np.std(y)
y = -y

points = np.stack([x, y], axis=1)

np.save("points_fancy_elephant.npy", points)
