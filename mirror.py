
from PIL import Image
from PIL import ImageTk

import math
import numpy as np
import matplotlib.pyplot as plt


# Image load
# image = Image.open('./Sierra16.jpg')
# image.show()
# print(image)

# matplot load
# image = plt.imread('./Sierra16.jpg')
# plt.imshow(image)


def loadImage(path):
    image = Image.open(path)
    # image.show()
    return image


def imageToMatrix(image):
    matrix = np.asarray(image)
    # print(matrix)
    return matrix


def matrixToImage(matrix):
    image = Image.fromarray(matrix)
    return image


# If there are 0 - 1 values in the image matrix, 
# or there are floating point numbers, then the 
# fromarray method of Image will not be able to
# convert the matrix into an image
def matrixToImages(matrix):
    # convert matrix values into 
    # data types supported by fromarray
    image = Image.fromarray((matrix * 255).astype(np.uint8))
    return image


mirrorlevel    = 0
mirrorvertical = 1


def imageMirror(image, direction=mirrorlevel):
    imagematrix = imageToMatrix(image)

    height = imagematrix.shape[0]
    width = imagematrix.shape[1]
    channel = imagematrix.shape[2]
    new = np.zeros((height, width, channel))

    if direction == 0:
        for i in range(height):
            k = width - 1
            for j in range(width):
                new[i, j] = imagematrix[i, k]
                k = k - 1
    else:
        k = height - 1
        for i in range(height): 
            for j in range(width):
                new[i, j] = imagematrix[k, j]
            k = k - 1        

    new = new / 255
    return matrixToImages(new)


photos = loadImage('./Sierra23.jpg')

image = imageMirror(photos, mirrorlevel)
image.show()

# plt.figure('Image Mirror')
# plt.imshow(image)
# plt.show()
