import cv2
import matplotlib.pyplot as plt
import  numpy as np

def Unity_image_show(name,img1,img2,img3):
    plt.title(name)
    plt.margins(0, 0)
    plt.subplot(1,3,1)
    plt.imshow(img1)
    plt.subplot(1, 3, 2)
    plt.imshow(img2)
    plt.subplot(1, 3, 3)
    plt.imshow(img3)
    plt.show()


