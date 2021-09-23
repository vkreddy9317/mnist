import sklearn
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
from sklearn.model_selection import train_test_split
def preprocessing(image,rescale_factor):
    resized_images = []
    for img in digits.images:
        resized_images.append(transform.rescale(img,resize_images_size[i],anti_aliasing=False))
    return resized_images

def create_split(data,target,train_size,valid_size,test_size):
    train_X,test_X,train_Y,test_Y = train_test_split(data,digits.target,test_size=test_size + valid_size,shuffle=False)
    val_X,test_X,val_Y,test_Y = train_test_split(test_X,test_Y,test_size =test_size,shuffle = False)
