from math import gamma
import sklearn
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
from sklearn.model_selection import train_test_split
from joblib import dump,load
import os
from utils import preprocessing,create_split

digits = datasets.load_digits()

n_sample = len(digits.images)
data = digits.images.reshape((n_sample,-1))
model_candidate = []
resize_images_size = [0.25,0.5,1,2,4,8]
hyperparameter_values =[0.00001,0.0001,0.001,0.05,1,2,5,10]
print("\t\tResize \t\tDataset \t\tgamma_value \t\tAccuracy")
os.mkdir('models')
for i in range(len(resize_images_size)):
    resized_images = []
    resized_images = np.array(resized_images)
    test_size = 0.15
    val_size = 0.15
    data = resized_images.reshape((n_sample,-1))
    resized_images = preprocessing(digits.images,resize_images_size[i])
    train_X,train_Y,test_X,test_Y,val_X,val_Y = create_split(data = data ,target = digits.targets,train_size=0.7,valid_size=0.15,test_size=0.15)
    for j in range(len(hyperparameter_values)):
        
        model = svm.SVC(gamma = hyperparameter_values[j])
        
        
        model.fit(train_X,train_Y)
        predict_val = model.predict(val_X)
        acc_val = metrics.accuracy_score(y_pred= predict_val,y_true = val_Y)
        if acc_val<0.20:
            print("Skipping for {}".format(hyperparameter_values[j]))
            continue
        candidate = {
            "acc_valid" : acc_val,
            "gamma" : hyperparameter_values[j],
        }
        model_candidate.append(candidate)
        output_folder = "./models/tt_{}_val_{}_rescale_{}_gamma_{}".format(test_size,val_size,resize_images_size[i],hyperparameter_values[j])
        os.mkdir(output_folder)
        dump(model,os.path.join(output_folder,'model.joblib'))
        print("Saving Model for {}".format(hyperparameter_values[j]))
    max_valid_acc_model_candidate = max(model_candidate,key=lambda x:x["acc_valid"])
    best_model_folder ="./models/tt_{}_val_{}_rescale_{}_gamma_{}".format(test_size,val_size,resize_images_size[i],max_valid_acc_model_candidate["gamma"])
    path = os.path.join(best_model_folder,'model.joblib')
    model = load(path)
    predict_test = model.predict(test_X)
    acc_test = metrics.accuracy_score(y_pred= predict_test,y_true = test_Y)
    print("\t\t ",resize_images_size[i] ,"\t\tValidation Set","\t\t",hyperparameter_values[j],"\t\t",acc_val*100)
    print("\t\t ",resize_images_size[i] ,"\t\tTest Set","\t\t",hyperparameter_values[j],"\t\t",acc_test*100)
