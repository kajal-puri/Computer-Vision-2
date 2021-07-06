import numpy as np
# !!!! Don't include any other package !!!!

import numpy as np
import cv2 as cv
import cv2.xfeatures2d as features
import os
import yaml
from sklearn.neighbors import KNeighborsClassifier

#!!! This function will be used in task3! 

def knn_classification(knn:int, img_num:int, distances: np.ndarray, response_hists=None):    
    print("Classify with knn = %d" % knn)
    #final_distances = []
    final_labels = []
    wrong = 0
    train_data = []
    test_data = []
    train_labels = []
    train_imgs = []
    lbl_idx = -1
    
    vocabulary_path = "offline/{:04d}".format(55)
    
    response_hists = np.array(response_hists)
    
    # iterate over all testing images
    for i_idx in range(img_num):
        if (i_idx%5==0):
            lbl_idx += 1
            test_data.append(response_hists[i_idx])
        else:
            train_data.append(response_hists[i_idx])
            train_labels.append(lbl_idx)
            train_data_path = os.path.join(vocabulary_path, "{:04d}.jpg".format(i_idx))
            train_imgs.append(train_data_path)
        
    test_data = np.array(test_data)
    train_data = np.array(train_data)
    
    classifier = KNeighborsClassifier(n_neighbors=knn, algorithm='ball_tree', metric='euclidean')
    classifier.fit(train_data,train_labels)
    
    final_labels = classifier.predict(test_data)
    dist, neighbor = classifier.kneighbors(test_data, knn)
    
    final_labels = np.array(final_labels).reshape((len(final_labels),1))
    correct = np.count_nonzero(final_labels == test_data)
                
    wrong = final_labels.shape[0]-correct
    all = final_labels.shape[0]         

        
    print("FINAL ALL", all)
    print("FINAL WRONG", wrong)
    print("WRONG/ALL", wrong / all)
    
    return 0

def task_2(response_hists:str, img_num:int):

    # ToDO:
    distances = 0

    knn_classification(1, img_num, distances, response_hists)
    print("----------------")
    knn_classification(3, img_num, distances, response_hists)
