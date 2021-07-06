
import numpy as np
import cv2 as cv
import cv2.xfeatures2d as features
import os
import yaml
#from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier

def generate_bow_response_hist(vocabulary_path, image_path, img_num, bag_num, feat_type='SIFT'):
    voc_file_path = os.path.join(vocabulary_path, 'vocabulary.yaml')

    if os.path.isfile(voc_file_path):
        print("Load vocabulary from %s" % voc_file_path)        
    else:
        if (feat_type == 'SURF'):
            extractor = features.SURF_create(hessianThreshold=450, extended=True, upright=True)
        else:
            extractor = features.SIFT_create()
        print("Generating vocabulary")
        bow = cv.BOWKMeansTrainer(bag_num)

        descriptors = []
        
        for i_idx in range(img_num):
            img_path = os.path.join(image_path, "{:03d}.jpg".format(i_idx))
            print(img_path)
            image = cv.imread(img_path)
            img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            keypoints, descriptors = extractor.detectAndCompute(img, None)
            bow.add(descriptors)
            
        bow_dict = bow.cluster()
        print("DONE")

        print("Writing vocabulary to %s" % voc_file_path )
        
        if not os.path.exists(os.path.dirname(voc_file_path)):
            try:
                os.makedirs(os.path.dirname(voc_file_path))
            except OSError as exc:
                raise
                
        with open(voc_file_path,'w+') as file:
            yaml.dump(bow_dict.tolist(),file)
  
    with open(voc_file_path) as file:
        bow_dict = yaml.safe_load(file)
        
    bow_dict = np.array(bow_dict)    
    bow_dict = np.float32(bow_dict)


    if (feat_type == 'SURF'):
        extractor = features.SURF_create(hessianThreshold=450, extended=True, upright=True)
    else:
        extractor = features.SIFT_create()
        
    flann_params = dict(algorithm=1, trees=5)  # Fast Library for Approximate Nearest Neighbors
    matcher = cv.FlannBasedMatcher(flann_params, {})
    bow_extract = cv.BOWImgDescriptorExtractor(extractor, matcher)
    bow_extract.setVocabulary(bow_dict)
    
    response_hists = []
    labels = [x for x in range(int(img_num / 5))]
    cls_labels = []
    lbl_idx = -1
    print("Acquiring Response Histogram")
    for i_idx in range(img_num):
        bow_responce_hist_path = os.path.join(vocabulary_path, str(feat_type)+"BOW_response_hist_%d.yaml" % i_idx)
        file_path = os.path.join(image_path, "{:03d}.jpg".format(i_idx))
        image = cv.imread(file_path)
        img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        keypoints = extractor.detect(img)
        bowsig = bow_extract.compute(img, keypoints)
        response_hists.extend(bowsig)
        if (i_idx % 5 == 0):
            lbl_idx += 1
        cls_labels.append(labels[lbl_idx])
        with open(bow_responce_hist_path, 'w') as file:
            yaml.dump(bowsig.tolist(), file)

    cls_label_path = os.path.join(vocabulary_path,"class_labels.yaml")
    with open(cls_label_path,'w') as file:
        yaml.dump(cls_labels,file)   

    return response_hists


def l2_mat_distance(X):

    

     return 0#l2


def mahalanobis_disstance(X, M):

    # _________________ ToDO _________________

    return 0#distances


def knn_classification(knn, img_num, distances, response_hists):    
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


def task_1(image_path, vocabulary_path, img_num, bag_num):
    response_hists_surf = generate_bow_response_hist(vocabulary_path,
                                                image_path,
                                                img_num,
                                                bag_num, 'SURF')
    
    print(len(response_hists_surf))

    response_hists_sift = generate_bow_response_hist(vocabulary_path,
                                                image_path,
                                                img_num,
                                                bag_num)
    
    print(len(response_hists_sift))
    
    
    return response_hists_surf, response_hists_sift

def task_2(response_hists, img_num):

    # __________ Your Part Here __________
    distances = 0

    knn_classification(1, img_num, distances, response_hists)
    print("----------------")
    knn_classification(3, img_num, distances, response_hists)


def task_3(response_hists, img_num):

    
   
    # _______ Your Part Here _________

    knn_classification(1, img_num, distances)
    print("----------------")
    knn_classification(3, img_num, distances)


def main():

    img_num = 265
    bag_num = 54

    vocabulary_path = "./offline/{:04d}".format(bag_num)
    image_path = "./images/"

    print("TASK 1")
    response_hists_surf, response_hists_sift = task_1(image_path, vocabulary_path, img_num, bag_num)
    print("TASK 2")
    task_2(response_hists_surf, img_num)
    #print("TASK 3")
    #task_3(response_hists_sift, img_num)


if __name__ == '__main__':

    main()
