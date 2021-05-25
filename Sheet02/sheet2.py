from RandomForest import Forest
from Sampler import PatchSampler
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

def main():

    # provide your implementation for the sheet 2 here
    #### Get training data
    f= open("images/train_images.txt","r")
    training_imgs =[]
    training_lbls = []
    for i,f_path in enumerate(f):
        if i>0: ## To skip the first line containing numOf images and labels
            im_fname,lbl_fname = f_path.split(' ') ## Image:img_3.bmp  Label:img_3_segMap.bmp
            img = cv2.imread('images/'+ im_fname) # (213, 320, 3)
            lbl = cv2.imread('images/'+lbl_fname.strip()) # (213, 320, 3) Use [:-1] to remove whitespace
            label = lbl[:,:,0]

            training_imgs.append(img)
            training_lbls.append(label)

    n_trees = 5
    tree_params = { "patch_size":[16,16],"depth": 15,
                    "minimum_patches_at_leaf":20, "random_color_values":10,
                    "pixel_locations": 100, "no_of_thresholds":50,
                    "classes":[0,1,2,3]}

    rand_forest = Forest(training_imgs,training_lbls,tree_params,n_trees)
    rand_forest.create_forest()
    test_images_files = open("images/test_images.txt", "r")
    for j, f_name in enumerate(test_images_files):
        if j > 0:
            print('Applying Forest to test image : ' + str(j))
            img, label = f_name.split(' ')
            im = cv2.imread('images/' + img)
            lp = f.test(im)
            plt.imshow(lp)
            plt.show()

if __name__ =='__main__':
    main()
