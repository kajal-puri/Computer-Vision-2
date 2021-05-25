import numpy as np
import cv2


class PatchSampler():

    def __init__(self, train_images_list, gt_segmentation_maps_list, classes_colors, patch_size):

        self.train_images_list = train_images_list
        self.gt_segmentation_maps_list = gt_segmentation_maps_list
        self.class_colors = classes_colors
        self.patch_size = patch_size

    # Function for sampling patches for each class
    # provide your implementation
    # should return extracted patches with labels

    def extractpatches(self):

      patches = []
      for image, img_seg in zip(self.train_images_list,self.gt_segmentation_maps_list):
        image = cv2.imread("images/"+image)
        img_seg = cv2.imread("images/"+img_seg)

        h = np.random.randint(0,img_seg[0]-self.patch_size)
        w = np.random.randint(0,img_seg[1]-self.patch_size)
        patch = img_seg[h:h+self.patch_size, w:w+self.patch_size]

        for color in self.class_colors:
          patches[color].append(image[i:i+self.patch_size, j:j+self.patch_size])

      least = min(len(patches[0]), len(patches[1]), len(patches[2]), len(patches[3]))
      patches = patches[0][:least] + patches[1][:least] + patches[2][:least] + patches[3][:least]
      labels = []
      for i in range(least):
        labels.append(least)
      return np.concatenate(patches), labels


