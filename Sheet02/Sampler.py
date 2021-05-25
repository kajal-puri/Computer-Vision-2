import numpy as np


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
        pixels= []
        lbls =[]
        for i,l in enumerate(self.gt_segmentation_maps_list):
            h,w = l.shape[0], l.shape[1]
            xs = np.arange(0,w-16,2,dtype=np.int16) # [0,2,4,...,302] ## 152
            ys = np.arange(0,h-16,2,dtype=np.int16) # [0,2,4,...,196] ## 99

            for y in ys:
                for x in xs:
                    ## create pixel location and extract corresponding labels
                    pix = [y,x,i]
                    lbl = l[y,x]
                    pixels.append(pix)
                    lbls.append(lbl)

            # print("Pixels length: ", len(pixels)/ (i+1)) #(152*99) 15048.0
            # print("labels length: ", len(lbls)/ (i+1)) # 15048.0
        # print("len(pixels): ",len(pixels)) # 60192
        # print("len(lbls): ",len(lbls))   # 60192

        ## Generate 10,000 random pixels with corresponding classes for each class
        shuffling = np.random.permutation(len(pixels))
        # print("len(shuffling): ",len(shuffling)) # 60192
        pixel_shuffled = []
        label_shuffled = []
        max_samples_per_class = 10000
        counts = [0,0,0,0] ## sample count for each class
        for s in shuffling:
            if(counts[lbls[s]]<max_samples_per_class):
                # print(s)
                # print(f"lbls[s]=lbls[{s}] : {lbls[s]}")
                # print("pixels[s]: ", pixels[s])
                pixel_shuffled.append(pixels[s])
                label_shuffled.append(lbls[s])
                counts[lbls[s]] += 1
                # print(f"counts[lbls[s]]= counts[{lbls[s]}] :  {counts[lbls[s]]}")
        # print("len(pixel_shuffled): ",len(pixel_shuffled))
        # print("len(label_shuffled)",len(label_shuffled))
        # print("Counts of classes: ",counts)

        return self.train_images_list, pixel_shuffled,label_shuffled


# feel free to add any helper functions


