import cv2 as cv
import numpy as np
# TODO: This class should implement the following functionality:
# - A HOGDescriptor combined with a sliding window
# - Perform detection at multiple scales, i.e. you need to scale the extracted patches when performing the detection
# - Non-maximum-suppression: eliminate detections using non-maximum-suppression based on the overlap area
# Some constants that you will be using in your implementation
detection_width	= 64 # the crop width dimension
detection_height = 128 # the crop height dimension
window_stride = 32 # the stride size
scaleFactors = [1.2] # scale each patch down by this factor, feel free to try other values
# You may play with different values for these two theshold values below as well
hit_threshold = 0 # detections above this threshold are counted as positive.
overlap_threshold = 0.1 # if the overlap between two detections is above this threshold, eliminate the one with the lower confidence score.

class Custom_Hog_Detector:


    def __init__(self, trained_svm_name):
        #load the trained SVM from file trained_svm_name
        self.svm = trained_svm_name
        self.win_size =(detection_height, detection_width)
        self.scale = scaleFactors[0]
        self.window_stride = window_stride


    def detection_scalePyramid(self,img_path,hog):
        image = cv.imread(img_path)
        detections= []
        scale_val = 1
        prediction_vals = []
        for i,resized_img in enumerate(self.scale_pyramid(image,self.scale,self.win_size)):
            for(x,y,cur_window) in self.sliding_win(resized_img,self.window_stride,self.win_size):
                ## Check if current window size matches with given win_size
                if(cur_window.shape[0]!= self.win_size[0] or cur_window.shape[1]!= self.win_size[1] ):
                    continue
                hog_feats = hog.compute(cur_window).T
                prediction = self.svm.predict(hog_feats,flags=cv.ml.StatModel_RAW_OUTPUT)
                pred_score = prediction[1][0][0]
                if pred_score<hit_threshold:
                    continue
                x1 = int (x * scale_val)
                y1 = int (y * scale_val)
                x2 = int((x+detection_width) * scale_val)
                y2 = int((y+detection_height) * scale_val)

                detections.append((x1,y1,x2,y2))
                prediction_vals.append(pred_score)

            scale_val *=self.scale
        print("Number of detections: ", len(detections))
        self.draw_bounding_boxes (image,detections)
        # cv.imshow("All detections", image )
        # cv.waitKey(0)
        img_name = img_path.split('/')[-1]
        cv.imwrite('task5_res/original_%s' % img_name, image)

        # reduced_detections = self.non_max_suppression(detections,overlap_threshold,prediction_vals)
        reduced_detections = self.nms_fast(detections,overlap_threshold,prediction_vals)
        print("reduced_detections: ",len(reduced_detections))
        self.draw_bounding_boxes(image,reduced_detections)
        # cv.imshow("Reduced Detections",image)
        # cv.waitKey(0)
        cv.imwrite('task5_res/reduced_%s' % img_name, image)

    def draw_bounding_boxes(self,image, detections):
        for detect in detections:
            x1,y1,x2,y2= detect
            cv.rectangle(image,(x1,y1),(x2,y2),(255,255,0),2)

    def scale_pyramid(self,image, scale= 1.2, min_size=(128,64)):
        yield image
        while True:
            width = int(image.shape[1]/scale)
            height = int(image.shape[0]/scale)
            image = cv.resize(image,(width,height))
            if(image.shape[0]<min_size[0] or image.shape[1]<min_size[1]):
                break
            yield image

    def sliding_win(self, image, step_size=32, win_size=(128,64)):
        '''

        :param image: Image
        :type image: np.array
        :param step_size: stride in x and y directions
        :type step_size: int
        :param win_size: Size of the sliding window
        :type win_size: (height,width)
        :return: sliding window (x,y,current_window)
        :rtype:
        '''
        for y in range(0,image.shape[0], step_size):
            for x in range(0,image.shape[1],step_size):
                current_win = image[y:y+win_size[0], x:x+win_size[1]]
                yield (x,y,current_win)


    def non_max_suppression(self,boxes, threshold,pred_vals):
        boxes = np.array(boxes)  ## from tuple to list
        pred_score = pred_vals
        if len(boxes) == 0:
            return []
        # initialize the list of picked indexes
        picked_idxs = []
        suppress = []
        ## Get the coordinates of bounding boxes
        x_1,y_1,x_2,y_2 = boxes[:,0], boxes[:,1],boxes[:,2],boxes[:,3]

        # compute the area of the bounding boxes
        area = (x_2 - x_1 ) * (y_2 - y_1)

        idxs = np.argsort(y_2)  # will be used to compute the overlap ratio of other bounding boxes
        while(len(idxs)>0):
            last = len(idxs)-1
            i = idxs[last]
            picked_idxs.append(i)
            suppress=[last]
            for pos in range(0,last):
                j = idxs[pos]
                xx1 = max(x_1[i],x_1[j])
                yy1 = max(y_1[i],y_2[j])
                xx2 = min(x_2[i],x_2[j])
                yy2 = min(y_2[i],y_2[j])

                #Compute the height and width of the bounding box
                w = max(0,xx2-xx1)
                h = max(0,yy2-yy1)

                intersect_area = float(w*h)
                overlap = intersect_area / (area[i]+area[j]-intersect_area)

                if overlap.any()> threshold:
                    if(pred_score[i]>= pred_score[j]):
                        np.delete(idxs,j)
                        return boxes[picked_idxs]

    def nms_fast(self,boxes,threshold,pred_vals):
        boxes =np.array(boxes)
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []
        # initialize the list of picked indexes
        picked_idxs = []

        # get the coordinates of the bounding boxes
        x_1,y_1,x_2,y_2 = boxes[:, 0],boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # compute the area of the bounding boxes
        area = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)
        idxs = np.argsort(y_2) # sort by y2 values

        while len(idxs) > 0:
            ## Get the highest y2 which is the last index of idxs
            last = len(idxs) - 1
            i = idxs[last]
            picked_idxs.append(i)
            ## Get the coordinates of rest of the indexes till the last index
            xx1 = np.maximum(x_1[i], x_1[idxs[:last]])
            yy1 = np.maximum(y_1[i], y_1[idxs[:last]])
            xx2 = np.minimum(x_2[i], x_2[idxs[:last]])
            yy2 = np.minimum(y_2[i], y_2[idxs[:last]])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > threshold)[0] )))


        return boxes[picked_idxs]