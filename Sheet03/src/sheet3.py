import cv2 as cv
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import os
from custom_hog_detector import Custom_Hog_Detector
# Global constants

# crop/patch dimensions for the training samples
width = 64
height = 128

num_negative_samples = 10 # number of negative samples per image
train_hog_path = '/home/kajal/Desktop/Bonn_CSE/Sem 4/CV-2/Sheet03/train_hog_descs.dat.npy' # the file to which you save the HOG descriptors of every patch
train_labels = '/home/kajal/Desktop/Bonn_CSE/Sem 4/CV-2/Sheet03/labels_train.dat.npy' # the file to which you save the labels of the training data
my_svm_filename = '/home/kajal/Desktop/Bonn_CSE/Sem 4/CV-2/Sheet03/my_pretrained_svm.dat' # the file to which you save the trained svm 

#data paths
test_images_1 = '/home/kajal/Desktop/Bonn_CSE/Sem 4/CV-2/Sheet03/data/task_1_testImages/'
path_train_2 = '/home/kajal/Desktop/Bonn_CSE/Sem 4/CV-2/Sheet03/data/task_2_3_data/train/'
path_test_2 = '/home/kajal/Desktop/Bonn_CSE/Sem 4/CV-2/Sheet03/data/task_2_3_data/test/'

#***********************************************************************************
# draw a bounding box in a given image
# Parameters:
# im: The image on which you want to draw the bounding boxes
# detections: the bounding box of the detections (people)
# returns None

def drawBoundingBox(im, detections):

	# the HOG detector returns slightly larger rectangles than the real objects.
    # so we slightly shrink the rectangles to get a nicer output.
        

	for x,y,w,h in detections:
		pad_w, pad_h = int(0.15*w), int(0.05*h)
		cv.rectangle(im, (x+pad_w,y+pad_h), (x+w-pad_w, y+h-pad_h), (0,255,0), 2)
	cv.imshow('BoundingBox',im)
	cv.waitKey(0)
	cv.destroyAllWindows()

def task1():

    print('Task 1 - OpenCV HOG')
    
    # Load images

    filelist = test_images_1 + 'filenames.txt'

    testimgs = []

    with open(filelist) as f:
    	for file in f.readlines():
    		testimgs.append(cv.imread(test_images_1 + file.strip()))

    print(len(testimgs))

    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    print(hog)


    for img in testimgs:
    	#print(img.shape)
    	#print(hog)
    	h, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32, 32), scale=1.05)
    	drawBoundingBox(img,h)

    cv.destroyAllWindows()
    cv.waitKey(0)


def task2():

    print('Task 2 - Extract HOG features - is starting')

    random.seed()
    np.random.seed()

    # Load image names
  
    filelist_train_pos = path_train_2+'filenamesTrainPos.txt'
    filelist_train_neg = path_train_2+'filenamesTrainNeg.txt'
    # TODO: Create a HOG descriptor object to extract the features from the set of positive and negative samples 

    # positive samples: Get a crop of size 64*128 at the center of the image then extract its HOG features
    # negative samples: Sample 10 crops from each negative sample at random and then extract their HOG features
    # In total you should have  (x+10*y) training samples represented as HOG features(x=number of positive images, y=number of negative images),
    # save them and their labels in the path train_hog_path and train_labels in order to load them in section 3 
    train_files = []
    labels = []
    hog = cv.HOGDescriptor()

    with open(filelist_train_pos) as tf:
    	for file in tf.readlines():
    		img = cv.imread(path_train_2+'pos/'+file.strip())
    		#print("img shape",img.shape[0], img.shape[1])
    		center = img.shape[0]//2,img.shape[1]//2
    		img = img[center[0]-height//2:center[0]+height//2,center[1]-width//2:center[1]+width//2]
    		train_files.append(hog.compute(img).squeeze())
    		labels.append(0)

    with open(filelist_train_neg) as tef:
    	for fi in tef.readlines():
    		img=cv.imread(path_train_2+'neg/'+fi.strip())
    		for i in range(10):
    			x,y = np.random.randint(0, img.shape[1]-width), np.random.randint(0, img.shape[0]-height)
    			crop = img[y:y+height, x:x+width]
    			train_files.append(hog.compute(crop).squeeze())
    			labels.append(1)

    np.save(train_labels,labels)
    np.save(train_hog_path,train_files)
    print("Task 2 is Done!!")


def task3(): 
    print('Task 3 - Train SVM and predict confidence values')
      #TODO Create 3 SVMs with different C values, train them with the training data and save them
      # then use them to classify the test images and save the results
    

    filelist_testPos = path_test_2 + 'filenamesTestPos.txt'
    filelist_testNeg = path_test_2 + 'filenamesTestNeg.txt'

    print('Loading of Training Data Starts')
    trained_hog = np.load(train_hog_path)
    trained_labels = np.load(train_labels)

    hog = cv.HOGDescriptor()
    test_files = []
    labels = []

    print("Testing Data")

    with open(filelist_testPos) as tf:
    	for file in tf.readlines():
    		img = cv.imread(path_test_2+'pos/'+file.strip())
    		#print("img shape",img.shape[0], img.shape[1])
    		center = img.shape[0]//2,img.shape[1]//2
    		img = img[center[0]-height//2:center[0]+height//2,center[1]-width//2:center[1]+width//2]
    		test_files.append(hog.compute(img).squeeze())
    		labels.append(0)

    with open(filelist_testNeg) as tef:
    	for fi in tef.readlines():
    		img=cv.imread(path_test_2+'neg/'+fi.strip())
    		for y in range(0, img.shape[0]-height,height):
    			for x in range(0, img.shape[1]-width, width):
    				crop = img[y:y+height, x:x+width]
    				test_files.append(hog.compute(crop).squeeze())
    				labels.append(1)

    test_files = np.stack(test_files)
    labels = np.stack(labels)

    range_c = [0.01, 1, 10]
    for c in range_c:
    	print(f"training with C={c}")
    	svm = cv.ml.SVM_create()
    	svm.setType(cv.ml.SVM_C_SVC)
    	svm.setKernel(cv.ml.SVM_LINEAR)
    	svm.setC(c)
    	svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    	svm.train(trained_hog, cv.ml.ROW_SAMPLE, trained_labels)
    	confidence_score = svm.predict(test_files,svm.predict(test_files)[1],cv.ml.StatModel_RAW_OUTPUT)
    	print(f'Saving Model and confidence scores and test labels for C = {c}')
    	np.save(f'confidence_{c}.npy',confidence_score[1].squeeze())
    	svm.save(f'SVM_{c}_.dat')
    print("Saving Test Labels")
    np.save('test_labels.npy',labels)


def task4():

	print("Task 4 ----- Calculating Precision and Recall for the trained SVM model")
	predicted_labels = np.load('test_labels.npy')

	precision = []
	recall = []
	hue = []
	range_c = [0.01,1,10]
	for c in range_c:
		confidence_score = np.load(f'confidence_{c}.npy')
		for i in np.random.rand(15)-0.5:
			labels = (confidence_score < i)
			tp = ((labels == predicted_labels) & (labels==1)).sum()
			fp = ((labels != predicted_labels) & (labels==1)).sum()
			fn = ((labels != predicted_labels) &(labels==0)).sum()

			recall.append(tp/tp+fp)
			precision.append(tp/tp+fn)
			hue.append(f'C={c}')

	sns.scatterplot(x=recall,y=precision,hue=hue)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title(f'Recall-Accuracy graph')
	plt.show()



def task5():

	print ('Task 5 - Eliminating redundant Detections')
	svm_fPath = "SVM_1_.dat"
	my_svm_filename = cv.ml.SVM_load(svm_fPath)

	# TODO: Write your own custom class myHogDetector

	my_detector = Custom_Hog_Detector(my_svm_filename)
	hog = cv.HOGDescriptor()
	# TODO Apply your HOG detector on the same test images as used in task 1 and display the results
	file_path = test_images_1+ "filenames.txt"
	imgFile_paths =[]
	with open(file_path,"r") as myFile:
		for line in myFile:
			img_name = line.rstrip(".")[2:]
			img_path = os.path.join(test_images_1,img_name).strip()
			imgFile_paths.append(img_path)



	for img_file_path in imgFile_paths:
		my_detector.detection_scalePyramid(img_file_path,hog)
		
	print('Done!')
	cv.waitKey()
	cv.destroyAllWindows()






if __name__ == "__main__":

    # Task 1 - OpenCV HOG
    #task1()

    # Task 2 - Extract HOG Features
    #task2()

    # Task 3 - Train SVM
    #task3()

    #Task 4 - Calculate Precision and Recall 
    # task4()

    # Task 5 - Multiple Detections
    task5()
	"""
	0.1 is the good value for the elimintation threshold when applying NMS 
	"""
