import numpy as np


class Node():
    def __init__(self):

        self.type = 'None'
        self.leftChild = -1
        self.rightChild = -1
        self.feature = {'color': -1, 'pixel_location': [-1, -1], 'th': -1}
        self.probabilities = []

    # Function to create a new split node
    # provide your implementation
    def create_SplitNode(self, leftchild, rightchild, feature):
        self.type = 'split'
        self.leftChild = leftchild
        self.rightChild = rightchild
        self.feature['color'] = float(feature['color_channel'])
        self.feature['pixel_location']= feature['location'].tolist()
        self.feature['th'] = float(feature['th'])

    # Function to create a new leaf node
    # provide your implementation
    def create_leafNode(self, labels, classes):
        self.type= 'leaf'
        N = len(labels)
        for c in classes:
            prob_c = len(np.argwhere(labels==c))
            self.probabilities.append(float(prob_c)/N)


    # feel free to add any helper functions