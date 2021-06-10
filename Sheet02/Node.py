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
      self.leftchild = leftchild
      self.rightchild = rightchild
      self.feature['color'] = float(feature['color'])
      self.feature['th'] = float(feature['th'])
      self.feature['pixel_location'] = feature['location'].tolist()

    # Function to create a new leaf node
    # provide your implementation
    def create_leafNode(self, labels, classes):

      self.type = 'leaf'
      total_labels = len(labels)

      for clas in classes:
        count = 0
        if(np.nonzero(labels==clas)):
          count = count + 1
          probabilites.append(count/total_labels)
