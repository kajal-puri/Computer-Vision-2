from ch_Tree import DecisionTree
import numpy as np
import json


class Forest():
    def __init__(self, patches=[], labels=[], tree_param=[], n_trees=1):

        self.patches, self.labels = patches, labels
        self.tree_param = tree_param
        self.ntrees = n_trees
        self.trees = []
        for i in range(n_trees):
            self.trees.append(DecisionTree(self.patches, self.labels, self.tree_param))

    # Function to create ensemble of trees
    # provide your implementation
    # Should return a trained forest with n_trees
    def create_forest(self):
        print("Creating Forest")
        for i, tree in enumerate(self.trees):
            tree.train()

    # Function to apply the trained Random Forest on a test image
    # provide your implementation
    # should return class for every pixel in the test image
    def test(self, I):
        width = I.shape[1]
        height = I.shape[0]
        labels = np.zeros((height,width,4))
        predictions = np.zeros((height,width),dtype= np.uint8)
        for i in self.trees:
            labels += i.predict(I)
        for h in range(height-16):
            for w in range(width-16):
                predictions[h,w]= np.argmax(labels[h,w,:] / self.trees )
        return predictions

    # feel free to add any helper functions
