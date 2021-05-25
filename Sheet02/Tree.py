import numpy as np
from Node import Node
from Sampler import PatchSampler

# np.random.seed(2805)
np.random.seed(2007)

class DecisionTree():
    def __init__(self, patches, labels, tree_param,mode='train'):
        if(mode=="train"):
            self.patches, self.labels = patches, labels
            self.depth = tree_param['depth']
            self.pixel_locations = tree_param['pixel_locations']
            self.random_color_values = tree_param['random_color_values']
            self.no_of_thresholds = tree_param['no_of_thresholds']
            self.minimum_patches_at_leaf = tree_param['minimum_patches_at_leaf']
            self.classes = tree_param['classes']
            self.patch_size = tree_param['patch_size']
        self.nodes = []

    # Function to train the tree
    # provide your implementation
    # should return a trained tree with provided tree param
    def train(self):
        sampler = PatchSampler(self.patches, self.labels,self.classes,self.patch_size)
        images,sample_patches,labels = sampler.extractpatches()
        self.images,self.sample_patches,self.labels= images,sample_patches,np.asarray(labels)
        # print("len(sample_patches): ",len(self.sample_patches)) # 32116
        # print("self.sample_patches[7]: ",self.sample_patches[7]) #[38, 214, 3]
        # print(sample_patches)
        patch_indexes = np.arange(0,len(self.sample_patches),dtype=np.int16)
        sample_queue = [ {'idx': patch_indexes, 'depth':0}]
        nodes = 0
        node_num =0
        previous_depth = 0
        position = 0
        while sample_queue:
            sample_data = sample_queue.pop(0)
            sample_data['idx'] = sample_data['idx'].astype(int)
            if(sample_data['depth']> previous_depth):
                node_num =1
                previous_depth = sample_data['depth']
            else:
                node_num +=1

            if( len(sample_data['idx'])<self.minimum_patches_at_leaf  or sample_data['depth'] >= self.depth \
                    or len(np.unique(self.labels[sample_data['idx']])) == 1    ):
                print("We are at leaf node")
                node = Node()
                labels = self.labels[sample_data['idx']]
                node.create_leafNode(labels,self.classes)
                self.nodes.append(node)
                nodes+=1


            else:
                print("We are at split node")
                l_node, r_node, feat = self.best_split(sample_data['idx'],labels)
                l_node['depth'] = sample_data['depth']+1
                r_node['depth'] = sample_data['depth']+1
                sample_queue.append(l_node)
                sample_queue.append(r_node)
                node = Node()
                l_child = position+1
                r_child = position+2
                position = r_child
                node.create_SplitNode(l_child,r_child,feat)
                self.nodes.append(node)
                nodes +=1




    # Function to predict probabilities for single image
    # provide your implementation
    # should return predicted class for every pixel in the test image
    def predict(self, I):
        x = np.arange(0,I.shape[1])
        y = np.arange(0,I.shape[0])
        labels = np.zeros((I.shape[0], I.shape[1],4))
        width,height = I.shape[1], I.shape[0]
        pixels = []
        for y_i in y:
            for x_i in x:
                pixels.append([y,x])
        for p in pixels:
            if p[0]> height-16 or p[1]>width-16:
                continue
            node = self.nodes[0]
            while True:
                if (node['type']=='split'):
                    feature = [node['location'],node['color_channel']]
                    response = self.test_feat_response(p,I,feature)
                    if (response <= node['th']):
                        node = self.nodes[node['left']]
                    else:
                        node = self.nodes[node['right']]
                else:
                    labels[p[0],p[1],:] = np.asarray(node['probabilities'])
                    break
        return  labels

    def test_feat_response(self, pixel,image,feature):
        dx = feature[0][1]
        dy = feature[0][0]
        channel = feature[1]
        y = pixel[0]
        x = pixel[1]
        ox = dx +x
        oy = dy + y
        response = float(image[oy,ox,channel])
        return response


    # Function to get feature response for a random color and pixel location
    # provide your implementation
    # should return feature response for all input patches
    """
    For each of the patches in the sample patch.
        ->select one patch 
        ->Extract location x,y and Class id 
        ->Use the class id to select training image
        ->Create new location by adding (x,y) with random location from feature parameter
        ->Get the "pixel value" at the new location of the selected training image as feature response 
    """
    def getFeatureResponse(self, patches, feature):
        '''

        :param patches: sample_patches
        :type patches: list of [y,x,class_id]
        :param feature: features
        :type feature: list [random_location(y,x), random_color_channel]
        :return: Feature response -> pixel value from (y,x)+ random_location
        :rtype:
        '''
        response = np.zeros(len(patches))
        dx = feature[0][1]
        dy = feature[0][0]
        channel = feature[1]
        # print("dx: ", dx)
        # print("dy: ", dy)
        # print("color: ", channel)

        for i, patch_id in enumerate(patches):
            running_sample = self.sample_patches[patch_id]
            y,x = running_sample[0],running_sample[1]
            ox = dx + x
            oy = dy + y
            class_id = running_sample[2]
            img = self.images[class_id]
            response[i] = float(img[oy,ox,channel])
            # print("running_sample",running_sample)
            # print(f"ox=(dx+running_sample[0])={dx}+ {x} = {ox}  ")
            # print(f"oy=(dy+running_sample[1])={dy}+ {y} = {oy}  ")
            # print("len(self.images): ",len(self.images))
            # print("img_id: ",class_id)
            # print("img.shape", img.shape)
        # print(response)
        # print(response.shape)
        return response





    # Function to get left/right split given feature responses and a threshold
    # provide your implementation
    # should return left/right split
    def getsplit(self, responses, threshold):
        left = np.argwhere(responses<=threshold)
        right = np.argwhere(responses>threshold)
        return left,right

    # Function to get a random pixel location
    # provide your implementation
    # should return a random location inside the patch
    def generate_random_pixel_location(self):
        location = np.random.randint(0,16,2)
        return location

    # Function to compute entropy over incoming class labels
    # provide your implementation
    def compute_entropy(self, patches):
        total_sample = len(patches)
        entropy =0
        if(total_sample==0):
            return entropy
        # print(patches)
        labels = self.labels[patches]

        for color in self.classes:
            ids = np.argwhere(labels==color)
            total_ids = len(ids)
            if(total_ids==0):
                continue
            prob = total_ids/total_sample
            entropy  += prob * np.log(prob)
        entropy *= -1
        return  entropy

    # Function to measure information gain for a given split
    # provide your implementation
    def get_information_gain(self, Entropyleft, Entropyright, EntropyAll, Nall, Nleft, Nright):
        gain = EntropyAll - ((Nleft/Nall)*Entropyleft + (Nright/Nall)*Entropyright)
        return gain


    # Function to get the best split for given patches with labels
    # provide your implementation
    # should return left,right split, color, pixel location and threshold
    def best_split(self, patches, labels):
        entropy_all = self.compute_entropy(patches)
        max_gain = 0
        feature = {'location': [], 'color_channel':-1, 'th':0}
        total_sample= len(patches)
        split = {'left':[], 'right':[]}
        left_node_entropy = 0
        right_node_entropy=0

        for color in range(self.random_color_values):
            rand_channel = np.random.randint(0,3,1)[0]
            for l in range(self.pixel_locations):
                location = self.generate_random_pixel_location()
                feat_rspns = self.getFeatureResponse(patches,[location,rand_channel])

                thresholds = np.random.randint(0,len(feat_rspns),
                                               np.minimum(len(feat_rspns)-1, self.no_of_thresholds))
                # print("thresholds: ",thresholds)
                # print("len(thresholds): ",len(thresholds))
                for th in thresholds:
                    # print("th: ", th)
                    # print("feat_rspns[{th}]: ",feat_rspns[th])
                    l_split,r_split = self.getsplit(feat_rspns,feat_rspns[th])
                    # print("l_split: ",l_split)
                    # print("l_split.length: ",len(l_split))
                    # print("r_split: ",r_split)
                    # print("r_split.length: ",len(r_split))
                    id_left = patches[l_split]
                    id_right = patches[r_split]
                    total_left = len(id_left)
                    total_right = len(id_right)
                    # print("... entropy_left  ...")
                    entropy_left = self.compute_entropy(id_left)
                    # print(".... entropy_right....")
                    entropy_right = self.compute_entropy(id_right)
                    info_gain = self.get_information_gain(entropy_left,entropy_right,entropy_all,total_sample,total_left,total_right)

                    if info_gain > max_gain:
                        feature['location']= location
                        feature['color_channel']= rand_channel
                        feature['th'] = feat_rspns[th]
                        split['left'] = id_left
                        split['right'] = id_right
                        max_gain = info_gain
                        left_node_entropy = entropy_left
                        right_node_entropy = entropy_right

        print('Info gain after split : ' + str(max_gain))
        print('Entropy in left : ' + str(left_node_entropy))
        print('Entropy in right : ' + str(right_node_entropy))
        print('OverAll Entropy before Split : ' + str(entropy_all))

        total_l = len(split['left'])
        total_r = len(split['right'])
        left_node = np.zeros(total_l)
        right_node = np.zeros(total_r)
        for elem in range(total_l):
            left_node[elem] = split['left'][elem][0]
        for elem in range(total_r):
            right_node[elem] = split['right'][elem][0]
        leftNode_id = {'idx': left_node}
        rightNode_id = {'idx': right_node}

        return leftNode_id,rightNode_id,feature




    # feel free to add any helper functions

