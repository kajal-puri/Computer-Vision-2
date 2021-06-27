import numpy as np
import os
import cv2
import cv2.xfeatures2d as features
import yaml

from typing import List

# !!!! Don't include any other package !!!!

def task_1(image_path: str, vocabulary_path: str, img_num: int, bag_num: int) -> List:
    voc_file_path = os.path.join(vocabulary_path, 'vocabulary.yaml')

    if os.path.isfile(voc_file_path):
        print("Load vocabulary from %s" % voc_file_path)
        # ToDo:
    else:
        print("Generating vocabulary")
        # ToDo:
        print("DONE")

        print("Writing vocabulary to %s" % voc_file_path )
        # ToDo

    # 1c)
    response_hists = []

    # ToDo:

    return response_hists
