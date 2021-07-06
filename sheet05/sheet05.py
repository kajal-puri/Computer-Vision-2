import os

from task1 import task_1
from task2 import task_2
from task3 import task_3

# !!!! Don't include any other package !!!!

def main():
    # =========================== Parameter =============================
    img_num = 265
    bag_num = 54

    vocabulary_path = "./offline/{:04d}".format(bag_num)
    image_path = "./images/"
    # ===================================================================

    os.makedirs(vocabulary_path, exist_ok=True)

    print("=========================== TASK 1 ================================")
    response_hists_sift = task_1(image_path, vocabulary_path, img_num, bag_num)
    print("===================================================================")

    print("=========================== TASK 2 ================================")
    task_2(response_hists_sift, img_num)
    print("===================================================================")

    print("=========================== TASK 3 ================================")
    task_3(response_hists_sift, img_num)
    print("===================================================================")


if __name__ == '__main__':

    main()
