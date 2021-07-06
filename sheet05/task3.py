import numpy as np
import itertools
from task2 import knn_classification
from typing import List
# !!!! Don't include any other package !!!!


def sim_dissim_pairs(img_num):
    """
    Generate indexes of images in pairs.

    :param img_num: number of images
    :type img_num: int
    :return: indexes of images in pairs
    :rtype: np.array
    """

    ## generate pairs of image numbers
    all_pairs = []
    for pair in itertools.combinations(range(img_num),2):
        all_pairs.append(pair)
    all_pairs = np.array(all_pairs)

    ## Divide pairs into similar pairs and dissimilar pairs (every class has 5 images)
    sim_pair = all_pairs[all_pairs[:,0]//5 == all_pairs[:,1]//5]
    dissim_pair= all_pairs[all_pairs[:,0]//5 != all_pairs[:,1]//5]

    return sim_pair, dissim_pair

def calc_covariance_mat(response_hists,pair):
    """
    Calculate covariance matrix for each pair of images.

    :param response_hists: SIFT histograms
    :type response_hists:
    :param pair: Either similar pair of images or dissimilar pair of images
    :type pair: np.array (indexes of images)
    :return: Covariance Matrix, \Sigma
    :rtype: np.array
    """
    x_i = response_hists[pair[:,0]]
    x_j = response_hists[pair[:,1]]
    cov_mat = (x_i-x_j)@ (x_i-x_j).T

    return cov_mat

def mahalanobis_metric(cov_mat_sim, cov_mat_dissim):
    """
    Learn squared Mahalanobis Distance Metric, M_hat.

    :param cov_mat_sim: Covariance Matrix of similar pair of images
    :type cov_mat_sim: np.array
    :param cov_mat_dissim: Covariance Matrix of dissimilar pair of images
    :type cov_mat_dissim: np.array
    :return: Mahalanobis Distance Metric, m_hat
    :rtype: np.array
    """
    inv_cov_sim = np.linalg.inv(cov_mat_sim)
    inv_cov_dissim = np.linalg.inv(cov_mat_dissim)
    m_hat = inv_cov_sim - inv_cov_dissim

    return m_hat

def get_psd_m(m_hat):
    """
    Verifies if the learned metric m_hat is positive semi definite. If not replace negative
    eigenvalues with 0.001 and calculate m = PDP^-1.

    :param m_hat: mahalanobis metric m_hat
    :type m_hat: np.array
    :return: psd mahalanobis metric M
    :rtype: np.array
    """
    eigen_val, eigen_vec = np.linalg.eig(m_hat)
    eigen_val[eigen_val<0] = 0.001
    eigen_val = np.expand_dims(eigen_val,axis=1)
    p_mat = eigen_vec * eigen_val
    p_inv = np.linalg.inv(p_mat)
    m = p_mat.dot(eigen_val*p_inv) ## wrong: 51
    return m

def calc_mahalanobis_dist(response_hists, m):
    """
    Calculate Mahalanobis Distance for each image's SIFT features

    :param response_hists: SIFT Features
    :type response_hists: np.array
    :param m: Matrix M
    :type m: np.array
    :return: Distance Matrix
    :rtype: np.array
    """
    distance_mat = np.zeros((response_hists.shape[0],response_hists.shape[0])) ## (265, 265)
    ## For each image SIFT, calculate the distance between other images' SIFT
    for i in range(distance_mat.shape[0]):
        x_i = response_hists[i]
        for j in range(distance_mat.shape[1]):
            x_j = response_hists[j]
            difference = x_i - x_j
            difference = np.expand_dims(difference,axis=0)
            single_dist = difference.dot(m.dot(difference.T))
            distance_mat[i,j]= single_dist

    return distance_mat



def task_3(response_hists:List, img_num:int):
    # ToDO:
    sim_pair, dissim_pair = sim_dissim_pairs(img_num)
    sim_cov_mat = calc_covariance_mat(response_hists,sim_pair)
    dissim_cov_mat = calc_covariance_mat(response_hists,dissim_pair)
    m_hat = mahalanobis_metric(sim_cov_mat,dissim_cov_mat)
    m = get_psd_m(m_hat)
    distances = calc_mahalanobis_dist(response_hists,m)

    knn_classification(1, img_num, distances)
    knn_classification(3, img_num, distances)
