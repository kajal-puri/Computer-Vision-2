import numpy as np
from numpy.linalg import inv

##############################################################################################################
#Auxiliary functions for Regression
##############################################################################################################
#returns features with bias X (num_samples*(1+num_features)) and target values Y (num_samples*target_dims)
def read_data_reg(filename):
    data = np.loadtxt(filename)
    Y = data[:,:2]
    X = np.concatenate((np.ones((data.shape[0], 1)), data[:,2:]), axis=1)
    return Y, X

#takes features with bias X (num_samples*(1+num_features)) and target values Y (num_samples*target_dims)
#returns regression coefficients w ((1+num_features)*target_dims)
def lin_reg(X, Y):
    pass

#takes features with bias X (num_samples*(1+num_features)), target Y (num_samples*target_dims) and regression coefficients w ((1+num_features)*target_dims)
#returns fraction of mean square error and variance of target prediction separately for each target dimension
def test_lin_reg(X, Y, w):
    pass

#takes features with bias X (num_samples*(1+num_features)), centers of clusters C (num_clusters*(1+num_features)) and std of RBF sigma
#returns matrix with scalar product values of features and cluster centers in higher embedding space (num_samples*num_clusters)
def RBF_embed(X, C, sigma):
    pass

############################################################################################################
#Linear Regression
############################################################################################################

def run_lin_reg(X_tr, Y_tr, X_te, Y_te):

    print('MSE/Var linear regression')
    print(err)

############################################################################################################
#Dual Regression
############################################################################################################
def run_dual_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list):
    for sigma_pow in range(-5, 3):
        sigma = np.power(3.0, sigma_pow)
        print('MSE/Var dual regression for val sigma='+str(sigma))
        print(err_dual)

    print('MSE/Var dual regression for test sigma='+str(opt_sigma))
    print(err_dual)

############################################################################################################
#Non Linear Regression
############################################################################################################
def run_non_lin_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list):
    from sklearn.cluster import KMeans
    for num_clusters in [10, 30, 100]:
        for sigma_pow in range(-5, 3):
            sigma = np.power(3.0, sigma_pow)
            print('MSE/Var non linear regression for val sigma='+str(sigma)+' val num_clusters='+str(num_clusters))
            print(err_dual)

    print('MSE/Var non linear regression for test sigma='+str(opt_sigma)+' test num_clusters='+str(opt_num_clusters))
    print(err_dual)

####################################################################################################################################
#Auxiliary functions for classification
####################################################################################################################################
#returns features with bias X (num_samples*(1+num_feat)) and gt Y (num_samples)
def read_data_cls(split):
    feat = {}
    gt = {}
    for category in [('bottle', 1), ('horse', -1)]: 
        feat[category[0]] = np.loadtxt('data/'+category[0]+'_'+split+'.txt')
        feat[category[0]] = np.concatenate((np.ones((feat[category[0]].shape[0], 1)), feat[category[0]]), axis=1)
        gt[category[0]] = category[1] * np.ones(feat[category[0]].shape[0])
    X = np.concatenate((feat['bottle'], feat['horse']), axis=0)
    Y = np.concatenate((gt['bottle'], gt['horse']), axis=0)
    return Y, X

#takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# Y must be from {-1, 1}
#returns gradient with respect to w (num_features)
def sig(X, w):
    z = np.matmul(X, w)
    return 1.0 / ((1.0 + np.exp(-z)) ) #+ 1e-14

def log_llkhd_grad(X, Y, w):
    s = sig(X, w)
    logl = -np.matmul(X.T, s - (Y+1)/2)
    return logl

#takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# Y must be from {-1, 1}
#returns log likelihood loss
def get_loss(X, Y, w):
    Y_op = sig(X,w)
    loss = -(np.dot((Y.T+1)/2, np.log(Y_op)) + np.dot(1 - (Y.T+1)/2, np.log(1 - Y_op)))
    return loss

#takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# Y must be from {-1, 1}
#returns accuracy
def get_accuracy(X, Y, w):
    Y_dash = sig(X,w)
    Y_dash[Y_dash >= .5] = 1
    Y_dash[Y_dash < .5] = -1
    return np.count_nonzero(Y_dash == Y)/Y.shape[0]


####################################################################################################################################
#Classification
####################################################################################################################################
def run_classification(X_tr, Y_tr, X_te, Y_te, step_size):
    weights = np.random.randn(X_tr.shape[1])
    print('classification with step size '+str(step_size))
    max_iter = 10000
    for step in range(max_iter):
        gradient = log_llkhd_grad(X_tr, Y_tr, weights)
        weights = weights - (step_size * gradient)
        if step%1000 == 0:
            loss = get_loss(X_tr, Y_tr, weights)
            accuracy = get_accuracy(X_tr, Y_tr, weights)
            print('step='+str(step)+' loss='+str(loss)+' accuracy='+str(accuracy))
    losst = get_loss(X_te, Y_te, weights)
    accuracyt = get_accuracy(X_te, Y_te, weights)
    print('test set loss='+str(losst)+' accuracy='+str(accuracyt))


####################################################################################################################################
#Exercises
####################################################################################################################################
Y_tr, X_tr = read_data_reg('./data/regression_train.txt')
Y_te, X_te = read_data_reg('./data/regression_test.txt')

#run_lin_reg(X_tr, Y_tr, X_te, Y_te)

tr_list = list(range(0, int(X_tr.shape[0]/2)))
val_list = list(range(int(X_tr.shape[0]/2), X_tr.shape[0]))

#run_dual_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list)
#run_non_lin_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list)

step_size = 0.0001
Y_tr, X_tr = read_data_cls('test')
Y_te, X_te = read_data_cls('test')
run_classification(X_tr, Y_tr, X_te, Y_te, step_size)

#For step size = 0.0001 we get 
#test set loss=78.80002497898153 accuracy=0.5

#Answer 2(a). We couldn't get accuracy more than 50% because our model got probably stuck in local minima



