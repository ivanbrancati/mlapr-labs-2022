########################
#####-----LAB8-----#####
########################

####LIBRARIES#####
import scipy.optimize
import numpy
import sklearn.datasets

#####USEFUL FUNCTIONS#####

#Function to obtain column vector
def vcol(D):
    return D.reshape(D.size, 1)

#Function to obtain row vector
def vrow(D):
    return D.reshape(1, D.size)

# Function to load iris dataset 
# (only virginica and versicolor)
def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

# Function to split dataset in training and validation
# train_size is the size of the training dataset
def split_db_2to1(dataset, labels, seed=0, train_size = 2.0 / 3.0):
    nTrain = int(dataset.shape[1] * train_size)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(dataset.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = dataset[:, idxTrain]
    DTE = dataset[:, idxTest]
    LTR = labels[idxTrain]
    LTE = labels[idxTest]
    return (DTR, LTR), (DTE, LTE)

# Function to evaluate a model
# predictions -> Array of predicted values
# LTE -> Testing Dataset correct labels
# print -> true to print accuracy and error rate
def evaluate_model2(predictions, LTE, printResults=False):
    
    correct_predictions = (predictions == LTE).sum()
    number_of_samples = LTE.shape[0]

    accuracy = correct_predictions / number_of_samples
    error_rate = (number_of_samples - correct_predictions) / number_of_samples

    if(printResults):
        print("Accuracy: {}%".format(accuracy * 100))
        print("Error Rate: {}%".format(error_rate * 100))

    return accuracy, error_rate

#####SVM FUNCTIONS#####

# Function that applies linear svm and prints primal loss, dual loss and duality gap
# and W star (to eventually compute scores)
# DTR -> Training dataset
# LTR -> Training labels
# C -> C parameter
# k -> k parameter

def linear_svm(DTR, LTR, C, k):
    D_cap = numpy.vstack([DTR, numpy.ones((1, DTR.shape[1])) * k])
    G_cap = numpy.dot(D_cap.T, D_cap)
    Z = numpy.zeros(LTR.shape)
    Z [LTR == 1] = 1
    Z [LTR == 0] = -1
    H_cap = vcol(Z) * vrow(Z) * G_cap
    
    def JDual(alpha):
        prod_one = numpy.dot(vrow(alpha), numpy.dot(H_cap, vcol(alpha)))
        grad = - numpy.dot(H_cap, vcol(alpha)).ravel() + numpy.ones(alpha.size)
        return -0.5 * prod_one.ravel() + alpha.sum(), grad
    
    def LDual(alpha):
        a, b = JDual(alpha)
        return -a, -b
    
    def JPrimal(w):
        S = numpy.dot(vrow(w), D_cap)
        prod_two = C * numpy.maximum(numpy.zeros(S.shape), 1- Z * S).sum()
        return 0.5 * numpy.linalg.norm(w)**2 + prod_two
    
    alpha_star, x, y = scipy.optimize.fmin_l_bfgs_b(
    LDual,
    numpy.zeros(DTR.shape[1]),
    bounds = [(0, C)] * DTR.shape[1],
    factr = 1.0,
    )
    
    w_star = numpy.dot(D_cap, vcol(alpha_star) * vcol(Z))
    
    print(f"Primal Loss: {JPrimal(w_star)}")
    print(f"Dual Loss: {JDual(alpha_star)[0][0]}")
    print(f"Duality Gap: {JPrimal(w_star)-JDual(alpha_star)[0][0]}")
    
    return w_star

# Function that applies polynomial non-linear svm and prints dual loss 
# and alpha star (to eventually compute scores)
# DTR -> Training dataset
# LTR -> Training labels
# C -> C parameter
# k -> k parameter
# d -> d parameter
# c -> c parameter
def polynomial_svm(DTR, LTR, C, k, d, c):
    Z = numpy.zeros(LTR.shape)
    Z [LTR == 1] = 1
    Z [LTR == 0] = -1
    polynomial_kernel = (numpy.dot(DTR.T, DTR) + c) ** d + k**2
    H_cap = vcol(Z) * vrow(Z) * polynomial_kernel
    
    def JDual(alpha):
        prod_one = numpy.dot(vrow(alpha), numpy.dot(H_cap, vcol(alpha)))
        grad = - numpy.dot(H_cap, vcol(alpha)).ravel() + numpy.ones(alpha.size)
        return -0.5 * prod_one.ravel() + alpha.sum(), grad
    
    def LDual(alpha):
        a, b = JDual(alpha)
        return -a, -b
    
    
    alpha_star, x, y = scipy.optimize.fmin_l_bfgs_b(
    LDual,
    numpy.zeros(DTR.shape[1]),
    bounds = [(0, C)] * DTR.shape[1],
    factr = 1.0,
    )
    
    
    print(f"Dual Loss: {JDual(alpha_star)[0][0]}")
    return alpha_star

# Function that applies radial basis function non-linear svm and prints dual loss 
# and alpha star (to eventually compute scores)
# DTR -> Training dataset
# LTR -> Training labels
# C -> C parameter
# k -> k parameter
# gamma -> gamma parameter
def rbf_svm(DTR, LTR, C, k, gamma):
    Z = numpy.zeros(LTR.shape)
    Z [LTR == 1] = 1
    Z [LTR == 0] = -1
    distance = vcol((DTR**2).sum(0)) + vrow((DTR**2).sum(0)) - 2 * numpy.dot(DTR.T, DTR)
    rbf_kernel = numpy.exp(-gamma*distance) + k**2
    H_cap = vcol(Z) * vrow(Z) * rbf_kernel
    
    def JDual(alpha):
        prod_one = numpy.dot(vrow(alpha), numpy.dot(H_cap, vcol(alpha)))
        grad = - numpy.dot(H_cap, vcol(alpha)).ravel() + numpy.ones(alpha.size)
        return -0.5 * prod_one.ravel() + alpha.sum(), grad
    
    def LDual(alpha):
        a, b = JDual(alpha)
        return -a, -b
    
    
    alpha_star, x, y = scipy.optimize.fmin_l_bfgs_b(
    LDual,
    numpy.zeros(DTR.shape[1]),
    bounds = [(0, C)] * DTR.shape[1],
    factr = 1.0,
    )
    
    
    print(f"Dual Loss: {JDual(alpha_star)[0][0]}")
    return alpha_star

#####EXECUTION#####


## Linear SVM

D, L = load_iris_binary()
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

values = [[1, 0.1], [1, 1.0], [1, 10.0], [10, 0.1], [10, 1.0], [10, 10.0]]
print("Linear SVM:\n")
for k, C in values:
    print(f"K: {k} C: {C} :")
    w_star = linear_svm(DTR, LTR, C, k)
    DTE_EXT = numpy.vstack([DTE, numpy.ones((1, DTE.shape[1])) * k])
    S = numpy.dot(vrow(w_star), DTE_EXT)
    predictions = numpy.array([1 if score > 0 else 0 for score in S.ravel()])
    error_rate = evaluate_model2(predictions, LTE)[1]
    print("Error Rate: {:.1f}%".format(error_rate * 100))
    print("\n")

## Kernel SVM

values = [[0.0, 1.0, 2, 0], [1.0, 1.0, 2, 0], [0.0, 1.0, 2, 1], [1.0, 1.0, 2, 1]]
Z = numpy.zeros(LTR.shape)
Z [LTR == 1] = 1
Z [LTR == 0] = -1
print("Polinomial Kernel: \n")
for k, C, d, c in values:
    print(f"K: {k} C: {C} : d: {d} c : {c}")
    alpha_star = polynomial_svm(DTR, LTR, C, k, d, c)
    S = numpy.dot(alpha_star * Z, (numpy.dot(DTR.T, DTE) + c) ** d + k**2)
    predictions = numpy.array([1 if score > 0 else 0 for score in S.ravel()])
    error_rate = evaluate_model2(predictions, LTE)[1]
    print("Error Rate: {:.1f}%".format(error_rate * 100))
    print("\n")

values = [[0.0, 1.0, 1.0], [0.0, 1.0, 10.0], [1.0, 1.0, 1.0], [1.0, 1.0, 10.0]]
Z = numpy.zeros(LTR.shape)
Z [LTR == 1] = 1
Z [LTR == 0] = -1
print("RBF Kernel: \n")
for k, C, gamma in values:
    print(f"K: {k} C: {C} : γ: {gamma}")
    alpha_star = rbf_svm(DTR, LTR, C, k, gamma)
    distance = vcol((DTR**2).sum(0)) + vrow((DTE**2).sum(0)) - 2 * numpy.dot(DTR.T, DTE)
    S = numpy.dot(alpha_star * Z, numpy.exp(-gamma*distance) + k**2)
    predictions = numpy.array([1 if score > 0 else 0 for score in S.ravel()])
    error_rate = evaluate_model2(predictions, LTE)[1]
    print("Error Rate: {:.1f}%".format(error_rate * 100))
    print("\n")