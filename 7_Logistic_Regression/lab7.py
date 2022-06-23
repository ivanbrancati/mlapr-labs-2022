########################
#####-----LAB7-----#####
########################

####LIBRARIES#####
import scipy.optimize
import numpy
import sklearn.datasets

#####USEFUL FUNCTIONS#####
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

#Function to obtain column vector
def vcol(D):
    return D.reshape(D.size, 1)

#Function to obtain row vector
def vrow(D):
    return D.reshape(1, D.size)

# Function to compute f(y, z) = (y + 3)^2 + sin(y) + (z + 1)^2
# arg -> array of shape (2,) (x, z)
def f(arg):
    a = (arg[0] + 3) ** 2
    b = numpy.sin(arg[0])
    c = (arg[1] + 1) ** 2
    return a + b + c

# Function to compute:
# ( f(y, z) = (y + 3)^2 + sin(y) + (z + 1)^2,
# numpy.array( ∂f(y, z) / ∂y = 2(y + 3) + cos(y) ,
# ∂f(y, z) / ∂z = 2(z + 1) )
# arg -> array of shape (2,) = (y, z)
def f2(arg):
    a = (arg[0] + 3) ** 2
    b = numpy.sin(arg[0])
    c = (arg[1] + 1) ** 2
    
    d = 2 * (arg[0] + 3) + numpy.cos(arg[0])
    e = 2 * (arg[1] + 1)
    
    return a + b + c, numpy.array([d, e])

# Function to load iris dataset 
# (only virginica and versicolor)
def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

# Function to obtain the regularized Logistic Regression Objective
# using #2 expression
# DTR -> Training Dataset
# LTR -> Training Dataset Labels
# l -> lambda parameter
def logreg_obj_wrap(DTR, LTR, l):
    Z = numpy.array([-1 if elem == 0 else 1 for elem  in LTR ]) # if label = 0 -> Z = -1, if label = 1 -> Z = +1
    def logreg_obj(v):
        w, b = vcol(v[0:-1]), v[-1]
        S = numpy.dot(w.T, DTR) + b
        crossEntropy = numpy.logaddexp(0, -Z*S).mean() 
        #computed directly on vectors (0 is broadcasted)
        #mean to do division by n
        return l * 0.5 * numpy.linalg.norm(w) ** 2 + crossEntropy
    return logreg_obj

# Function to evaluate logistic regression classifier
# S -> Scores matrix (S=wT*x+b)
# LTE -> Testing Dataset correct labels
# print -> true to print accuracy and error rate
def evaluate_model_logreg(S, LTE, printResults=False):
	predicted = numpy.array([1 if score > 0 else 0 for score in S.ravel()])

	correct_predictions = (predicted == LTE).sum()
	number_of_samples = LTE.shape[0]

	accuracy = correct_predictions / number_of_samples
	error_rate = (number_of_samples - correct_predictions) / number_of_samples

	if(printResults):
		print("Accuracy: {}%".format(accuracy * 100))
		print("Error Rate: {}%".format(error_rate * 100))
	
	return accuracy, error_rate

#####EXECUTION#####

#Numerical Optimization
print("Numerical optimization of function f:")
a, b, c = scipy.optimize.fmin_l_bfgs_b(f, numpy.array([0,0]), approx_grad=True, iprint=1)
print(a)
print(b)
print(c)
print("\nNumerical optimization of function f with explicit gradient:")
a, b, c = scipy.optimize.fmin_l_bfgs_b(f2, numpy.array([0,0]), iprint=1)
print(a)
print(b)
print(c)

#Binary logistic regression
D, L = load_iris_binary()
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

lambdas = [1e-6, 1e-3, 1e-1, 1]
lambdas_results = {}
for l in lambdas:
    logreg_obj = logreg_obj_wrap(DTR, LTR, l)
    v, J, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0] + 1), approx_grad=True, iprint=1)
    lambdas_results[l]={"J": J, "w": vcol(v[:-1]), "b": v[-1]}

for key, value in lambdas_results.items():
    l = key
    J = value['J']
    w = value['w']
    b = value['b']
    S = numpy.dot(w.T, DTE) + b
    error_rate = evaluate_model_logreg(S, LTE)[1]*100
    print(f"Lambda : {l} -> J(w*,b*) = {J} Error Rate = %.1f %%" %error_rate)
