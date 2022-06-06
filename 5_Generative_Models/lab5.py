########################
#####-----LAB5-----#####
########################

####LIBRARIES#####
import numpy
import matplotlib.pyplot as plt
import scipy.special

#####USEFUL FUNCTIONS#####
#Function to obtain column vector
def vcol(D):
    return D.reshape(D.size, 1)

#Function to obtain row vector
def vrow(D):
    return D.reshape(1, D.size)

#Function to compute dataset D mean with broadcasting
#returns one-dimension vector
def dataset_mean_broadcasting(D):
    return D.mean(1)

#Function to center dataset D
def center_data(D):
    DC = D - D.mean(1).reshape(D.shape[0], 1)
    return DC

def covariance_matrix(centered_dataset):
    DC = centered_dataset
    N = centered_dataset.shape[1]
    C = C = (1/N)*numpy.dot(DC, DC.T)
    return C

#Function to compute log-density for a dataset X
#--X is a dataset with shape (M,N)
#--mu is the dataset mean with shape (M, 1)
#--C is the dataset covariance matrix of shape (M,M)
#Version with broadcasting
def logpdf_GAU_ND_2(X, mu, C):
    M = X.shape[0]
    sigma_inv = numpy.linalg.inv(C)
    sigma_log_mod = numpy.linalg.slogdet(sigma_inv)[1]
    
    
    logpdf = -M/2 * numpy.log(2*numpy.pi) + sigma_log_mod/2 - ((X-mu) * numpy.dot(sigma_inv, (X-mu))).sum(0)/2
    
    return logpdf

#Function to load Iris Dataset
def load_iris(filename):
    #Iris-setosa -> 0, Iris-versicolor -> 1, Iris-virginica -> 2
    iris_dict = {
                    "Iris-setosa" : 0,
                    "Iris-versicolor" : 1, 
                    "Iris-virginica" : 2  
    }

    dataset_list = []
    labels_list = []

    file = open(filename, "r")
    for line in file:
        dataset_list.append(numpy.array(line.split(",")[:-1], dtype = numpy.float64).reshape(4,1))
        labels_list.append(iris_dict[line.rstrip().split(",")[-1]])
    dataset_matrix = numpy.hstack(dataset_list)
    labels_matrix = numpy.array(labels_list)
    return dataset_matrix, labels_matrix

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

#####CLASSIFIERS FUNCTIONS#####

# Function to compute class posterior probability matrix
# for Multivariate Gaussian Classifier
# (Equal prior probability)
# DTR -> training dataset
# LTR -> training dataset labels
# DTE -> test dataset
# LTE -> test dataset labels
def mvg_classifier(DTR, LTR, DTE, LTE):
    classes = list(set(sorted(LTR)))
	
    #class-conditional probabilities in score matrix S
    S_list = []
    for label in classes:
        S_list.append(numpy.exp(logpdf_GAU_ND_2(DTE, vcol(dataset_mean_broadcasting(DTR[:, LTR == label])), covariance_matrix(center_data(DTR[:, LTR == label])))))
    S = numpy.vstack(tuple(S_list))

    #join densities
    SJoint = S/len(classes)

    #class posterior probabilities
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal

    return SPost

# Function to compute class posterior probability matrix (of log-densities)
# for Multivariate Gaussian Classifier
# (Equal prior probability)
# DTR -> training dataset
# LTR -> training dataset labels
# DTE -> test dataset
# LTE -> test dataset labels
def mvg_classifier_log(DTR, LTR, DTE, LTE):
    classes = list(set(sorted(LTR)))
	
    #class-conditional probabilities in score matrix S
    S_list = []
    for label in classes:
        S_list.append(logpdf_GAU_ND_2(DTE, vcol(dataset_mean_broadcasting(DTR[:, LTR == label])), covariance_matrix(center_data(DTR[:, LTR == label]))))
    logS = numpy.vstack(tuple(S_list))

    #join densities
    logSJoint = logS + numpy.log(1/len(classes))

    #class posterior probabilities
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis = 0))
    logSPost = logSJoint - logSMarginal

    return logSPost

# Function to compute class posterior probability matrix
# for Multivariate Gaussian Classifier with Naive Bayes assumption 
# (Equal prior probability)
# DTR -> training dataset
# LTR -> training dataset labels
# DTE -> test dataset
# LTE -> test dataset labels
def mvg_classifier_naivebayes(DTR, LTR, DTE, LTE):
    classes = list(set(sorted(LTR)))
	
    #class-conditional probabilities in score matrix S
    S_list = []
    for label in classes:
        S_list.append(numpy.exp(logpdf_GAU_ND_2(DTE, vcol(dataset_mean_broadcasting(DTR[:, LTR == label])), covariance_matrix(center_data(DTR[:, LTR == label]))*numpy.identity(DTR.shape[0]))))
    S = numpy.vstack(tuple(S_list))

    #join densities
    SJoint = S/len(classes)

    #class posterior probabilities
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal

    return SPost

# Function to compute class posterior probability matrix (of log-densities)
# for Multivariate Gaussian Classifier with Naive Bayes assumption 
# (Equal prior probability)
# DTR -> training dataset
# LTR -> training dataset labels
# DTE -> test dataset
# LTE -> test dataset labels
def mvg_classifier_log_naivebayes(DTR, LTR, DTE, LTE):
    classes = list(set(sorted(LTR)))
	
    #class-conditional probabilities in score matrix S
    S_list = []
    for label in classes:
        S_list.append(logpdf_GAU_ND_2(DTE, vcol(dataset_mean_broadcasting(DTR[:, LTR == label])), covariance_matrix(center_data(DTR[:, LTR == label]))*numpy.identity(DTR.shape[0])))
    logS = numpy.vstack(tuple(S_list))

    #join densities
    logSJoint = logS + numpy.log(1/len(classes))

    #class posterior probabilities
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis = 0))
    logSPost = logSJoint - logSMarginal

    return logSPost

# Function to compute class posterior probability matrix
# for Tied Gaussian Classifier
# (Equal prior probability)
# DTR -> training dataset
# LTR -> training dataset labels
# DTE -> test dataset
# LTE -> test dataset labels
def tg_classifier(DTR, LTR, DTE, LTE):

    classes = list(set(sorted(LTR)))

    #within class covariance
    covariances = [covariance_matrix(center_data(DTR[:, LTR == label])) for label in classes]
    classes_length = [DTR[:, LTR == label].shape[1] for label in classes]
    N = DTR.shape[1]

    sigma_star = numpy.zeros([DTR.shape[0], DTR.shape[0]], dtype = numpy.float64)

    for i in range(0, len(covariances)):
        sigma_star = sigma_star + classes_length[i] * covariances[i]
    sigma_star = sigma_star/N

    #class-conditional probabilities in score matrix S
    S_list = []
    for label in classes:
        S_list.append(numpy.exp(logpdf_GAU_ND_2(DTE, vcol(dataset_mean_broadcasting(DTR[:, LTR == label])), sigma_star)))
    S = numpy.vstack(tuple(S_list))

    #join densities
    SJoint = S/len(classes)

    #class posterior probabilities
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal

    return SPost

# Function to compute class posterior probability matrix (of log-densities)
# for Tied Gaussian Classifier
# (Equal prior probability)
# DTR -> training dataset
# LTR -> training dataset labels
# DTE -> test dataset
# LTE -> test dataset labels
def tg_classifier_log(DTR, LTR, DTE, LTE):

    classes = list(set(sorted(LTR)))

    #within class covariance
    covariances = [covariance_matrix(center_data(DTR[:, LTR == label])) for label in classes]
    classes_length = [DTR[:, LTR == label].shape[1] for label in classes]
    N = DTR.shape[1]

    sigma_star = numpy.zeros([DTR.shape[0], DTR.shape[0]], dtype = numpy.float64)

    for i in range(0, len(covariances)):
        sigma_star = sigma_star + classes_length[i] * covariances[i]
    sigma_star = sigma_star/N

    #class-conditional probabilities in score matrix S
    S_list = []
    for label in classes:
        S_list.append(logpdf_GAU_ND_2(DTE, vcol(dataset_mean_broadcasting(DTR[:, LTR == label])), sigma_star))
    logS = numpy.vstack(tuple(S_list))

    #join densities
    logSJoint = logS + numpy.log(1/len(classes))

    #class posterior probabilities
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis = 0))
    logSPost = logSJoint - logSMarginal

    return logSPost

# Function to compute class posterior probability matrix
# for Tied Gaussian Classifier with Naive Bayes assumption
# (Equal prior probability)
# DTR -> training dataset
# LTR -> training dataset labels
# DTE -> test dataset
# LTE -> test dataset labels
def tg_classifier_naivebayes(DTR, LTR, DTE, LTE):

    classes = list(set(sorted(LTR)))

    #within class covariance
    covariances = [covariance_matrix(center_data(DTR[:, LTR == label])) * numpy.identity(DTR.shape[0]) for label in classes]
    classes_length = [DTR[:, LTR == label].shape[1] for label in classes]
    N = DTR.shape[1]

    sigma_star = numpy.zeros([DTR.shape[0], DTR.shape[0]], dtype = numpy.float64)

    for i in range(0, len(covariances)):
        sigma_star = sigma_star + classes_length[i] * covariances[i]
    sigma_star = sigma_star/N

    #class-conditional probabilities in score matrix S
    S_list = []
    for label in classes:
        S_list.append(numpy.exp(logpdf_GAU_ND_2(DTE, vcol(dataset_mean_broadcasting(DTR[:, LTR == label])), sigma_star)))
    S = numpy.vstack(tuple(S_list))

    #join densities
    SJoint = S/len(classes)

    #class posterior probabilities
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal

    return SPost

# Function to compute class posterior probability matrix (of log-densities)
# for Tied Gaussian Classifier with Naive Bayes assumption
# (Equal prior probability)
# DTR -> training dataset
# LTR -> training dataset labels
# DTE -> test dataset
# LTE -> test dataset labels
def tg_classifier_log_naivebayes(DTR, LTR, DTE, LTE):

    classes = list(set(sorted(LTR)))

    #within class covariance
    covariances = [covariance_matrix(center_data(DTR[:, LTR == label])) * numpy.identity(DTR.shape[0]) for label in classes]
    classes_length = [DTR[:, LTR == label].shape[1] for label in classes]
    N = DTR.shape[1]

    sigma_star = numpy.zeros([DTR.shape[0], DTR.shape[0]], dtype = numpy.float64)

    for i in range(0, len(covariances)):
        sigma_star = sigma_star + classes_length[i] * covariances[i]
    sigma_star = sigma_star/N

    #class-conditional probabilities in score matrix S
    S_list = []
    for label in classes:
        S_list.append(logpdf_GAU_ND_2(DTE, vcol(dataset_mean_broadcasting(DTR[:, LTR == label])), sigma_star))
    logS = numpy.vstack(tuple(S_list))

    #join densities
    logSJoint = logS + numpy.log(1/len(classes))

    #class posterior probabilities
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis = 0))
    logSPost = logSJoint - logSMarginal

    return logSPost

# Function to evaluate a model
# Post -> Class posterior probability matrix
# LTE -> Testing Dataset correct labels
# print -> true to print accuracy and error rate
def evaluate_model(Post, LTE, printResults=False):
	predicted = numpy.argmax(Post, axis = 0)

	correct_predictions = (predicted == LTE).sum()
	number_of_samples = LTE.shape[0]

	accuracy = correct_predictions / number_of_samples
	error_rate = (number_of_samples - correct_predictions) / number_of_samples

	if(printResults):
		print("Accuracy: {}%".format(accuracy * 100))
		print("Error Rate: {}%".format(error_rate * 100))
	
	return accuracy, error_rate

# Function to apply k-fold cross validation
# k is the number of non overlapping subsets
# k-1 is the training dataset size
# classifier is the applied classifier
def apply_kfold(dataset, labels, k, classifier):
    result_model = numpy.empty((len(set(labels)), dataset.shape[1]))
   
    #split in k-folds and apply classifier
    for i in range(k):
        idx = [i for i in range (dataset.shape[1])]
        for j in range(i, int(i+1*dataset.shape[1]/k)):
            idx.remove(i)
        idxTrain=numpy.array(idx)
        idxTest = numpy.array([j for j in range(i, int(i+1*dataset.shape[1]/k))])
        DTR = dataset[:, idxTrain]
        DTE = dataset[:, idxTest]
        LTR = labels[idxTrain]
        LTE = labels[idxTest]
        result_model[:, i] = classifier(DTR, LTR, DTE, LTE).ravel()
    result_model = numpy.array(result_model)
    
    return result_model

#####EXECUTION#####

#Loading Iris Dataset
dataset, labels = load_iris('iris.csv')

#We split the datasets in two parts: the first part will be used for model training, the second for evaluation (validation set).
# DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
(DTR, LTR), (DTE, LTE) = split_db_2to1(dataset, labels)
#(We use 100 samples for training and 50 samples for evaluation.)

#Multivariate Gaussian Classifier
print("Multivariate Gaussian Classifier")
evaluate_model(mvg_classifier(DTR, LTR, DTE, LTE), LTE, True)
print("Multivariate Gaussian Classifier(logarithmic)")
evaluate_model(mvg_classifier_log(DTR, LTR, DTE, LTE), LTE, True)

#Naive Bayes Classifier
print("Nayve Bayes Classifier")
evaluate_model(mvg_classifier_naivebayes(DTR, LTR, DTE, LTE), LTE, True)
print("Naive Bayes Classifier(logarithmic)")
evaluate_model(mvg_classifier_log_naivebayes(DTR, LTR, DTE, LTE), LTE, True)

#Tied Covariance Gaussian Classifier
print("Tied Covariance Gaussian Classifier")
evaluate_model(tg_classifier(DTR, LTR, DTE, LTE), LTE, True)
print("Tied Covariance Gaussian Classifier(logarithmic)")
evaluate_model(tg_classifier_log(DTR, LTR, DTE, LTE), LTE, True)

#Tied Naive Bayes
print("Tied Covariance Gaussian Classifier with Naive bayes assumption")
evaluate_model(tg_classifier_naivebayes(DTR, LTR, DTE, LTE), LTE, True)
print("Tied Covariance Gaussian Classifier with Naive bayes assumption(logarithmic)")
evaluate_model(tg_classifier_log_naivebayes(DTR, LTR, DTE, LTE), LTE, True)

#K-fold cross validation
print("\nLeave-One-Out approach:\n")
print("Multivariate Gaussian Model:")
evaluate_model(apply_kfold(dataset, labels, 150, mvg_classifier_log), labels, True)
print("\nNaive Bayes Model:")
evaluate_model(apply_kfold(dataset, labels, 150, mvg_classifier_log_naivebayes), labels, True)
print("\nTied Covariance Model:")
evaluate_model(apply_kfold(dataset, labels, 150, tg_classifier_log), labels, True)
print("\nTied Naive Bayes Model:")
evaluate_model(apply_kfold(dataset, labels, 150, tg_classifier_log_naivebayes), labels, True)