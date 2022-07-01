########################
#####-----LAB8-----#####
########################

####LIBRARIES#####
import numpy
import matplotlib.pyplot as plt
import scipy.special

#####USEFUL FUNCTIONS#####

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

# Functions to load Divina Commedia data and split
# them into training and evaluation lists
def load_data():

    lInf = []
    f=open('inferno.txt', encoding="ISO-8859-1")
    for line in f:
        lInf.append(line.strip())
    f.close()

    lPur = []
    f=open('purgatorio.txt', encoding="ISO-8859-1")
    for line in f:
        lPur.append(line.strip())
    f.close()

    lPar = []
    f=open('paradiso.txt', encoding="ISO-8859-1")
    for line in f:
        lPar.append(line.strip())
    f.close()
    
    return lInf, lPur, lPar

def split_data(l, n):

    lTrain, levaluation = [], []
    for i in range(len(l)):
        if i % n == 0:
            levaluation.append(l[i])
        else:
            lTrain.append(l[i])
            
    return lTrain, levaluation

# Function to compute divina commedia predictions
def predict_divina_commedia(lInf, lPur, lPar):
    lInf_train, lInf_evaluation = split_data(lInf, 4)
    lPur_train, lPur_evaluation = split_data(lPur, 4)
    lPar_train, lPar_evaluation = split_data(lPar, 4)
    
    # Creating training words dictionary
    training_words = ' '.join([' '.join(lInf_train), ' '.join(lPur_train), ' '.join(lPar_train)]).split()
    #training_words = sorted(' '.join([' '.join(lInf_train), ' '.join(lPur_train), ' '.join(lPar_train)]).split(' '))
    i = 0
    words_labels = {}
    for word in training_words:
        if(word not in words_labels):
            words_labels[word] = i
            i+=1
    
    # Creating an array of occurrencies for each cantica
    lInf_train_occ = [0 for elem in words_labels]
    for word in ' '.join(lInf_train).split():
        if word in words_labels:
            index = words_labels[word]
            lInf_train_occ[index] += 1
    lPur_train_occ = [0 for elem in words_labels]
    for word in ' '.join(lPur_train).split():
        if word in words_labels:
            index = words_labels[word]
            lPur_train_occ[index] += 1
    lPar_train_occ = [0 for elem in words_labels]
    for word in ' '.join(lPar_train).split():
        if word in words_labels:
            index = words_labels[word]
            lPar_train_occ[index] += 1
            
    # Normalizing occurrencies to get frequencies
    total_Inf = len(' '.join(lInf_train).split())
    total_Pur = len(' '.join(lPur_train).split())
    total_Par = len(' '.join(lPar_train).split())

    lInf_train_occ_norm = numpy.array([(elem + 0.001) / total_Inf for elem in lInf_train_occ])
    lPur_train_occ_norm = numpy.array([(elem + 0.001) / total_Inf for elem in lPur_train_occ])
    lPar_train_occ_norm = numpy.array([(elem + 0.001) / total_Inf for elem in lPar_train_occ])
    
    lInf_train_log = numpy.log(lInf_train_occ_norm)
    lPur_train_log = numpy.log(lPur_train_occ_norm)
    lPar_train_log = numpy.log(lPar_train_occ_norm)
    
    training_model = numpy.zeros((3, len(words_labels)))
    training_model[0] = lInf_train_log
    training_model[1] = lPur_train_log
    training_model[2] = lPar_train_log

    tercets = lInf_evaluation + lPur_evaluation + lPar_evaluation

    SList = []
    for tercet in tercets:
        occ_array = numpy.zeros(len(words_labels))
        for word in (tercet.split()):
            if word in words_labels:
                index = words_labels[word]
                occ_array[index] += 1
        SList.append(numpy.dot(training_model, vcol(occ_array)))
    logS = numpy.hstack(SList)
    
    logSJoint = logS + numpy.log(1/3)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis = 0))
    logSPost = logSJoint - logSMarginal
    
    return numpy.exp(logSPost)

#####LAB FUNCTIONS#####
# Function to compute confusion matrix
# predictions -> Predicted values
# (if Post is true this is the Posterior probability matrix)
# num_classes -> number of possible classes
# LTE -> Testing Dataset correct labels
# Post -> false if predictions are predicted values
# print -> true to print the confusion matrix
def confusion_matrix(predictions, num_classes, LTE, Post=False, printMatrix=False):
    if(Post):
        predictions = numpy.argmax(predictions, axis = 0)
    confusion_matrix = numpy.zeros((num_classes, num_classes))
    zipped = zip(predictions, LTE)
    for i,j in zipped:
        confusion_matrix[i][j] += 1
    if(printMatrix):
        print("Confusion Matrix:\n")
        upper = "\t"+"\t".join(["C="+str(int(i)) for i in range(num_classes)])
        print(upper)
        for i in range(num_classes):
            print("P="+str(i)+"\t"+"\t".join([str(int(elem)) for elem in confusion_matrix[i]]))
        #print(confusion_matrix)
    return confusion_matrix
    
# Function to compute optimal Bayes decisions starting from
# binary log-likelihood ratios
# llr -> binary log-likelihood ratios array
# pi1 -> prior class probability for class 1
# cfn -> cost of predicting 0 when class is 1
# cfn -> cost of predicting 1 when class is 0
def bayes_decisions_binary(llr, pi1, cfn, cfp):
    pi0 = 1 - pi1
    t = -(numpy.log(pi1 * cfn) - numpy.log(pi0 * cfp)) 
    decisions = numpy.array([1 if (elem > t) else 0 for elem in llr])
    
    return decisions

# Function to compute FPR and TPR 
# conf_matrix -> confusion matrix
def compute_fpr_tpr(conf_matrix):
    fp = int(conf_matrix[1][0])
    fn = int(conf_matrix[0][1])
    tp = int(conf_matrix[1][1])
    tn = int(conf_matrix[0][0])
    FNR = fn / (fn + tp)
    TPR = 1 - FNR
    FPR = fp / (fp + tn)
    return FPR, TPR

# Function to compute the Bayes risk from the confusion matrix corresponding to the optimal
# decisions for an application (π1, Cfn, Cfp)
# confusion_matrix -> confusion matrix of the application
# num_classes -> number of possible classes
# LTE -> Testing Dataset correct labels
# pi1 -> prior class probability for class 1
# cfn -> cost of predicting 0 when class is 1
# cfp -> cost of predicting 1 when class is 0
def compute_dcf(confusion_matrix, pi1, cfn, cfp):
    fp = int(confusion_matrix[1][0])
    fn = int(confusion_matrix[0][1])
    tp = int(confusion_matrix[1][1])
    tn = int(confusion_matrix[0][0])
    FNR = fn / (fn + tp)
    FPR = fp / (fp + tn)
    result = pi1 * cfn * FNR + (1 - pi1) * cfp * FPR
    return result

# Function to compute the Bayes risk from the confusion matrix corresponding to the optimal
# decisions for an application (π1, Cfn, Cfp) divided by the risk of an optimal system that does not
# use the test data at all
# confusion_matrix -> confusion matrix of the application
# pi1 -> prior class probability for class 1
# cfn -> cost of predicting 0 when class is 1
# cfp -> cost of predicting 1 when class is 0
def compute_dcf_norm(confusion_matrix, pi1, cfn, cfp):
    fp = int(confusion_matrix[1][0])
    fn = int(confusion_matrix[0][1])
    tp = int(confusion_matrix[1][1])
    tn = int(confusion_matrix[0][0])
    FNR = fn / (fn + tp)
    FPR = fp / (fp + tn)
    result = pi1 * cfn * FNR + (1 - pi1) * cfp * FPR
    dcf_dummy = min([pi1 * cfn, (1 - pi1) * cfp])
    result = result / dcf_dummy
    return result

# Function to compute the min Bayes risk from the confusion matrix corresponding to the optimal
# decisions for an application (π1, Cfn, Cfp) divided by the risk of an optimal system that does not
# use the test data at all
# llr -> binary log-likelihood ratios array
# num_classes -> number of possible classes
# LTE -> Testing Dataset correct labels
# pi1 -> prior class probability for class 1
# cfn -> cost of predicting 0 when class is 1
# cfp -> cost of predicting 1 when class is 0
# thresholds -> list of possible thresholds
def compute_min_dcf(llr, num_classes, LTE, pi1, cfn, cfp, thresholds):
    norm_dcfs = []
    for t in thresholds:
        predictions = numpy.array([1 if elem > t else 0 for elem in llr])
        cm = confusion_matrix(predictions, num_classes, LTE)
        norm_dcf = compute_dcf_norm(cm, pi1, cfn, cfp)
        norm_dcfs.append(norm_dcf)
    return min(norm_dcfs)

#####EXECUTION#####

##Confusion matrices and accuracy

#Loading Iris Dataset
dataset, labels = load_iris('iris.csv')
(DTR, LTR), (DTE, LTE) = split_db_2to1(dataset, labels)
print("Iris dataset confusion matrices:")
print("MVG Classifier")
confusion_matrix(mvg_classifier_log(DTR, LTR, DTE, LTE), 3, LTE, Post=True, printMatrix=True)
print("Tied Gaussian Classifier")
confusion_matrix(tg_classifier_log(DTR, LTR, DTE, LTE), 3, LTE, Post=True, printMatrix=True)
print("\n Divina Commedia confusion matrix:")
lInf, lPur, lPar = load_data()
SPost = predict_divina_commedia(lInf, lPur, lPar)
LTE = numpy.load('commedia_labels.npy')
confusion_matrix(SPost, 3, LTE, Post = True, printMatrix=True)

##Binary task:optimal decisions

llr = numpy.load('commedia_llr_infpar.npy')
LTE = numpy.load('commedia_labels_infpar.npy')
values = [[0.5, 1, 1], [0.8, 1, 1], [0.5, 10, 1], [0.8, 1, 10]]
for pi1, cfn, cfp in values:
    predictions = bayes_decisions_binary(llr, pi1, cfn, cfp)
    print(f"pi1 = {pi1} Cfn = {cfn} Cfp = {cfp}")
    confusion_matrix(predictions, 2, LTE, printMatrix=True)
    print("\n")

##ROC curves

llrs= numpy.load('commedia_llr_infpar.npy')
thresholds = numpy.array(llrs)
thresholds.sort()
thresholds = numpy.concatenate((numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])))

fprs = []
tprs = []
for t in thresholds:
    predictions = numpy.array([1 if elem > t else 0 for elem in llrs])
    LTE = numpy.load('commedia_labels_infpar.npy')
    conf_matrix = confusion_matrix(predictions, 2, LTE)
    FPR, TPR = compute_fpr_tpr(conf_matrix)
    fprs.append(FPR)
    tprs.append(TPR)

plt.title("ROC curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid()
plt.plot(fprs, tprs)
plt.show()

##Binary task: evaluation

llr = numpy.load('commedia_llr_infpar.npy')
LTE = numpy.load('commedia_labels_infpar.npy')
values = [[0.5, 1, 1], [0.8, 1, 1], [0.5, 10, 1], [0.8, 1, 10]]
for pi1, cfn, cfp in values:
    predictions = bayes_decisions_binary(llr, pi1, cfn, cfp)
    cm = confusion_matrix(predictions, 2, LTE)
    dcf = compute_dcf(cm, pi1, cfn, cfp)
    print(f"pi1 = {pi1} Cfn = {cfn} Cfp = {cfp}")
    print(f"DCF = {dcf:.3f}")
    print("\n")

llr = numpy.load('commedia_llr_infpar.npy')
LTE = numpy.load('commedia_labels_infpar.npy')
values = [[0.5, 1, 1], [0.8, 1, 1], [0.5, 10, 1], [0.8, 1, 10]]
for pi1, cfn, cfp in values:
    predictions = bayes_decisions_binary(llr, pi1, cfn, cfp)
    cm = confusion_matrix(predictions, 2, LTE)
    dcf_norm = compute_dcf_norm(cm, pi1, cfn, cfp)
    print(f"pi1 = {pi1} Cfn = {cfn} Cfp = {cfp}")
    print(f"NORMALIZED DCF = {dcf_norm:.3f}")
    print("\n")

##Minimum detection costs

llr = numpy.load('commedia_llr_infpar.npy')
LTE = numpy.load('commedia_labels_infpar.npy')
thresholds = numpy.array(llr)
thresholds.sort()
thresholds = numpy.concatenate((numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])))
values = [[0.5, 1, 1], [0.8, 1, 1], [0.5, 10, 1], [0.8, 1, 10]]
for pi1, cfn, cfp in values:
    min_dcf = compute_min_dcf(llr, 2, LTE, pi1, cfn, cfp, thresholds)
    print(f"pi1 = {pi1} Cfn = {cfn} Cfp = {cfp}")
    print(f"MIN DCF = {min_dcf:.3f}")
    print("\n")

##Bayes error plots

effPriorLogOdds = numpy.linspace(-3, 3, 21) 
effective_priors = 1 / (1 + numpy.exp(-effPriorLogOdds))
llr = numpy.load('commedia_llr_infpar.npy')
LTE = numpy.load('commedia_labels_infpar.npy')
thresholds = numpy.array(llr)
thresholds.sort()
thresholds = numpy.concatenate((numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])))
norm_dcfs = []
min_dcfs = []
for pi_tilde in effective_priors:
    predictions = bayes_decisions_binary(llr, pi_tilde, 1, 1) 
    cm = confusion_matrix(predictions, 2, LTE)
    norm_dcfs.append(compute_dcf_norm(cm, pi_tilde, 1, 1))
    min_dcfs.append(compute_min_dcf(llr, 2, LTE, pi_tilde, 1, 1, thresholds))

plt.title('Bayes Error Plot\n\n')
plt.plot(effPriorLogOdds, norm_dcfs, label='DCF (ε = 0.001)', color='r')
plt.plot(effPriorLogOdds, min_dcfs, label='min DCF (ε = 0.001)', color='b')
plt.ylim([0, 1.1])
plt.xlim([-3, 3])
plt.xlabel('Prior log-odds')
plt.ylabel('DCF value')
plt.legend()
plt.show()

##Comparing recognizers

llr = numpy.load('commedia_llr_infpar.npy')
LTE = numpy.load('commedia_labels_infpar.npy')
thresholds = numpy.array(llr)
thresholds.sort()
thresholds = numpy.concatenate((numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])))
values = [[0.5, 1, 1], [0.8, 1, 1], [0.5, 10, 1], [0.8, 1, 10]]
print("ε = 0.001\n")
for pi1, cfn, cfp in values:
    predictions = bayes_decisions_binary(llr, pi1, cfn, cfp)
    cm = confusion_matrix(predictions, 2, LTE)
    dcf_norm = compute_dcf_norm(cm, pi1, cfn, cfp)
    min_dcf = compute_min_dcf(llr, 2, LTE, pi1, cfn, cfp, thresholds)
    print(f"pi1 = {pi1} Cfn = {cfn} Cfp = {cfp}")
    print(f"NORMALIZED DCF = {dcf_norm:.3f}")
    print(f"MIN DCF = {min_dcf:.3f}")
    print("\n")
llr1 = numpy.load('commedia_llr_infpar_eps1.npy')
LTE1 = numpy.load('commedia_labels_infpar_eps1.npy')
thresholds1 = numpy.array(llr1)
thresholds1.sort()
thresholds1 = numpy.concatenate((numpy.array([-numpy.inf]), thresholds1, numpy.array([numpy.inf])))
values = [[0.5, 1, 1], [0.8, 1, 1], [0.5, 10, 1], [0.8, 1, 10]]
print("\nε = 1\n")
for pi1, cfn, cfp in values:
    predictions = bayes_decisions_binary(llr1, pi1, cfn, cfp)
    cm = confusion_matrix(predictions, 2, LTE1)
    dcf_norm = compute_dcf_norm(cm, pi1, cfn, cfp)
    min_dcf = compute_min_dcf(llr1, 2, LTE1, pi1, cfn, cfp, thresholds)
    print(f"pi1 = {pi1} Cfn = {cfn} Cfp = {cfp}")
    print(f"NORMALIZED DCF = {dcf_norm:.3f}")
    print(f"MIN DCF = {min_dcf:.3f}")
    print("\n")

effPriorLogOdds = numpy.linspace(-3, 3, 21) 
effective_priors = 1 / (1 + numpy.exp(-effPriorLogOdds))
llr = numpy.load('commedia_llr_infpar.npy')
LTE = numpy.load('commedia_labels_infpar.npy')
thresholds = numpy.array(llr)
thresholds.sort()
thresholds = numpy.concatenate((numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])))
norm_dcfs = []
min_dcfs = []
for pi_tilde in effective_priors:
    predictions = bayes_decisions_binary(llr, pi_tilde, 1, 1) 
    cm = confusion_matrix(predictions, 2, LTE)
    norm_dcfs.append(compute_dcf_norm(cm, pi_tilde, 1, 1))
    min_dcfs.append(compute_min_dcf(llr, 2, LTE, pi_tilde, 1, 1, thresholds))

plt.title('Bayes Error Plot\n\n')
plt.plot(effPriorLogOdds, norm_dcfs, label='DCF (ε = 0.001)', color='r')
plt.plot(effPriorLogOdds, min_dcfs, label='min DCF (ε = 0.001)', color='b')


llr = numpy.load('commedia_llr_infpar_eps1.npy')
LTE = numpy.load('commedia_labels_infpar_eps1.npy')
thresholds = numpy.array(llr)
thresholds.sort()
thresholds = numpy.concatenate((numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])))
norm_dcfs = []
min_dcfs = []
for pi_tilde in effective_priors:
    predictions = bayes_decisions_binary(llr, pi_tilde, 1, 1) 
    cm = confusion_matrix(predictions, 2, LTE)
    norm_dcfs.append(compute_dcf_norm(cm, pi_tilde, 1, 1))
    min_dcfs.append(compute_min_dcf(llr, 2, LTE, pi_tilde, 1, 1, thresholds))

plt.plot(effPriorLogOdds, norm_dcfs, label='DCF (ε = 1)', color='y')
plt.plot(effPriorLogOdds, min_dcfs, label='min DCF (ε = 1)', color='g')
plt.ylim([0, 1.1])
plt.xlim([-3, 3])
plt.xlabel('Prior log-odds')
plt.ylabel('DCF value')
plt.legend()
plt.show()



