########################
#####-----LAB6-----#####
########################

####LIBRARIES#####
import numpy
import scipy.special

#####USEFUL FUNCTIONS#####
#Function to obtain column vector
def vcol(D):
    return D.reshape(D.size, 1)

#Function to obtain row vector
def vrow(D):
    return D.reshape(1, D.size)

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


#####EXECUTION#####

# Load the tercets and split the lists in training and evaluation lists
lInf, lPur, lPar = load_data()
lInf_train, lInf_evaluation = split_data(lInf, 4)
lPur_train, lPur_evaluation = split_data(lPur, 4)
lPar_train, lPar_evaluation = split_data(lPar, 4)

#Training the model
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

#Predicting the cantica

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

inf_logSPost = logSPost[:, :400]
pur_logSPost = logSPost[:, 400:802]
par_logSPost = logSPost[:, 802:]

print("Inferno results:")
evaluate_model(inf_logSPost,numpy.array([0 for i in range(400)]),True)
print("\nPurgatorio results:")
evaluate_model(pur_logSPost,numpy.array([1 for i in range(402)]),True)
print("\nParadiso results:")
evaluate_model(par_logSPost,numpy.array([2 for i in range(402)]),True)
print("\nTotal results:")
evaluate_model(logSPost,numpy.array([0 for i in range(400)]+[1 for i in range(402)]+[2 for i in range(402)]),True)

#INFERNO VS PARADISO
# Creating training words dictionary
training_words = ' '.join([' '.join(lInf_train), ' '.join(lPar_train)]).split()

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
lPar_train_occ = [0 for elem in words_labels]
for word in ' '.join(lPar_train).split():
    if word in words_labels:
        index = words_labels[word]
        lPar_train_occ[index] += 1
# Normalizing occurrencies to get frequencies
total_Inf = len(' '.join(lInf_train).split())
total_Par = len(' '.join(lPar_train).split())

lInf_train_occ_norm = numpy.array([(elem + 0.001) / total_Inf for elem in lInf_train_occ])
lPar_train_occ_norm = numpy.array([(elem + 0.001) / total_Inf for elem in lPar_train_occ])

lInf_train_log = numpy.log(lInf_train_occ_norm)
lPar_train_log = numpy.log(lPar_train_occ_norm)
training_model = numpy.zeros((2, len(words_labels)))
training_model[0] = lInf_train_log
training_model[1] = lPar_train_log

tercets = lInf_evaluation + lPar_evaluation

SList = []
for tercet in tercets:
    occ_array = numpy.zeros(len(words_labels))
    for word in (tercet.split()):
        if word in words_labels:
            index = words_labels[word]
            occ_array[index] += 1
    SList.append(numpy.dot(training_model, vcol(occ_array)))
logS = numpy.hstack(SList)
logSJoint = logS + numpy.log(1/2)
logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis = 0))
logSPost = logSJoint - logSMarginal
print("\nInferno VS Paradiso results:")
evaluate_model(logSPost,numpy.array([0 for i in range(400)]+[1 for i in range(402)]),True)

#INFERNO VS PURGATORIO
# Creating training words dictionary
training_words = ' '.join([' '.join(lInf_train), ' '.join(lPur_train)]).split()
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
# Normalizing occurrencies to get frequencies
total_Inf = len(' '.join(lInf_train).split())
total_Pur = len(' '.join(lPur_train).split())

lInf_train_occ_norm = numpy.array([(elem + 0.001) / total_Inf for elem in lInf_train_occ])
lPur_train_occ_norm = numpy.array([(elem + 0.001) / total_Inf for elem in lPur_train_occ])

lInf_train_log = numpy.log(lInf_train_occ_norm)
lPur_train_log = numpy.log(lPur_train_occ_norm)
training_model = numpy.zeros((2, len(words_labels)))
training_model[0] = lInf_train_log
training_model[1] = lPur_train_log

tercets = lInf_evaluation + lPur_evaluation 

SList = []
for tercet in tercets:
    occ_array = numpy.zeros(len(words_labels))
    for word in (tercet.split()):
        if word in words_labels:
            index = words_labels[word]
            occ_array[index] += 1
    SList.append(numpy.dot(training_model, vcol(occ_array)))
logS = numpy.hstack(SList)
logSJoint = logS + numpy.log(1/2)
logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis = 0))
logSPost = logSJoint - logSMarginal
print("\nInferno VS Purgatorio results:")
evaluate_model(logSPost,numpy.array([0 for i in range(400)]+[1 for i in range(402)]),True)

#PURGATORIO VS PARADISO
# Creating training words dictionary
training_words = ' '.join([' '.join(lPur_train), ' '.join(lPar_train)]).split()
i = 0
words_labels = {}
for word in training_words:
    if(word not in words_labels):
        words_labels[word] = i
        i+=1
# Creating an array of occurrencies for each cantica
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
total_Pur = len(' '.join(lPur_train).split())
total_Par = len(' '.join(lPar_train).split())

lPur_train_occ_norm = numpy.array([(elem + 0.001) / total_Pur for elem in lPur_train_occ])
lPar_train_occ_norm = numpy.array([(elem + 0.001) / total_Par for elem in lPar_train_occ])

lPur_train_log = numpy.log(lPur_train_occ_norm)
lPar_train_log = numpy.log(lPar_train_occ_norm)
training_model = numpy.zeros((2, len(words_labels)))
training_model[0] = lPur_train_log
training_model[1] = lPar_train_log

tercets = lPur_evaluation + lPar_evaluation 

SList = []
for tercet in tercets:
    occ_array = numpy.zeros(len(words_labels))
    for word in (tercet.split()):
        if word in words_labels:
            index = words_labels[word]
            occ_array[index] += 1
    SList.append(numpy.dot(training_model, vcol(occ_array)))
logS = numpy.hstack(SList)
logSJoint = logS + numpy.log(1/2)
logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis = 0))
logSPost = logSJoint - logSMarginal
print("\nPurgatorio VS Paradiso results:")
evaluate_model(logSPost,numpy.array([0 for i in range(402)]+[1 for i in range(402)]),True)