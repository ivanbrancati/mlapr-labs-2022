########################
#####-----LAB3-----#####
########################

#####LIBRARIES#####
import numpy
import matplotlib.pyplot as plt
import scipy.linalg

#####USEFUL FUNCTIONS#####

#Function to load iris dataset
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

#Function to obtain column vector
def vcol(D):
    return D.reshape(D.shape[0], 1)

#Function to obtain row vector
def vrow(D):
    return D.reshape(1, D.shape[1])

#Function to compute dataset D mean with braodcasting
#returns one-dimension vector
def dataset_mean_braodcasting(D):
    return D.mean(1)

#Function to center dataset D
def center_data(D):
    DC = D - D.mean(1).reshape(D.shape[0], 1)
    return DC

#Function to compute covariance matrix
def covariance_matrix(centered_dataset):
    DC = centered_dataset
    N = centered_dataset.shape[1]
    C = C = (1/N)*numpy.dot(DC, DC.T)
    return C

#Function to compute PCA with eigh method
def pca_eigh(dataset, m):
    D = dataset
    C = covariance_matrix(center_data(dataset))
    s, U = numpy.linalg.eigh(C) 
    P = U[:, ::-1][:, 0:m]
    return P

#Function to compute PCA with SVD method
def pca_svd(dataset, m):
    D = dataset
    C = covariance_matrix(center_data(dataset))
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return P

#Function to compute LDA with Generalized eigenvalue problem
def lda_eigh(dataset, number_of_labels, m):
    mu = vcol(dataset_mean_braodcasting(dataset))
    swc = [covariance_matrix(center_data(dataset[:, labels == i])) for i in range(0, number_of_labels)]
    nc = [dataset[:, labels == i].shape[1] for i in range(0, number_of_labels)] 
    muc = [vcol(dataset_mean_braodcasting(dataset[:, labels == i])) for i in range(0, number_of_labels)]
    
    N = dataset.shape[1]

    #between class covariance
    SB = numpy.zeros([dataset.shape[0], dataset.shape[0]], dtype = numpy.float64)

    for i in range(0, len(muc)):
        SB += nc[i]*numpy.dot((muc[i]-mu),(muc[i]-mu).T)
    SB = SB/N

    #within class covariance
    SW = numpy.zeros([dataset.shape[0], dataset.shape[0]], dtype = numpy.float64)

    for i in range(0, len(swc)):
        SW += (nc[i] * swc[i])
    SW = SW/N
    
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    
    dataset_with_lda = numpy.dot(W.T, dataset)
    
    return W, dataset_with_lda

#Function to compute LDA with joint diagonalization of SB and SW
def lda_diag(dataset, number_of_labels, m):
    mu = vcol(dataset_mean_braodcasting(dataset))
    swc = [covariance_matrix(center_data(dataset[:, labels == i])) for i in range(0, number_of_labels)]
    nc = [dataset[:, labels == i].shape[1] for i in range(0, number_of_labels)] 
    muc = [vcol(dataset_mean_braodcasting(dataset[:, labels == i])) for i in range(0, number_of_labels)]

    N = dataset.shape[1]

    #between class covariance
    SB = numpy.zeros([dataset.shape[0], dataset.shape[0]], dtype = numpy.float64)

    for i in range(0, len(muc)):
        SB += nc[i]*numpy.dot((muc[i]-mu),(muc[i]-mu).T)
    SB = SB/N

    #within class covariance
    SW = numpy.zeros([dataset.shape[0], dataset.shape[0]], dtype = numpy.float64)

    for i in range(0, len(swc)):
        SW += nc[i] * swc[i]
    SW = SW/N
    
    U, s, _ = numpy.linalg.svd(SW)
    P1 = numpy.dot( numpy.dot(U, numpy.diag(1.0/(s**0.5))), U.T )
    SBT = numpy.dot(P1, numpy.dot(SB, P1.T))
    P2 = numpy.linalg.eigh(SBT)[1][:, ::-1][:, 0:m]
    W = numpy.dot(P1.T, P2)

    dataset_with_lda = numpy.dot(W.T, dataset)

    return W, dataset_with_lda

#####EXECUTION#####

dataset, labels = load_iris("iris.csv")
mu = vcol(dataset_mean_braodcasting(dataset))
C = covariance_matrix(center_data(dataset))

#---PCA---#

#Checking function correctness
pca1 = pca_eigh(dataset, 4)
pca2 = pca_svd(dataset, 4)
sol = numpy.load('IRIS_PCA_matrix_m4.npy')
print("Eigh function:")
print (pca1)
print("SVD function")
print(pca2)
print("Solution")
print(sol)

#Plotting results
P1 = pca_eigh(dataset, 2)
P2 = pca_svd(dataset, 2)
D = dataset
DP1 = numpy.dot(P1.T, D)
DP2 = numpy.dot(P2.T, D)

setosa1 = DP1[:, labels == 0]
versicolor1 = DP1[:, labels == 1]
virginica1 = DP1[:, labels == 2]
setosa2 = DP2[:, labels == 0]
versicolor2 = DP2[:, labels == 1]
virginica2 = DP2[:, labels == 2]

plt.figure(figsize=(7,6)).suptitle("PCA with eigh function")
plt.scatter(setosa1[0], setosa1[1])
plt.scatter(versicolor1[0], versicolor1[1], color = "orange")
plt.scatter(virginica1[0], virginica1[1], color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.show()

plt.figure(figsize=(7,6)).suptitle("PCA with SVD function")
plt.scatter(setosa2[0], setosa2[1])
plt.scatter(versicolor2[0], versicolor2[1], color = "orange")
plt.scatter(virginica2[0], virginica2[1], color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.show()

#---LDA---#

lda_matrix1, dataset_lda1 = lda_eigh(dataset, 3, 2)
lda_matrix2, dataset_lda2 = lda_diag(dataset, 3, 2)

setosa1 = dataset_lda1[:, labels == 0]
versicolor1 = dataset_lda1[:, labels == 1]
virginica1 = dataset_lda1[:, labels == 2]
setosa2 = dataset_lda2[:, labels == 0]
versicolor2 = dataset_lda2[:, labels == 1]
virginica2 = dataset_lda2[:, labels == 2]

plt.figure(figsize=(7,6)).suptitle("LDA with Generalized eigenvalue problem")
plt.scatter(setosa1[0], setosa1[1])
plt.scatter(versicolor1[0], versicolor1[1], color = "orange")
plt.scatter(virginica1[0], virginica1[1], color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.show()

plt.figure(figsize=(7,6)).suptitle("PCA with Joint diagonalization of SB and SW")
plt.scatter(setosa2[0], setosa2[1])
plt.scatter(versicolor2[0], versicolor2[1], color = "orange")
plt.scatter(virginica2[0], virginica2[1], color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.show()