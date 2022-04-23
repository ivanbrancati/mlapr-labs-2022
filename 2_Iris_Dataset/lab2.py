########################
#####-----LAB2-----#####
########################

#####LIBRARIES#####
import sys
import numpy
import matplotlib.pyplot as plt

#####EXECUTION#####
#---Reading file---#
input = open("iris.csv", "r") 

#---Loading the dataset---#
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

dataset, labels = load_iris("iris.csv")

#---Visualizing the dataset---#
setosa = dataset[:, labels == 0]
versicolor = dataset[:, labels == 1]
virginica = dataset[:, labels == 2]

#SEPAL LENGTH
plt.figure(figsize=(20, 3))
plt.subplot(141)
plt.hist(setosa[0], bins = 10, density = True, alpha= 0.5)
plt.hist(versicolor[0], bins = 10, density = True, color = "orange", alpha= 0.5)
plt.hist(virginica[0], bins = 10, density = True, color = "green", alpha= 0.5)
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal length")

#SEPAL WIDTH
plt.subplot(142)
plt.hist(setosa[1], bins = 10, density = True, alpha= 0.5)
plt.hist(versicolor[1], bins = 10, density = True, color = "orange", alpha= 0.5)
plt.hist(virginica[1], bins = 10, density = True, color = "green", alpha= 0.5)
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal width")

#PETAL LENGTH
plt.subplot(143)
plt.hist(setosa[2], bins = 10, density = True, alpha= 0.5)
plt.hist(versicolor[2], bins = 10, density = True, color = "orange", alpha= 0.5)
plt.hist(virginica[2], bins = 10, density = True, color = "green", alpha= 0.5)
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal length")

#PETAL WIDTH
plt.subplot(144)
plt.hist(setosa[3], bins = 10, density = True, alpha= 0.5)
plt.hist(versicolor[3], bins = 10, density = True, color = "orange", alpha= 0.5)
plt.hist(virginica[3], bins = 10, density = True, color = "green", alpha= 0.5)
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal width")
plt.show()

fig = plt.figure(figsize=(20, 20))
plt.subplot(4, 4, 1)
fig.suptitle("Dataset Visualization", fontsize = '22')

#SEPAL LENGTH
plt.hist(setosa[0], bins = 10, density = True, alpha= 0.5)
plt.hist(versicolor[0], bins = 10, density = True, color = "orange", alpha= 0.5)
plt.hist(virginica[0], bins = 10, density = True, color = "green", alpha= 0.5)
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal length")

plt.subplot(4, 4, 2)
plt.scatter(setosa[0], setosa[1], alpha = 0.5)
plt.scatter(versicolor[0], versicolor[1], alpha = 0.5, color = "orange")
plt.scatter(virginica[0], virginica[1], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

plt.subplot(4, 4, 3)
plt.scatter(setosa[0], setosa[2], alpha = 0.5)
plt.scatter(versicolor[0], versicolor[2], alpha = 0.5, color = "orange")
plt.scatter(virginica[0], virginica[2], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal length")
plt.ylabel("Petal length")

plt.subplot(4, 4, 4)
plt.scatter(setosa[0], setosa[3], alpha = 0.5)
plt.scatter(versicolor[0], versicolor[3], alpha = 0.5, color = "orange")
plt.scatter(virginica[0], virginica[3], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal length")
plt.ylabel("Petal width")

#SEPAL WIDTH
plt.subplot(4, 4, 5)
plt.scatter(setosa[1], setosa[0], alpha = 0.5)
plt.scatter(versicolor[1], versicolor[0], alpha = 0.5, color = "orange")
plt.scatter(virginica[1], virginica[0], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal width")
plt.ylabel("Sepal length")

plt.subplot(4, 4, 6)
plt.hist(setosa[1], bins = 10, density = True, alpha= 0.5)
plt.hist(versicolor[1], bins = 10, density = True, color = "orange", alpha= 0.5)
plt.hist(virginica[1], bins = 10, density = True, color = "green", alpha= 0.5)
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal width")

plt.subplot(4, 4, 7)
plt.scatter(setosa[1], setosa[2], alpha = 0.5)
plt.scatter(versicolor[1], versicolor[2], alpha = 0.5, color = "orange")
plt.scatter(virginica[1], virginica[2], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal width")
plt.ylabel("Petal length")

plt.subplot(4, 4, 8)
plt.scatter(setosa[1], setosa[3], alpha = 0.5)
plt.scatter(versicolor[1], versicolor[3], alpha = 0.5, color = "orange")
plt.scatter(virginica[1], virginica[3], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal width")
plt.ylabel("Petal width")

#PETAL LENGTH
plt.subplot(4, 4, 9)
plt.scatter(setosa[2], setosa[0], alpha = 0.5)
plt.scatter(versicolor[2], versicolor[0], alpha = 0.5, color = "orange")
plt.scatter(virginica[2], virginica[0], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal length")
plt.ylabel("Sepal length")

plt.subplot(4, 4, 10)
plt.scatter(setosa[2], setosa[1], alpha = 0.5)
plt.scatter(versicolor[2], versicolor[1], alpha = 0.5, color = "orange")
plt.scatter(virginica[2], virginica[1], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal length")
plt.ylabel("Sepal width")

plt.subplot(4, 4, 11)
plt.hist(setosa[2], bins = 10, density = True, alpha= 0.5)
plt.hist(versicolor[2], bins = 10, density = True, color = "orange", alpha= 0.5)
plt.hist(virginica[2], bins = 10, density = True, color = "green", alpha= 0.5)
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal length")

plt.subplot(4, 4, 12)
plt.scatter(setosa[2], setosa[3], alpha = 0.5)
plt.scatter(versicolor[2], versicolor[3], alpha = 0.5, color = "orange")
plt.scatter(virginica[2], virginica[3], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal length")
plt.ylabel("Petal width")

#PETAL WIDTH
plt.subplot(4, 4, 13)
plt.scatter(setosa[3], setosa[0], alpha = 0.5)
plt.scatter(versicolor[3], versicolor[0], alpha = 0.5, color = "orange")
plt.scatter(virginica[3], virginica[0], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal width")
plt.ylabel("Sepal length")

plt.subplot(4, 4, 14)
plt.scatter(setosa[3], setosa[1], alpha = 0.5)
plt.scatter(versicolor[3], versicolor[1], alpha = 0.5, color = "orange")
plt.scatter(virginica[3], virginica[1], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal width")
plt.ylabel("Sepal width")

plt.subplot(4, 4, 15)
plt.scatter(setosa[3], setosa[2], alpha = 0.5)
plt.scatter(versicolor[3], versicolor[2], alpha = 0.5, color = "orange")
plt.scatter(virginica[3], virginica[2], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal width")
plt.ylabel("Petal length")

plt.subplot(4, 4, 16)
plt.hist(setosa[3], bins = 10, density = True, alpha= 0.5)
plt.hist(versicolor[3], bins = 10, density = True, color = "orange", alpha= 0.5)
plt.hist(virginica[3], bins = 10, density = True, color = "green", alpha= 0.5)
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal width")


plt.show()

#---Statistics---#

#Functions

#Function to compute dataset D mean with for loop
#returns column vector
def dataset_mean_loop(D):
    mu = 0
    for i in range(D.shape[1]):
        mu = mu + D[:, i:i+1]
    
    mu = mu / float(D.shape[1])
    return mu

#Function to compute dataset D mean with braodcasting
#returns one-dimension vector
def dataset_mean_braodcasting(D):
    return D.mean(1)

#Function to center dataset D
def center_data(D):
    DC = D - D.mean(1).reshape(D.shape[0], 1)
    return DC

#Function to obtain column vector
def mcol(D):
    return D.reshape(D.shape[0], 1)

#Function to obtain row vector
def mrow(D):
    return D.reshape(1, D.shape[1])

#---Centered Dataset Visualization---#
#centered dataset
dataset = center_data(dataset)
setosa = dataset[:, labels == 0]
versicolor = dataset[:, labels == 1]
virginica = dataset[:, labels == 2]

fig = plt.figure(figsize=(20, 20))
plt.subplot(4, 4, 1)
fig.suptitle("Centered Dataset Visualization", fontsize = '22')

#SEPAL LENGTH
plt.hist(setosa[0], bins = 10, density = True, alpha= 0.5)
plt.hist(versicolor[0], bins = 10, density = True, color = "orange", alpha= 0.5)
plt.hist(virginica[0], bins = 10, density = True, color = "green", alpha= 0.5)
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal length")

plt.subplot(4, 4, 2)
plt.scatter(setosa[0], setosa[1], alpha = 0.5)
plt.scatter(versicolor[0], versicolor[1], alpha = 0.5, color = "orange")
plt.scatter(virginica[0], virginica[1], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

plt.subplot(4, 4, 3)
plt.scatter(setosa[0], setosa[2], alpha = 0.5)
plt.scatter(versicolor[0], versicolor[2], alpha = 0.5, color = "orange")
plt.scatter(virginica[0], virginica[2], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal length")
plt.ylabel("Petal length")

plt.subplot(4, 4, 4)
plt.scatter(setosa[0], setosa[3], alpha = 0.5)
plt.scatter(versicolor[0], versicolor[3], alpha = 0.5, color = "orange")
plt.scatter(virginica[0], virginica[3], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal length")
plt.ylabel("Petal width")

#SEPAL WIDTH
plt.subplot(4, 4, 5)
plt.scatter(setosa[1], setosa[0], alpha = 0.5)
plt.scatter(versicolor[1], versicolor[0], alpha = 0.5, color = "orange")
plt.scatter(virginica[1], virginica[0], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal width")
plt.ylabel("Sepal length")

plt.subplot(4, 4, 6)
plt.hist(setosa[1], bins = 10, density = True, alpha= 0.5)
plt.hist(versicolor[1], bins = 10, density = True, color = "orange", alpha= 0.5)
plt.hist(virginica[1], bins = 10, density = True, color = "green", alpha= 0.5)
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal width")

plt.subplot(4, 4, 7)
plt.scatter(setosa[1], setosa[2], alpha = 0.5)
plt.scatter(versicolor[1], versicolor[2], alpha = 0.5, color = "orange")
plt.scatter(virginica[1], virginica[2], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal width")
plt.ylabel("Petal length")

plt.subplot(4, 4, 8)
plt.scatter(setosa[1], setosa[3], alpha = 0.5)
plt.scatter(versicolor[1], versicolor[3], alpha = 0.5, color = "orange")
plt.scatter(virginica[1], virginica[3], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Sepal width")
plt.ylabel("Petal width")

#PETAL LENGTH
plt.subplot(4, 4, 9)
plt.scatter(setosa[2], setosa[0], alpha = 0.5)
plt.scatter(versicolor[2], versicolor[0], alpha = 0.5, color = "orange")
plt.scatter(virginica[2], virginica[0], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal length")
plt.ylabel("Sepal length")

plt.subplot(4, 4, 10)
plt.scatter(setosa[2], setosa[1], alpha = 0.5)
plt.scatter(versicolor[2], versicolor[1], alpha = 0.5, color = "orange")
plt.scatter(virginica[2], virginica[1], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal length")
plt.ylabel("Sepal width")

plt.subplot(4, 4, 11)
plt.hist(setosa[2], bins = 10, density = True, alpha= 0.5)
plt.hist(versicolor[2], bins = 10, density = True, color = "orange", alpha= 0.5)
plt.hist(virginica[2], bins = 10, density = True, color = "green", alpha= 0.5)
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal length")

plt.subplot(4, 4, 12)
plt.scatter(setosa[2], setosa[3], alpha = 0.5)
plt.scatter(versicolor[2], versicolor[3], alpha = 0.5, color = "orange")
plt.scatter(virginica[2], virginica[3], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal length")
plt.ylabel("Petal width")

#PETAL WIDTH
plt.subplot(4, 4, 13)
plt.scatter(setosa[3], setosa[0], alpha = 0.5)
plt.scatter(versicolor[3], versicolor[0], alpha = 0.5, color = "orange")
plt.scatter(virginica[3], virginica[0], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal width")
plt.ylabel("Sepal length")

plt.subplot(4, 4, 14)
plt.scatter(setosa[3], setosa[1], alpha = 0.5)
plt.scatter(versicolor[3], versicolor[1], alpha = 0.5, color = "orange")
plt.scatter(virginica[3], virginica[1], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal width")
plt.ylabel("Sepal width")

plt.subplot(4, 4, 15)
plt.scatter(setosa[3], setosa[2], alpha = 0.5)
plt.scatter(versicolor[3], versicolor[2], alpha = 0.5, color = "orange")
plt.scatter(virginica[3], virginica[2], alpha = 0.5, color = "green")
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal width")
plt.ylabel("Petal length")

plt.subplot(4, 4, 16)
plt.hist(setosa[3], bins = 10, density = True, alpha= 0.5)
plt.hist(versicolor[3], bins = 10, density = True, color = "orange", alpha= 0.5)
plt.hist(virginica[3], bins = 10, density = True, color = "green", alpha= 0.5)
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel("Petal width")


plt.show()

#---Closing file---#
input.close() 