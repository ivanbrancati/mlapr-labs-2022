########################
#####-----LAB4-----#####
########################

####LIBRARIES#####
import numpy
import matplotlib.pyplot as plt
import scipy.linalg

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

#Function to compute log-density for a sample x
#--x is a sample with shape(M,)
#--mu is the dataset mean with shape (M, 1)
#--C is the dataset covariance matrix of shape (M,M)
def logpdf_GAU_ND_sample(x, mu, C):
    M = x.shape[0]
    sigma_log_mod = numpy.linalg.slogdet(C)[1]
    sigma_inv = numpy.linalg.inv(C)
    
    logpdf = -M/2 * numpy.log(2*numpy.pi) - sigma_log_mod/2 - numpy.dot((x-mu).T, numpy.dot(sigma_inv, (x-mu)))/2
    return logpdf.ravel()

#Function to compute log-density for a dataset X
#--X is a dataset with shape (M,N)
#--mu is the dataset mean with shape (M, 1)
#--C is the dataset covariance matrix of shape (M,M)
def logpdf_GAU_ND(X, mu, C):
    log_densities = [logpdf_GAU_ND_sample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return numpy.array(log_densities).ravel()

#Version with broadcasting
def logpdf_GAU_ND_2(X, mu, C):
    M = X.shape[0]
    sigma_inv = numpy.linalg.inv(C)
    sigma_log_mod = numpy.linalg.slogdet(sigma_inv)[1]
    
    
    logpdf = -M/2 * numpy.log(2*numpy.pi) + sigma_log_mod/2 - ((X-mu) * numpy.dot(sigma_inv, (X-mu))).sum(0)/2
    
    return logpdf

#Function to compute log-likelihood
def loglikelihood(X, m_ML, C_ML):
    result = 0
    for i in range(X.shape[1]):
        result += logpdf_GAU_ND_sample(X[:,i:i+1], m_ML, C_ML)
    return result

#####EXECUTION#####

#Multivariate Gaussian density
plt.figure()
XPlot = numpy.linspace(-8, 12, 1000)
m = numpy.ones((1,1)) * 1.0
C = numpy.ones((1,1)) * 2.0
plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
plt.show()
#Check if the results are correct
pdfSol = numpy.load('llGAU.npy')
pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)
print("Difference between sol and computed one: "+str(numpy.abs(pdfSol - pdfGau).max()))
#The result should be zero or very close to zero (it may not be exactly zero due to numerical errors,
#however it should be a very small number, e.g. ≈ 10−17)
#You can also check the density for the multi-dimensional case using the samples contained in Solution/XND.npy:
XND = numpy.load('XND.npy')
mu = numpy.load('muND.npy')
C = numpy.load('CND.npy')
pdfSol = numpy.load('llND.npy')
pdfGau = logpdf_GAU_ND(XND, mu, C)
print("Difference between sol and computed one(multi-dimensional case): "+str(numpy.abs(pdfSol - pdfGau).max()))
#Again, the result should be zero or close to zero

#Maximum Likelihood Estimate
mu_ml= vcol(dataset_mean_broadcasting(XND))
print("mu_ML:")
print(mu_ml)
sigma_ml = covariance_matrix(center_data(XND))
print("sigma_ML:")
print(sigma_ml)
ll = loglikelihood(XND, mu_ml, sigma_ml)
print("Log-likelihood:")
print(ll)

X1D = numpy.load('X1D.npy')
print("1-dimension")
mu_ml = vcol(dataset_mean_broadcasting(X1D))
print("mu_ML:")
print(mu_ml)
sigma_ml = covariance_matrix(center_data(X1D))
print("sigma_ML:")
print(sigma_ml)
#We can visualize how well the estimated density fits the samples plotting both the histogram of the
#samples and the density (again, m_ML and C_ML are the ML estimates):
plt.figure()
plt.hist(X1D.ravel(), bins=50, density=True)
XPlot = numpy.linspace(-8, 12, 1000)
plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), mu_ml, sigma_ml)))
plt.show()
#where m and C are the ML estimates for the dataset
print("Log-likelihood:")
ll = loglikelihood(X1D, mu_ml, sigma_ml)
print(ll)