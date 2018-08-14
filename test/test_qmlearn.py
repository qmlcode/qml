import glob
from qml.ml.representations import *
from qml.ml.kernels import *
from qml.models import *
import numpy as np
from sklearn.pipeline import make_pipeline
import copy

class Test(object):

    def fit(self, X, y=None):
        print(X)
        print("test,fit")

    def predict(self, X):
        print("test, predict")

class Rep(object):

    def fit(self, X, y=None):
        print("rep, fit")
        return self

    def transform(self, X):
        print("rep, trans")

    def fit_transform(self, X, y=None):
        print("rep, fit_trans")
        return 1, 2, 3

if __name__ == "__main__":
    train_data = Data(filenames="qm7/0*.xyz")
    train_energies = np.loadtxt('data/hof_qm7.txt', usecols=1)[:train_data.ncompounds]
    train_data.set_energies(train_energies)
    test_data = Data(filenames="qm7/1*.xyz")
    test_energies = np.loadtxt('data/hof_qm7.txt', usecols=1)[:test_data.ncompounds]
    test_data.set_energies(test_energies)

    # Generate the representations
    rep = CoulombMatrix().generate(train_data)
    print(rep.shape)
    # Generate a symmetric kernel
    kernel = GaussianKernel().generate(rep)
    print(kernel.shape)
    # Generate an asymmetric kernel
    kernel = GaussianKernel().generate(rep, rep[:10])
    print(kernel.shape)
    # Generate symmetric kernel from pipeline
    model = make_pipeline(CoulombMatrix(size=23), GaussianKernel(sigma=30))
    train_kernel = model.fit_transform(train_data).kernel
    print(train_kernel.shape)
    # Generate asymmetric kernel from pipeline after fit
    test_kernel = model.transform(test_data).kernel
    print(test_kernel.shape)
    # Fit and predict KRR from kernels
    model = KernelRidgeRegression()
    model.fit(train_kernel, train_energies)
    predictions = model.predict(test_kernel)
    print(predictions.shape)

    # Fit and predict KRR from pipeline
    model = make_pipeline(CoulombMatrix(size=23), GaussianKernel(sigma=30), KernelRidgeRegression(llambda=1e-6))
    model.fit(train_data)
    predictions = model.predict(test_data)
    print(predictions.shape)


    #model = make_pipeline(Rep(), Test())
    #print("fit")
    #model.fit(data)
    #print("predict")
    #model.predict(data)
    #quit()
    #rep = model.fit_transform(data)
    #print(rep.shape)

