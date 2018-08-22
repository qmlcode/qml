import glob
from qml.qmlearn.representations import *
from qml.qmlearn.kernels import *
from qml.qmlearn.models import *
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import copy

# TODO
# other representations
# other kernels
# other models
# atomic properties
# molecular properties

if __name__ == "__main__":
    train_data = Data(filenames="qm7/0*.xyz")
    train_energies = np.loadtxt('data/hof_qm7.txt', usecols=1)[:train_data.ncompounds]
    train_data.set_energies(train_energies)
    test_data = Data(filenames="qm7/1*.xyz")
    #test_energies = np.loadtxt('data/hof_qm7.txt', usecols=1)[:test_data.ncompounds]
    #test_data.set_energies(test_energies)

    model = make_pipeline(GlobalSLATM(), GaussianKernel(sigma=30), KernelRidgeRegression(l2_reg=1e-6))#, memory='/dev/shm')
    model.fit(train_data)
    y = model.predict(test_data)
    print(y.shape)
    quit()
    #rep = AtomicCoulombMatrix().generate(train_data)
    #kernel = GaussianKernel().generate(rep[:100], rep[:50], representation_type='atomic')

    ## Generate the representations
    #rep = CoulombMatrix().generate(train_data)
    #print(rep.shape)
    ## Generate a symmetric kernel
    #kernel = GaussianKernel().generate(rep)
    #print(kernel.shape)
    ## Generate an asymmetric kernel
    #kernel = GaussianKernel().generate(rep, rep[:10])
    #print(kernel.shape)
    ## Generate symmetric kernel from pipeline
    #model = make_pipeline(CoulombMatrix(size=23), GaussianKernel(sigma=30))
    #train_kernel = model.fit_transform(train_data).kernel
    #print(train_kernel.shape)
    ## Generate asymmetric kernel from pipeline after fit
    #test_kernel = model.transform(test_data).kernel
    #print(test_kernel.shape)
    ## Fit and predict KRR from kernels
    #model = KernelRidgeRegression()
    #model.fit(train_kernel, train_energies)
    #predictions = model.predict(test_kernel)
    #print(predictions.shape)

    ## Fit and predict KRR from pipeline
    #model = make_pipeline(AtomicCoulombMatrix(size=max(train_data.natoms), data=train_data), GaussianKernel(sigma=30), KernelRidgeRegression(l2_reg=1e-6), memory='/dev/shm')
    #model.fit(train_data)
    #predictions = model.predict(test_data)
    #print(predictions.shape)

    # Gridsearch CV of hyperparams
    params = {
              'atomiccoulombmatrix__sorting': ['distance', 'row-norm'],
              'gaussiankernel__sigma': [77, 100, 300],
              'kernelridgeregression__l2_reg': [1e-6,1e-4]
             }

    #grid = GridSearchCV(model, cv=3, refit=False, param_grid = params)
    #grid.fit(train_data)
    #print(grid.best_params_, grid.best_score_)

    ## Alternate procedure when Data object is passed in advance and
    ## indices are passed at fit/predict time
    #model = make_pipeline(CoulombMatrix(size=23, data=train_data), GaussianKernel(sigma=30), KernelRidgeRegression(l2_reg=1e-6))
    #model.fit(np.arange(50))
    #predictions = model.predict(np.arange(50,80))
    #print(predictions.shape)

    # Gridsearch CV of hyperparams
    grid = GridSearchCV(model, cv=3, refit=False, param_grid = params, verbose=2)

    idx = np.arange(len(train_data))
    np.random.shuffle(idx)

    grid.fit(idx[:(3*200)//2])
    print(grid.best_params_, grid.best_score_)



    #model = make_pipeline(Rep(), Test())
    #print("fit")
    #model.fit(data)
    #print("predict")
    #model.predict(data)
    #quit()
    #rep = model.fit_transform(data)
    #print(rep.shape)

