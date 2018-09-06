import glob
from qml.qmlearn.representations import *
from qml.qmlearn.kernels import *
from qml.qmlearn.models import *
from qml.qmlearn.preprocessing import *
import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
import copy
import sklearn

# TODO
# other representations
# other kernels
# other models
# atomic properties
# molecular properties

if __name__ == "__main__":
    train_data = Data(filenames="qm7/*.xyz")
    train_energies = np.loadtxt('data/hof_qm7.txt', usecols=1)[:train_data.ncompounds]
    train_data.set_energies(train_energies)
    test_data = Data(filenames="qm7/11*.xyz")
    test_energies = np.loadtxt('data/hof_qm7.txt', usecols=1)[:test_data.ncompounds]
    test_data.set_energies(test_energies)

    #rep = FCHLRepresentation().generate(test_data)
    #model = FCHLKernel(local=False)
    #kernel = model.generate(rep)



    #model = make_pipeline(CoulombMatrix(data=train_data), GaussianKernel(sigma=30), KernelRidgeRegression(l2_reg=1e-6), memory='/dev/shm')
    #model.fit(train_data)
    #y = model.score(test_data)
    #print(y)
    #quit()
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

    #model = make_pipeline(AtomScaler(train_data), AtomicSLATM(), GaussianKernel(sigma='auto'), KernelRidgeRegression(l2_reg=1e-8))
    #idx = np.arange(len(train_data))
    #np.random.shuffle(idx)
    #model.fit(idx[:(3*100)//2])
    #quit()
    model = make_pipeline(AtomScaler(train_data), CoulombMatrix(), GaussianKernel(), KernelRidgeRegression())

    ## Fit and predict KRR from pipeline
    #model = make_pipeline(CoulombMatrix(train_data), GaussianKernel(), KernelRidgeRegression())
    model = Pipeline([('preprocessing', AtomScaler(train_data)), ('representation', FCHLRepresentation()), ('kernel',GaussianKernel()), ('model', KernelRidgeRegression())])
    #predictions = model.predict(test_data)
    #print(predictions.shape)


    # Gridsearch CV of hyperparams
    params = {'representation': [AtomicCoulombMatrix(train_data)],
              'kernel': [GaussianKernel()],
              'kernel__sigma': [1000],
              'model__l2_reg': [1e-8]
             }

    grid = GridSearchCV(model, cv=3, refit=False, param_grid = params)
    idx = np.arange(len(train_data))
    np.random.shuffle(idx)
    grid.fit(idx[:(3*50)//2])
    print(grid.best_params_, grid.best_score_)
    quit()

    ## Alternate procedure when Data object is passed in advance and
    ## indices are passed at fit/predict time
    #model = make_pipeline(CoulombMatrix(size=23, data=train_data), GaussianKernel(sigma=30), KernelRidgeRegression(l2_reg=1e-6))
    #model.fit(np.arange(50))
    #predictions = model.predict(np.arange(50,80))
    #print(predictions.shape)

    # Gridsearch CV of hyperparams
    grid = GridSearchCV(model, cv=3, refit=False, param_grid = params, verbose=2)

    #model = make_pipeline(FCHLRepresentation(train_data), FCHLKernel(sigma='auto', local=False), KernelRidgeRegression(l2_reg=1e-8))

    #import sklearn
    #scores = sklearn.model_selection.cross_validate(model, idx[:(3*100)//2], cv=3)['test_score']
    #print(scores)
    #quit()
    print(grid.best_params_, grid.best_score_)
    quit()



    #model = make_pipeline(Rep(), Test())
    #print("fit")
    #model.fit(data)
    #print("predict")
    #model.predict(data)
    #quit()
    #rep = model.fit_transform(data)
    #print(rep.shape)

