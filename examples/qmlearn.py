import glob
import numpy as np
from qml import qmlearn
import sklearn.pipeline
import sklearn.model_selection

def data():
    """
    Using the Data object.
    """
    print("*** Begin data examples ***")

    # The Data object has the same role as the Compound class.
    # Where the Compound class is for one compound, the Data class
    # Is for multiple

    # One can load in a set of xyz files
    filenames = sorted(glob.glob("../test/qm7/00*.xyz"))
    data = qmlearn.Data(filenames)
    print("length of filenames", len(filenames))
    print("length of nuclear_charges", len(data.nuclear_charges))
    print("length of coordinates", len(data.coordinates))

    # Or just load a glob string
    data = qmlearn.Data("../test/qm7/00*.xyz")
    print("length of nuclear_charges", len(data.nuclear_charges))

    # Energies (or other molecular properties) can be stored in the object
    energies = np.loadtxt("../test/data/hof_qm7.txt", usecols=1)[:98]
    data.set_energies(energies)
    print("length of energies", len(data.energies))

    print("*** End data examples ***")
    print()

def preprocessing():
    """
    Rescaling energies
    """

    print("*** Begin preprocessing examples ***")

    # The AtomScaler object does a linear fit of the number of each element to the energy.
    data = qmlearn.Data("../test/qm7/*.xyz")
    energies = np.loadtxt("../test/data/hof_qm7.txt", usecols=1)

    # Input can be nuclear_charges and energies
    print("Energies before rescaling", energies[:3])
    rescaled_energies = qmlearn.preprocessing.AtomScaler().fit_transform(data.nuclear_charges, energies)
    print("Energies after rescaling", rescaled_energies[:3])

    # Or a data object can be used
    data.set_energies(energies)
    data2 = qmlearn.preprocessing.AtomScaler().fit_transform(data)
    print("Energies after rescaling", data2.energies[:3])

    print("*** End preprocessing examples ***")
    print()

def representations():
    """
    Creating representations. Currently implemented representations are
    CoulombMatrix, AtomicCoulombMatrix, AtomicSLATM, GlobalSLATM,
    FCHLRepresentations, AtomCenteredSymmetryFunctions. 
    (BagOfBonds is still missing)
    """

    print("*** Begin representations examples ***")

    data = qmlearn.Data("../test/qm7/*.xyz")

    # Representations can be created from a data object
    model = qmlearn.representations.CoulombMatrix(sorting ='row-norm')
    representations = model.generate(data)
    print("Shape of representations:", representations.shape)

    # Alternatively the data object can be passed at initialization of the representation class
    # and only select molecule indices can be parsed

    model = qmlearn.representations.CoulombMatrix(data)
    representations = model.generate([0,5,7,16])
    print("Shape of representations:", representations.shape)

    print("*** End representations examples ***")
    print()

def kernels():
    """
    Create kernels. Currently implemented kernels are GaussianKernel,
    LaplacianKernel, FCHLKernel.
    """

    print("*** Begin kernels examples ***")

    data = qmlearn.Data("../test/qm7/*.xyz")
    energies = np.loadtxt("../test/data/hof_qm7.txt", usecols=1)
    data.set_energies(energies)

    # Kernels can be created from representations
    model = qmlearn.representations.CoulombMatrix(data)
    indices = np.arange(100)
    representations = model.generate(indices)

    model = qmlearn.kernels.GaussianKernel(sigma='auto')
    symmetric_kernels = model.generate(representations[:80])
    print("Shape of symmetric kernels:", symmetric_kernels.shape)

    asymmetric_kernels = model.generate(representations[:80], representations[80:])
    print("Shape of asymmetric kernels:", asymmetric_kernels.shape)

    # Atomic representations can be used as well
    model = qmlearn.representations.AtomicCoulombMatrix(data)
    indices = np.arange(100)
    representations = model.generate(indices)

    model = qmlearn.kernels.GaussianKernel(sigma='auto')
    symmetric_kernels = model.generate(representations[:80], representation_type = 'atomic')
    print("Shape of symmetric kernels:", symmetric_kernels.shape)

    asymmetric_kernels = model.generate(representations[:80], representations[80:], representation_type = 'atomic')
    print("Shape of asymmetric kernels:", asymmetric_kernels.shape)

    print("*** End kernels examples ***")
    print()

def models():
    """
    Regression models. Only KernelRidgeRegression implemented so far.
    """

    print("*** Begin models examples ***")

    filenames = sorted(glob.glob("../test/qm7/*.xyz"))
    data = qmlearn.Data(filenames)
    energies = np.loadtxt("../test/data/hof_qm7.txt", usecols=1)
    model = qmlearn.representations.CoulombMatrix(data)
    # Create 1000 random indices
    indices = np.arange(1000)
    np.random.shuffle(indices)

    representations = model.generate(indices)
    model = qmlearn.kernels.GaussianKernel(sigma='auto')
    symmetric_kernels = model.generate(representations[:800])
    asymmetric_kernels = model.generate(representations[:800], representations[800:])

    # Model can be fit giving kernel matrix and energies

    model = qmlearn.models.KernelRidgeRegression()
    model.fit(symmetric_kernels, energies[indices[:800]])
    print("Fitted KRR weights:", model.alpha[:3])

    # Predictions can be had from an asymmetric kernel
    predictions = model.predict(asymmetric_kernels)
    print("Predicted energies:", predictions[:3])
    print("True energies:", energies[indices[:3]])

    # Or the score (default negative mae) can be had directly
    scores = model.score(asymmetric_kernels, energies[indices[800:]])
    print("Negative MAE:", scores)

    print("*** End models examples ***")
    print()

def pipelines():
    """
    Constructing scikit-learn pipelines
    """

    print("*** Begin pipelines examples ***")

    # It is much easier to do all this with a scikit-learn pipeline

    # Create data
    data = qmlearn.Data("../test/qm7/*.xyz")
    energies = np.loadtxt("../test/data/hof_qm7.txt", usecols=1)
    data.set_energies(energies)

    # Create model
    model = sklearn.pipeline.make_pipeline(
            qmlearn.preprocessing.AtomScaler(data),
            qmlearn.representations.CoulombMatrix(),
            qmlearn.kernels.GaussianKernel(),
            qmlearn.models.KernelRidgeRegression(),
            )

    # Create 1000 random indices
    indices = np.arange(1000)
    np.random.shuffle(indices)

    model.fit(indices[:800])
    scores = model.score(indices[800:])
    print("Negative MAE:", scores)

    # Passing alchemy=False to kernels makes sure that the atomic kernel only compares C to C, H to H etc.
    # This will speed up kernels of some representations dramatically, but only works in pipelines

    # Create model
    model = sklearn.pipeline.make_pipeline(
            qmlearn.preprocessing.AtomScaler(data),
            qmlearn.representations.CoulombMatrix(),
            qmlearn.kernels.GaussianKernel(alchemy=False),
            qmlearn.models.KernelRidgeRegression(),
            )

    # Create 1000 random indices
    indices = np.arange(1000)
    np.random.shuffle(indices)

    model.fit(indices[:800])
    scores = model.score(indices[800:])
    print("Negative MAE without alchemy:", scores)

    print("*** End pipelines examples ***")
    print()

def cross_validation():
    """
    Doing cross validation with qmlearn
    """

    print("*** Begin CV examples ***")

    # Create data
    data = qmlearn.Data("../test/qm7/*.xyz")
    energies = np.loadtxt("../test/data/hof_qm7.txt", usecols=1)
    data.set_energies(energies)

    # Create model
    model = sklearn.pipeline.make_pipeline(
            qmlearn.preprocessing.AtomScaler(data),
            qmlearn.representations.CoulombMatrix(),
            qmlearn.kernels.GaussianKernel(),
            qmlearn.models.KernelRidgeRegression(),
            # memory='/dev/shm/' ### This will cache the previous steps to the virtual memory and might speed up gridsearch
            )

    # Create 1000 random indices
    indices = np.arange(1000)
    np.random.shuffle(indices)

    # 3-fold CV of a given model can easily be done
    scores = sklearn.model_selection.cross_validate(model, indices, cv=3)
    print("Cross-validated scores:", scores['test_score'])

    # Doing a grid search over hyper parameters
    params = {'gaussiankernel__sigma': [10, 30, 100],
              'kernelridgeregression__l2_reg': [1e-8, 1e-4],
             }

    grid = sklearn.model_selection.GridSearchCV(model, cv=3, refit=False, param_grid=params)
    grid.fit(indices)
    print("Best hyper parameters:", grid.best_params_)
    print("Best score:", grid.best_score_)

    # As an alternative the pipeline can be constructed slightly different, which allows more complex CV
    # Create model
    model = sklearn.pipeline.Pipeline([
            ('preprocess', qmlearn.preprocessing.AtomScaler(data)),
            ('representations', qmlearn.representations.CoulombMatrix()),
            ('kernel', qmlearn.kernels.GaussianKernel()),
            ('model', qmlearn.models.KernelRidgeRegression())
            ],
            # memory='/dev/shm/' ### This will cache the previous steps to the virtual memory and might speed up gridsearch
            )

    # Doing a grid search over hyper parameters
    # including which kernel to use
    params = {'kernel': [qmlearn.kernels.LaplacianKernel(), qmlearn.kernels.GaussianKernel()],
              'kernel__sigma': [10, 30, 100, 1000, 3000, 1000],
              'model__l2_reg': [1e-8, 1e-4],
             }

    grid = sklearn.model_selection.GridSearchCV(model, cv=3, refit=False, param_grid=params)
    grid.fit(indices)
    print("Best hyper parameters:", grid.best_params_)
    print("Best score:", grid.best_score_)

    print("*** End CV examples ***")

if __name__ == '__main__':
    data()
    preprocessing()
    representations()
    kernels()
    models()
    pipelines()
    cross_validation()
