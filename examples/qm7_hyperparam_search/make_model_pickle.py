import pickle
import glob
import numpy as np
from aglaia.wrappers import OSPMRMP

estimator = OSPMRMP(batch_size = 100, representation = "sorted_coulomb_matrix")
filenames = glob.glob("qm7/*.xyz")[:1000]
energies = np.loadtxt('qm7/hof_qm7.txt', usecols=[1])[:1000]
estimator.generate_compounds(filenames)
estimator.set_properties(energies)
#real_estimator = OSPMRMP(batch_size = 100, representation = "sorted_coulomb_matrix", compounds = estimator.compounds,
#        properties = energies)
#estimator.set_properties(energies)
#print(estimator.properties.size, estimator.compounds.size)

pickle.dump(estimator, open('model.pickle', 'wb'))
with open('idx.csv', 'w') as f:
    for i in range(energies.size):
        f.write('%s\n' % i)

    
#np.save('idx.npy', np.arange(0,energies.size)[:,None])
