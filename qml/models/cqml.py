# MIT License
#
# Copyright (c) 2018 Peter Zaspel, Helmut Harbrecht
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from math import *
import os
import numpy as np
import qml
from scipy import special
from qml.kernels import laplacian_kernel,gaussian_kernel
from qml.math import cho_solve
import scipy.spatial.distance as ssd
import pickle

from qml.representations import get_slatm_mbtypes



# struct / class containing all configuration parameters for the combination technique
class cqml:

    # space dimensionality
    space_dims = [0,0]
    # list of subspaces
    subspaces = [];
    # dictionary of learning data files per subspace
    data_files = dict()
    # dictionary of directories with xyz files per subspace
    # WARNING: cqml currently only supports one single type of geometry for all subspaces !
    xyz_directories = dict()
    # dictionary of number of compounds to load for training and testing per subspace
    N_load = dict()
    # representation type ("slatm" / "coulomb_matrix")
    representation_type = "slatm"

    # dictionary of representations arrays for each subspace
    # WARNING: cqml currently only supports one single type of geometry for all subspaces !
    X = dict()
    # dictionary of energy arrays for each subspace
    Y = dict()
    # dictionary of cost arrays for each level
    c = dict()
    # dictionary of full kernel matrices (for all input data) for each subspace
    A_full = dict()
    # dictionary of subspace-wise cost of data
    subspace_costs = dict()
   
    X_eval = []
    
    # ct
    ct_dim = 3;
    # dictionary of regularization parameters for each subspace (l1,l2)
    regularizations = dict()
    # dictionary of scaling parameters for each subspace (l1,l2)
    scalings = dict()
    # (l1,l2,l3)
    subspace_coefficients = dict()
    # (l1,l2,l3)
    alphas = dict() 
    # dictionary of (l1,l2,l3)s associating for each subspace s=(l1,l2) and a given sampling level l3 a list of indices of representations used on that sampling level
    samplings = dict()
    # training subspaces with all multi-indices (l1,l2,l3) for current training task
    training_subspaces = dict()

#    evaluation_subspace = None
    evaluation_sampling = None
#    evaluation_sampling_size = 0

    # function to store precomputed class data to file
    def save(self, file_name):
        f = open(file_name, "wb")
        pickle.dump(self.space_dims, f)
        pickle.dump(self.subspaces, f)
        pickle.dump(self.data_files, f)
        pickle.dump(self.xyz_directories, f)
        pickle.dump(self.N_load, f)
        pickle.dump(self.representation_type, f)
        pickle.dump(self.X, f)
        pickle.dump(self.Y, f)
        pickle.dump(self.c, f)
        pickle.dump(self.A_full, f)
        pickle.dump(self.subspace_costs, f)
        pickle.dump(self.ct_dim, f)
        pickle.dump(self.regularizations, f)
        pickle.dump(self.scalings, f)
        pickle.dump(self.subspace_coefficients, f)
        pickle.dump(self.alphas, f)
        pickle.dump(self.samplings, f)
        pickle.dump(self.training_subspaces, f)
        pickle.dump(self.X_eval, f)
        pickle.dump(self.evaluation_sampling, f)
        f.close()

    # function to store precomputed class data to binary file
    def save_binary(self, file_name):
        f = open(file_name, "wb")
        pickle.dump(self.space_dims, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.subspaces, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.data_files, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.xyz_directories, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.N_load, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.representation_type, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.X, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.Y, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.c, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.A_full, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.subspace_costs, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.ct_dim, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.regularizations, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.scalings, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.subspace_coefficients, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.alphas, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.samplings, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.training_subspaces, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.X_eval, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.evaluation_sampling, f, pickle.HIGHEST_PROTOCOL)
        f.close()


    
    # function to load precomputed class data
    @staticmethod
    def load(file_name):
        combi = cqml()
        f = open(file_name, "rb")
        combi.space_dims = pickle.load(f);
        combi.subspaces = pickle.load(f);
        combi.data_files = pickle.load(f);
        combi.xyz_directories = pickle.load(f);
        combi.N_load = pickle.load(f);
        combi.representation_type = pickle.load(f);
        combi.X = pickle.load(f);
        combi.Y = pickle.load(f);
        combi.c = pickle.load(f);
        combi.A_full = pickle.load(f);
        combi.subspace_costs = pickle.load(f);
        combi.ct_dim = pickle.load(f);
        combi.regularizations = pickle.load(f);
        combi.scalings = pickle.load(f);
        combi.subspace_coefficients = pickle.load(f);
        combi.alphas = pickle.load(f);
        combi.samplings = pickle.load(f);
        combi.training_subspaces = pickle.load(f);
        combi.X_eval = pickle.load(f);
        combi.evaluation_sampling = pickle.load(f);
        f.close()
        return combi


    # function to load the energies / the information to be learned
    def load_energies(self):
    
        # go over all subspaces that shall be loaded
        for s in self.subspaces:
            # take the file name of the data file
            filename = self.data_files[s]
    
            # read in all lines of the data file
            f = open(filename, "r")
            lines = f.readlines()
            f.close()

            # this list will contain the learning data for the current subspace
            current_energies = []
    
            # read in learning data from read file data
            for line in lines[:self.N_load[s]]:
                current_energies.append(float(line))
    
            # set the learning data for the current subspace
            self.Y[s]=np.array(current_energies)
    

    # load data for all subspaces
    # i.e. load training data and xyz files and generate
    #      the selected representations
    #      for now, only coulomb matrices and slatm are supported
    def load_data(self):

        # get all energies / information to be learned
        self.load_energies()
    
        # go over all subspaces
        for s in self.subspaces:

            # information
            print("Loading data for subspace ",s)
    
            # generate a list of qml.Compound() objects
            mols = []
        
            # loop over all samples / molecules, which shall be loaded
            for i in range(self.N_load[s]):
    
                # get file name of current xyz file, which is to be used
                filename = ("%s/frag_%04d.xyz" % (self.xyz_directories[s],i+1))

                # initialize the qml.Compound() object, loading the xyz date file
                mol = qml.Compound(xyz=filename);
        
                # attach the current molecule to the list of all molecules / samples
                mols.append(mol)

            # in case of the slatm representation, precompute the necessary data
            if self.representation_type == "slatm":
                mbtypes = get_slatm_mbtypes(np.array([mol.nuclear_charges for mol in mols]))

            # loop over all samples / molecules
            for mol in mols:
                # generate coulomb matrix, if this type was chosen
                if self.representation_type == "coulomb_matrix":
                    # This is a Molecular Coulomb matrix sorted by row norm
                    mol.generate_coulomb_matrix(size=23, sorting="row-norm")
                # generate slatm representation, if this type was chosen
                elif self.representation_type == "slatm":
                    mol.generate_slatm(mbtypes)
                else:
                    print("Error: Representation type not supported")
        
            # add all representations to the representation list for the current level
            self.X[s]=np.array([mol.representation for mol in mols])


    # generate kernel matrices using laplacian kernel
    def generate_full_kernel_matrices(self, scalings, regularizations):
                
        for s in self.subspaces:
            # setting hyper parameter
            sigma = scalings[s];
            # generate kernel matrix
            current_A_full = laplacian_kernel(self.X[s], self.X[s], sigma)
            # adding regularization
            current_A_full[np.diag_indices_from(current_A_full)] += regularizations[s];
            # store to global array
            self.A_full[s] = current_A_full

    
    # do training on the previously defined training_subspaces
    def train(self):
        alphas = dict()

        # loop over all training subspaces
        for t in self.training_subspaces:
            # get theory subspace multiindices
            s = t[:self.ct_dim-1]
            # get subset of kernel matrix
            Asmall = self.A_full[s][np.ix_(self.samplings[t],self.samplings[t])];

            # do the learning
            self.alphas[t] = cho_solve(Asmall, self.Y[s][self.samplings[t]]);


    # evaluate trained ML model in the given evaluation subspace
    def evaluate(self):

        # set result vector to zeros
        result_size = len(self.evaluation_sampling)
        result_pos = np.zeros(result_size);
        result_neg = np.zeros(result_size);
        result = np.zeros(result_size);

        # loop over all training subspaces with positive coefficients
        for t in self.training_subspaces:
            if self.subspace_coefficients[t]>0:
                # get training subspace index without sampling level 
                s_train = t[:self.ct_dim-1]
                # get evaluation matrix for current subspace
                X_eval = self.X_eval[self.evaluation_sampling]
                X_train = self.X[s_train][self.samplings[t]]
                A_eval = self.A_full[s_train][np.ix_(self.evaluation_sampling,self.samplings[t])];
                # add up subspace contribution
                result_pos = result_pos + self.subspace_coefficients[t] * A_eval.dot(self.alphas[t])

        # loop over all training subspaces with negative coefficients
        for t in self.training_subspaces:
            if self.subspace_coefficients[t]<0:
                # get training subspace index without sampling level 
                s_train = t[:self.ct_dim-1]
                # get evaluation matrix for current subspace
                X_eval = self.X_eval[self.evaluation_sampling]
                X_train = self.X[s_train][self.samplings[t]]
                A_eval = self.A_full[s_train][np.ix_(self.evaluation_sampling,self.samplings[t])];
                # add up subspace contribution
                result_pos = result_pos + self.subspace_coefficients[t] * A_eval.dot(self.alphas[t])

        # finally combine positive and negative contributionts (avoiding cancellation issues)
        result = result_pos + result_neg

        return result


    # helper function to set up training subspaces and standard parameters for standard uniform CT
    # warning: for now, only 2d and 3d are implemented
    def generate_uniform_ct_training_subspaces(self, level, scaling, regularization, offsets):
        # empty training subspace list
        self.training_subspaces = [];
        # empty subspace coefficients dictionary
        self.subspace_coefficients = dict();

        self.scalings = dict();

        self.regularizations = dict();
       
        print(level)

        L = level;
        d = self.ct_dim;

        # case of 3d CT
        if (self.ct_dim==3):
            # loop over multi-index (l1,l2,l3)
            for l1 in range(offsets[0],L+d-1 +1):
                for l2 in range(offsets[1],L+d-1 +1):
                    for l3 in range(offsets[2],L+d-1 +1):
                        # loop over q = 0 .. ct_dim-1
                        for q in range(d-1 +1):
                            # identify valid subspace
                            if (l1+l2+l3 == L+(d-1)-q):
                                # define multi-index
                                s = (l1,l2,l3)
                                print(s)
                                # add current subspace to training subspaces
                                self.training_subspaces.append(s)
                                # compute coefficient for current subspace
                                self.subspace_coefficients[s] = pow(-1,q) * special.binom(d-1,q)
                                # set default regularization
                                self.regularizations[s] = regularization[(l1,l2)]
                                # set default scaling
                                self.scalings[s] = scaling[(l1,l2)]


            # reset combination coefficients following eqs. (12),(13) in the paper
            for l in self.training_subspaces:
                coeff = 0;
                for l1 in range(2):
                    for l2 in range(2):
                        for l3 in range(2):
                            z = (l1,l2,l3)
                            norm1 = l1+l2+l3
                            l_plus_z = (l[0]+l1,l[1]+l2,l[2]+l3)
                            if l_plus_z in self.training_subspaces:
                                coeff = coeff + (-1)**norm1
                self.subspace_coefficients[l] = coeff

        # case of 2d CT
        if (self.ct_dim==2):
            # loop over multi-index (l1,l2)
            for l1 in range(offsets[0],level+1):
                for l2 in range(offsets[1],level+1):
                    # loop over q = 0 .. ct_dim-1
                    for q in range(2):
                        # identify valid subspace
                        if (l1+l2 == level-q):
                            # define multi-index
                            s = (l1,l2)
                            # add current subspace to training subspaces
                            self.training_subspaces.append(s)
                            # compute coefficient for current subspace
                            self.subspace_coefficients[s] = pow(-1,q) * special.binom(1,q)
                            # set default regularization
                            self.regularizations[s] = regularization[(l1,)]
                            # set default scaling
                            self.scalings[s] = scaling[(l1,)]

    # helper function to set up training subspaces and standard parameters for CT with "shift"
    # warning: this is currently only supported in 3d
    def generate_uniform_ct_training_subspaces_with_shift(self, level,scaling, regularization, offsets, shift):
        # empty training subspace list
        self.training_subspaces = [];
        # empty subspace coefficients dictionary
        self.subspace_coefficients = dict();

        self.scalings = dict();

        self.regularizations = dict();
       
        print(level)

        L = level;
        d = self.ct_dim;

        # case of 3d CT
        if (self.ct_dim==3):
            # loop over multi-index (l1,l2,l3)
            for l1 in range(offsets[0],L+d-1 +1):
                for l2 in range(offsets[1],L+d-1 +1):
                    for l3 in range(offsets[2]-shift,L+d-1 +1):
                        # loop over q = 0 .. ct_dim-1
                        for q in range(d-1 +1):
                            # identify valid subspace
                            if (l1+l2+l3 == L+(d-1)-q):
                                # define multi-index
                                s = (l1,l2,l3+shift)
                                print(s)
                                # add current subspace to training subspaces
                                self.training_subspaces.append(s)
                                # compute coefficient for current subspace
                                self.subspace_coefficients[s] = pow(-1,q) * special.binom(d-1,q)
                                # set default regularization
                                self.regularizations[s] = regularization[(l1,l2)]
                                # set default scaling
                                self.scalings[s] = scaling[(l1,l2)]

            # reset combination coefficients following eqs. (12),(13) in the paper
            for l in self.training_subspaces:
                coeff = 0;
                for l1 in range(2):
                    for l2 in range(2):
                        for l3 in range(2):
                            z = (l1,l2,l3)
                            norm1 = l1+l2+l3
                            l_plus_z = (l[0]+l1,l[1]+l2,l[2]+l3)
                            if l_plus_z in self.training_subspaces:
                                coeff = coeff + (-1)**norm1
                self.subspace_coefficients[l] = coeff


    # helper function to set up training subspaces and standard parameters for 3D CQML, where we fix one dimension
    # -> 2D CQML for 3D data
    # warning: this is for now only supported in 3d
    def generate_uniform_ct_training_subspaces_with_one_fixed_dimension(self, level, scaling, regularization, offsets, fixed_dimension, fixed_level):
        # empty training subspace list
        self.training_subspaces = [];
        # empty subspace coefficients dictionary
        self.subspace_coefficients = dict();

        self.scalings = dict();

        self.regularizations = dict();
       
        print(level)

        if (self.ct_dim==3):
            L = level;
            d = self.ct_dim-1;
            
            if fixed_dimension==0:
                # loop over multi-index (.,l2,l3)
                l1 = fixed_level
                for l2 in range(offsets[1],L+d-1 +1):
                    for l3 in range(offsets[2],L+d-1 +1):
                        # loop over q = 0 .. ct_dim-1
                        for q in range(d-1 +1):
                            # identify valid subspace
                            if (l2+l3 == L+(d-1)-q):
                                # define multi-index
                                s = (l1,l2,l3)
                                print(s)
                                # add current subspace to training subspaces
                                self.training_subspaces.append(s)
                                # compute coefficient for current subspace
                                self.subspace_coefficients[s] = pow(-1,q) * special.binom(d-1,q)
                                # set default regularization
                                self.regularizations[s] = regularization[(l1,l2)]
                                # set default scaling
                                self.scalings[s] = scaling[(l1,l2)]

            if fixed_dimension==1:
                # loop over multi-index (l1,.,l3)
                l2 = fixed_level
                for l1 in range(offsets[0],L+d-1 +1):
                    for l3 in range(offsets[2],L+d-1 +1):
                        # loop over q = 0 .. ct_dim-1
                        for q in range(d-1 +1):
                            # identify valid subspace
                            if (l1+l3 == L+(d-1)-q):
                                # define multi-index
                                s = (l1,l2,l3)
                                print(s)
                                # add current subspace to training subspaces
                                self.training_subspaces.append(s)
                                # compute coefficient for current subspace
                                self.subspace_coefficients[s] = pow(-1,q) * special.binom(d-1,q)
                                # set default regularization
                                self.regularizations[s] = regularization[(l1,l2)]
                                # set default scaling
                                self.scalings[s] = scaling[(l1,l2)]

            



    # helper function to generate the different levels of sampling given an already defined training_subspaces list
    def generate_samplings(self, jump_factor, base_count):
        
        # empty indices dictionary which will contain per energy subspace the global indexing of all samples for that subspace
        indices = dict()

        # go over all energy subspaces (without sampling dimension)
        for s in self.subspaces:
            # generate for each energy subspace an indexing for all samples
            indices[s] = np.arange(len(self.Y[s]))
            # shuffle the indexing
            np.random.shuffle(indices[s])

        # iterate over all training subspaces
        for t in self.training_subspaces:
            # compute number of samples for current level 
            current_sampling_size = int(base_count*(pow(jump_factor,t[self.ct_dim-1])))
            # take first "current_sampling_size" many indices as sampling indices
            self.samplings[t] = indices[t[:self.ct_dim-1]][:current_sampling_size]


    # helper function to generate the different levels of nested (!) sampling given an already defined training_subspaces list
    # ATTENTION: This function assumes that the data for all subspaces is fo the same size and data is sorted identical in all
    #            data sets !
    def generate_nested_samplings(self, jump_factor, base_count):

        
        # empty indices dictionary which will contain per energy subspace the global indexing of all samples for that subspace
        indices = dict()

        subspace_size=0
        # find size of largest subspace  <- only some size is needed (since all are assumed to be identical)
        for s in self.subspaces:
            if len(self.Y[s])>subspace_size:
                subspace_size = len(self.Y[s])
        
        # generate ONE randomly shuffled indexing
        indexing = np.arange(subspace_size)
        np.random.shuffle(indexing)

        # go over all energy subspaces (without sampling dimension)
        for s in self.subspaces:
            # generate for each energy subspace an indexing for all samples
            indices[s] = indexing

        # iterate over all training subspaces
        for t in self.training_subspaces:
            # compute number of samples for current level 
            current_sampling_size = int(base_count*(pow(jump_factor,t[self.ct_dim-1])))
            # take first "current_sampling_size" many indices as sampling indices
            self.samplings[t] = indices[t[:self.ct_dim-1]][:current_sampling_size]


    # helper function to randomly select out molecules that were not part of the training set
    def generate_out_of_sample_evaluation_from_training_data(self,evaluation_subspace, evaluation_sampling_size):

        # iterate over all training subspaces to find union over all molecules used for training
        used_molecules = set();
        for t in self.training_subspaces:
            for i in self.samplings[t]:
                used_molecules.add(i)

        # get evaluation subspace index without sampling level
        s_eval = evaluation_subspace[:self.ct_dim-1];

        # copy evaluation subspace data
        self.X_eval = self.X[s_eval]

        # find molecules in evaluation subspace that were not used before
        self.evaluation_sampling = range(len(self.X_eval))
        self.evaluation_sampling = list(set(self.evaluation_sampling)-used_molecules)

        # generate (randomized) subset of out of sample index list
        np.random.shuffle(self.evaluation_sampling)        
        self.evaluation_sampling = self.evaluation_sampling[:evaluation_sampling_size]


    # helper function to compute the number of samples that are used for training from a given energy subspace
    def get_sample_counts(self):
        
        # dictionary for the sample counts
        sample_counts = dict()

        # initialize dictionary
        for t in self.training_subspaces:
            s = t[:self.ct_dim-1]
            sample_counts[s] = 0
        
        # add up samples from the corresponding subspaces
        for t in self.training_subspaces:
            s = t[:self.ct_dim-1]
            sample_counts[s] = sample_counts[s] + len(self.samplings[t])

        # output
        return sample_counts
 
 
    # helper function to compute the number of samples that are use for training from a given energy subspace in case of nested subspaces
    def get_sample_counts_for_nested_subspaces(self):
        
        # dictionary for the sample counts
        sample_counts = dict()

        # dictionary for the per-subspace samplings
        samplings_per_subspace = dict()

        # ititialize samplings_per_subspace
        for t in self.training_subspaces:
            s = t[:self.ct_dim-1]
            samplings_per_subspace[s] = []

        # append all samplings done on a given subspace to that subspace (including multiple nested samplings)
        for t in self.training_subspaces:
            s = t[:self.ct_dim-1]
            samplings_per_subspace[s].extend(self.samplings[t])

        # remove identical samplings on all subspaces
        for s in samplings_per_subspace:
            samplings_per_subspace[s] = list(set(samplings_per_subspace[s]))

        # store sample_counts
        for s in samplings_per_subspace:
            sample_counts[s] = len(samplings_per_subspace[s])

        # output
        return sample_counts

    # function to compute convergence results / training curves
    def check_convergence(self, max_level, level_jump_size, evaluation_subspace, evaluation_sampling_size, trial_count):
        average_error_list = []
        sample_counts_list = []

        for i in range(self.ct_dim+1,max_level):
            average_error = 0
            for t in range(trial_count):
                self.generate_samplings(level_jump_size, 2**i)
                self.generate_out_of_sample_evaluation_from_training_data(evaluation_subspace,evaluation_sampling_size)
                self.train()
                res = self.evaluate();
                data = self.Y[evaluation_subspace[:self.ct_dim-1]][self.evaluation_sampling]
                error = np.mean(np.abs(res-data))
                average_error = average_error + error
            average_error = average_error / trial_count
            average_error_list.append(average_error)
            sample_counts = self.get_sample_counts()
            sample_counts_list.append(sample_counts)
            print(average_error, sample_counts)

        return (average_error_list, sample_counts_list)


     # function to compute convergence results / training curves in case of nested molecule subspaces (with the default to compute MAE errors)
    def check_nested_convergence(self, max_level, level_jump_size, evaluation_subspace, evaluation_sampling_size, trial_count):
        average_error_list = []
        sample_counts_list = []

        for i in range(self.ct_dim-1,max_level):
            average_error = 0
            for t in range(trial_count):
                self.generate_nested_samplings(level_jump_size, 2**i)
                self.generate_out_of_sample_evaluation_from_training_data(evaluation_subspace,evaluation_sampling_size)
                self.train()
                res = self.evaluate()
                data = self.Y[evaluation_subspace[:self.ct_dim-1]][self.evaluation_sampling]
                error = np.mean(np.abs(res-data))
                average_error = average_error + error
            average_error = average_error / trial_count
            average_error_list.append(average_error)
            sample_counts = self.get_sample_counts_for_nested_subspaces()

            sample_counts_list.append(sample_counts)
            print(average_error, sample_counts)
        
        return (average_error_list, sample_counts_list)


    # function to compute convergence results / training curves in case of nested molecule subspaces (with RMSE errors)
    def check_nested_convergence_rmse(self, max_level, level_jump_size, evaluation_subspace, evaluation_sampling_size, trial_count):
        average_error_list = []
        sample_counts_list = []

        for i in range(self.ct_dim-1,max_level):
            average_error = 0
            for t in range(trial_count):
                self.generate_nested_samplings(level_jump_size, 2**i)
                self.generate_out_of_sample_evaluation_from_training_data(evaluation_subspace,evaluation_sampling_size)
                self.train()
                res = self.evaluate()
                data = self.Y[evaluation_subspace[:self.ct_dim-1]][self.evaluation_sampling]
                error = np.sqrt(((res - data) ** 2).mean())
                average_error = average_error + error
            average_error = average_error / trial_count
            average_error_list.append(average_error)
            sample_counts = self.get_sample_counts_for_nested_subspaces()

            sample_counts_list.append(sample_counts)
            print(average_error, sample_counts)
        
        return (average_error_list, sample_counts_list)



