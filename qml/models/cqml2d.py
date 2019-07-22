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
from qml.kernels import laplacian_kernel,gaussian_kernel
from qml.math import cho_solve
import scipy.spatial.distance as ssd
import pickle

from qml.representations import get_slatm_mbtypes

# struct / class containting parameters for the multilevel data
class multilevel_data_config:
    # list of learning data files per level
    data_files = None
    # list of directories with xyz files per level
    xyz_directories = None
    # number of compounds to load for training and testing
    N_load = 0
    # representation type ("slatm" / "coulomb_matrix")
    representation_type = "slatm"
    # number of levels which shall actually be loaded
    level_count_load = 0

# struct / class containing the multilevel data
class multilevel_data:
    # list of representation arrays for each level
    X = None
    # list of learning data arrays for each level
    Y = None
    # list of cost arrays for each level
    c = None
    # level-wise cost of the data
    level_costs = None
    # function to store data
    def save(self, file_name):
        pickle.dump(self, file(file_name, 'wb'))
    # function to load data
    @staticmethod
    def load(file_name):
        return pickle.load(file(file_name, 'rb'))


# function to load the energies / the information to be learned
def get_multilevel_energies(ml_data_cfg):
    
    # list which will contain the learning data per level
    energies = [];

    # go over all levels that shall be loaded
    for l in range(ml_data_cfg.level_count_load):
        # take the file name of the data file
        filename = ml_data_cfg.data_files[l]

        # read in all lines of the data file
        f = open(filename, "r")
        lines = f.readlines()
        f.close()

        # this list will contain the learning data for the current level
        current_energies = []

        i = 0

        # read in learning data from read file data
        for line in lines[:ml_data_cfg.N_load]:
            current_energies.append(float(line))
            i = i + 1

        # attach the learning data from the current level to the per-level list
        energies.append(np.array(current_energies));

    # return multilevel learning data
    return energies


# function to load the energies / the information to be learned
def get_multilevel_costs(ml_data_cfg):
    
    # list which will contain the learning data per level
    costs = [];

    # go over all levels that shall be loaded
    for l in range(ml_data_cfg.level_count_load):
        # take the file name of the data file
        filename = ml_data_cfg.cost_files[l]

        # read in all lines of the data file
        f = open(filename, "r")
        lines = f.readlines()
        f.close()

        # this list will contain the cost data for the current level
        current_costs = []

        # read in learning data from read file data
        for line in lines[:ml_data_cfg.N_load]:
            current_costs.append(float(line))

        # attach the learning data from the current level to the per-level list
        costs.append(np.array(current_costs));

    # return multilevel learning data
    return costs


# load multilevel data, i.e. representations and learning data
def load_multilevel_data(ml_data_cfg):

    # retrieve instance of multilevel_data structure for configuration
    ml_data = multilevel_data()

    # get all levels of energies / information to be learned
    ml_data.Y = get_multilevel_energies(ml_data_cfg)

    # list, which will contain the representation lists per level
    ml_data.X = []

    # go over all levels
    for l in range(ml_data_cfg.level_count_load):

        # information
        print("Loading data for level %d ..." % (l))

        # generate a list of qml.Compound() objects
        mols = []
    
        # loop over all samples / molecules, which shall be loaded
        for i in range(ml_data_cfg.N_load):
   
            # get file name of current xyz file, which is to be used
            filename = ("%s/frag_%04d.xyz" % (ml_data_cfg.xyz_directories[l],i+1))

            # initialize the qml.Compound() object, loading the xyz date file
            mol = qml.Compound(xyz=filename);
        
            # attach the current molecule to the list of all molecules / samles
            mols.append(mol)

        # in case of the slatm representation, precompute the necessary data
        if ml_data_cfg.representation_type == "slatm":
            mbtypes = get_slatm_mbtypes(np.array([mol.nuclear_charges for mol in mols]))

        # loop over all samples / molecules
        for mol in mols:
            # generate coulomb matrix, if this type was chosen
            if ml_data_cfg.representation_type == "coulomb_matrix":
                # This is a Molecular Coulomb matrix sorted by row norm
                mol.generate_coulomb_matrix(size=23, sorting="row-norm")
            # generate slatm representation, if this type was chosen
            elif ml_data_cfg.representation_type == "slatm":
                mol.generate_slatm(mbtypes)
            else:
                print("Error: Representation type not supported")
    
        # add all representations to the representation list for the current level
        ml_data.X.append(np.array([mol.representation for mol in mols]))
    
    # return loaded multilevel data
    return ml_data


# struct / class containing all configuration parameters for the combination technique
class cqml2d_config:
    # list of regularization parameters for each global level (=> for all levels as in ml_data)
    regularizations = None
    # list of regularization parameters for each global level
    scalings = None
    # number of local (!) levels (=> number of levels actually used in the CT-lerning, not(!) len(X))
    level_count = 0
    # (global!) base level used in the CT-leraning (starting from 0, numbering as in ml_data)
    base_level = 0
    # (global!) error level used in the CT test phase (starting from 0, numbering as in ml_data)
    error_level = 0
    # resolution level stating that the maximum sample count for which the convergence test is done is 2^max_resolution_level
    max_resolution_level = 12
    # array containing the jump sizes per local (!) level (i.e. starting from base_level) 
    level_jump_size = None


# this function generates the kernel matrix for a given representations list (for one level)
def generate_kernel_matrix(X, llambda, scaling):

    # setting hyper parameter
    sigma = scaling;
    # generate kernel matrix
    K = laplacian_kernel(X, X, sigma)
    # adding regularization
    K[np.diag_indices_from(K)] += llambda;

    return K
       
# this function generates the kernel evaluation matrix between two levels of representations
def generate_evaluation_matrix(X1, X2, llambda, scaling):

    # setting hyper parameter
    sigma = scaling
    # generate kernel matrix
    K = laplacian_kernel(X1, X2, sigma)
    # adding regularization
    K[np.diag_indices_from(K)] += llambda;

    return K


# this function learns the kernel coefficients giving the appropriate data limiting it to a subset of the data (a subset of the learning info is given)
def learn_from_subset_using_data(K, Y_subset, subset_indices):

    # get subset of kernel matrix
    Ksmall = K[np.ix_(subset_indices, subset_indices)];
    
    # do the learning
    alpha = cho_solve(Ksmall, Y_subset);
    
    # return computed coefficients
    return alpha;


# this function does the kernel evaluation given a known set of subset indices for the training and test part)
def predict_from_subset(K, coeffs, learning_subset_indices, to_be_predicted_subset):
    
    # get subset of kernel matrix
    Ke = K[np.ix_(to_be_predicted_subset, learning_subset_indices)];
  
    # predict and return the prediction
    return  np.dot(Ke,coeffs);


# this function evaluates the combination technique - based QML model and gives back the error
def compute_cqml2d_error_using_subsets_fixed_error_level( ml_dat_cfg, ml_data, ct_cfg, K, N_subsets):

    # ATTENTION: In this function, data structures will be built such that they are relative to ct_cfg.base_level !!!

    ################################
    # Combination Technique Learning
    ################################

    # list containing the subset indices used for learning for each level
    learning_subset_indices = []
    
    # list containing the remaining subset indices for each level
    remaining_subset_indices = []
  
    # building subset indices lists; idea: shuffle 1:N randomly and then split up according to N_subsets
    for l in range(ct_cfg.level_count):
        indices = np.array(range(len(ml_data.Y[ct_cfg.base_level+l])))
        np.random.shuffle(indices)
        learning_subset_indices.append(indices[:N_subsets[l]]);
        remaining_subset_indices.append(indices[N_subsets[l]:]);
   
    # list of the learning coefficients for each level
    coeffs = []

    # list of predictions for each level
    predicted_results=[]

    # list of hierarchical surpluses for each level
    hierarchical_surplus = []

    # go over all levels (again: these are the levels relative to ct_cfg.base_level, not the levels as in ml_data !)
    for l in range(ct_cfg.level_count):
        # use combined models from all previous levels to predict energies at training points on new level
        predicted_results.append(np.zeros((len(learning_subset_indices[l]))))
        for ll in range(l):
            predicted_results[l] = predicted_results[l] + predict_from_subset(K[ct_cfg.base_level+l][ct_cfg.base_level+ll], coeffs[ll], learning_subset_indices[ll], learning_subset_indices[l])
        
        # compute offset / hierarchical surplus between the data on this level and the prediction for this level => this is the DELTA to learn
        hierarchical_surplus.append(ml_data.Y[ct_cfg.base_level+l][learning_subset_indices[l]].T-predicted_results[l]);
        
        # learn hierarchical surplus
        coeffs.append(learn_from_subset_using_data(K[ct_cfg.base_level+l][ct_cfg.base_level+l], hierarchical_surplus[l], learning_subset_indices[l]))



    ##################################
    # Combination Technique Evaluation
    ##################################

    # list of the result contributions for each level when doing the error evaluation
    evaluation_result_contributions = [];
    
    # evaluate machine learning model on points that have been used for
    # none (!) of the levels (this is why I use "remaining_subset_indices[0]" with a "0"
    # attention: ct_cfg.error_level is the absolute level wrt. ml_data, not the relative level as uses in all other data structures.
    #            This becomes necessary since it shall be possible to decouble the learning level from the evaluation level.
    for l in range(ct_cfg.level_count):
        evaluation_result_contributions.append(predict_from_subset(K[ct_cfg.error_level][ct_cfg.base_level+l], coeffs[l], learning_subset_indices[l], remaining_subset_indices[0]));
   
    # add up hierarchical contributions to get evaluation of results on training points
    results = evaluation_result_contributions[0];
    for l in range(1,ct_cfg.level_count):
        results = results + evaluation_result_contributions[l];

    # measure error wrt. model on ct_cfg.error_level (again: this is the absolute level!)
    error = np.mean(np.abs(results - ml_data.Y[ct_cfg.error_level][remaining_subset_indices[0]]));
    
    return error
    
    
# this function computes for the given parameters the combination technique learning and testing and performs a convergence study (with averaging over the errors)
def compute_cqml2d_convergence(ml_data_cfg, ml_data, ct_cfg, trials):
    
    # initialize array which will hold the averaged errors
    averaged_errors = np.array(0)

    # loop over all trials to compute the error
    for t in range(trials):

        # array containting the errors
        errors = []
    
        # array containing the size of the subsets per level
        N_subsets = []
    
        # array of arrays / matrix containing kernel matrices between all representation levels
        K=[];
    
        # filling kernel matrices matrix
        for l1 in range(ml_data_cfg.level_count_load):
            K_current_l1 = [];
            print("Filling matrix of kernel matrices ...")
            for l2 in range(ml_data_cfg.level_count_load):
                print("(%d, %d)" % (l1,l2))
                K_current_l1.append(generate_evaluation_matrix(ml_data.X[l1], ml_data.X[l2], ct_cfg.regularizations[l2], ct_cfg.scalings[l2]))
            K.append(K_current_l1)
        
        # array holding for each convergence point the number of subsets per level 
        nums = [0]*(ct_cfg.max_resolution_level+1);
    
    
        if len(ct_cfg.level_jump_size)==1:
            for i in range(((ct_cfg.level_count-1)*ct_cfg.level_jump_size[0]),ct_cfg.max_resolution_level+1):
                
                # computing size of the subsets for the current convergence point
                N_subsets = []
                for l in range(ct_cfg.level_count):
                    N_subsets.append(2**(i-ct_cfg.level_jump_size[0]*(l)))
               
                print("Computing error for coarse problem size of %d..." % N_subsets[0])

                # computing combination technique getting back the error for the current convergence point
                error = compute_cqml2d_error_using_subsets_fixed_error_level( ml_data_cfg, ml_data, ct_cfg, K, N_subsets);
    
                nums[i] = N_subsets
                errors.append(error);
    
        else: 
            for i in range((ct_cfg.level_jump_size(ct_cfg.level_count)),ct_cfg.max_resolution_level+1):
                
                # computing size of the subsets for the current convergence point
                N_subsets = []
                for l in range(level_count):
                    N_subsets.append(2**(i-level_jump_size(l)));
    
                # computing combination technique getting back the error for the current convergence point
                error = compute_cqml2d_error_using_subsets_fixed_error_level( ml_data_cfg, ml_data, ct_cfg, K, N_subsets);
    
                nums[i] = N_subsets;
                errors.append(error);
    
        # array holding the costs for each convergence point
        costs = []
    
        if (len(ct_cfg.level_jump_size)==1):
            for i in range(((ct_cfg.level_count-1)*ct_cfg.level_jump_size[0]),ct_cfg.max_resolution_level+1):
                current_cost = 0;
                for l in range(ct_cfg.level_count):
                    current_cost = current_cost + nums[i][l]*ml_data.level_costs[ct_cfg.base_level+l];
                costs.append(current_cost);
        else:
            for i in range((ct_cfg.level_jump_size(ct_cfg.level_count)),ct_cfg.max_resolution_level+1):
                current_cost = 0;
                for l in range(ct_cfg.level_count):
                    current_cost = current_cost + nums[i][l]*ml_data.level_costs[ct_cfg.base_level+l];
                costs.apend(current_cost);
            

        # add errors to total sum of errors (which will be used for averaging)
        averaged_errors = averaged_errors + np.array(errors);

    # average the errors
    averaged_errors = averaged_errors / trials;

    return (averaged_errors,costs)

