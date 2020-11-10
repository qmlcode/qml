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

from __future__ import print_function

from math import *
import os
import numpy as np
import sys
# import matplotlib.pyplot as plt


import qml
import qml.models.cqml as cq


trials = 20

def write_data(filename, costs, errors):
    outfile = open(filename,'w')
    outfile.write("N error\n")
    for i in range(len(errors)):
        outfile.write("%d %e\n" % (costs[i],errors[i]))
    outfile.close()

# this test compares cqml with different numbers of level (HF + MP2 + CCSD(T) vs. MP2 + CCSD(T) vs. CCSD(T)) for the QM7b data set
# in two dimensions (i.e. fixed basis set size) and "s=1", cf. Figure 5 (left) in the paper
# "Boosting quantum machine learning models with multi-level cqml_objectnation technique: Pople diagrams revisited"
def test_cqml_in_2d_jump_factor_2(cqml_object, scalings, regularizations):

    ###############################
    # testing combination technique
    ###############################


    # set level jump size
    level_jump_size = 2

    # set evaluation subspace
    evaluation_subspace = (2,2)

    # set number of evaluations to do for a single trial
    evaluation_count = 200



    # create a figure
#    f = plt.figure()

    # HF + MP2 + CCSD(T)
    
    # define maximum learning level (for smallest sampling level)
    maximum_level = 11
    # define training offset
    training_offset = (0,0)
    # generate training subspaces
    cqml_object.generate_uniform_ct_training_subspaces(2,scalings,regularizations,training_offset)
    # combpute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)

    # plot the result
#    plt.loglog([sample_counts_list[l][(2,)] for l in range(len(average_error_list))],average_error_list, label="HF+MP2+CCSD(T)")
    costs = [sample_counts_list[l][(2,)] for l in range(len(average_error_list))]
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_2d_jump1_HF_MP2_CCSDT.csv',costs,average_error_list)

    # MP2 + CCSD(T)

    # define maximum learning level (for smallest sampling level)
    maximum_level = 12
    # define training offset
    training_offset = (1,0)
    # generate training subspaces
    cqml_object.generate_uniform_ct_training_subspaces(2,scalings,regularizations,training_offset)
    # combpute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)

    # plot the result
#    plt.loglog([sample_counts_list[l][(2,)] for l in range(len(average_error_list))],average_error_list, label="MP2+CCSD(T)")
    costs = [sample_counts_list[l][(2,)] for l in range(len(average_error_list))]
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_2d_jump1_MP2_CCSDT.csv',costs,average_error_list)


    # CCSD(T)

    # define maximum learning level (for smallest sampling level)
    maximum_level = 13
    # define training offset
    training_offset = (2,0)
    # generate training subspaces
    cqml_object.generate_uniform_ct_training_subspaces(2,scalings,regularizations,training_offset)
    # combpute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)

    # plot the result
#    plt.loglog([sample_counts_list[l][(2,)] for l in range(len(average_error_list))],average_error_list, label="CCSD(T)")
    costs = [sample_counts_list[l][(2,)] for l in range(len(average_error_list))]
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_2d_jump1_CCSDT.csv',costs,average_error_list)

    
    # generate a legend for the plot
#    plt.legend()
    # add axis labels and a title
#    plt.xlabel("number of most expensive molecules")
#    plt.ylabel("MAE [kcal/mol] wrt. CCSD(T) solution")
#    plt.title("Combination technique: Plot wrt. number of most expensive molecules")

#    plt.show()


# this test compares cqml with different numbers of level (HF + MP2 + CCSD(T) vs. MP2 + CCSD(T) vs. CCSD(T)) for the QM7b data set
# in two dimensions (i.e. fixed basis set size) and "s=2", cf. Figure 5 (left) in the paper
# "Boosting quantum machine learning models with multi-level cqml_objectnation technique: Pople diagrams revisited"
def test_cqml_in_2d_jump_factor_4(cqml_object, scalings, regularizations):

    ###############################
    # testing combination technique
    ###############################


    # set level jump size
    level_jump_size = 4

    # set evaluation subspace
    evaluation_subspace = (2,2)

    # set number of evaluations to do for a single trial
    evaluation_count = 200



    # create a figure
#    f = plt.figure()

    # HF + MP2 + CCSD(T)
    
    # define maximum learning level (for smallest sampling level)
    maximum_level = 9
    # define training offset
    training_offset = (0,0)
    # generate training subspaces
    cqml_object.generate_uniform_ct_training_subspaces(2,scalings,regularizations,training_offset)
    # combpute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)

    # plot the result
#    plt.loglog([sample_counts_list[l][(2,)] for l in range(len(average_error_list))],average_error_list, label="HF+MP2+CCSD(T)")
    costs = [sample_counts_list[l][(2,)] for l in range(len(average_error_list))]
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_2d_jump2_HF_MP2_CCSDT.csv',costs,average_error_list)

    # MP2 + CCSD(T)

    # define maximum learning level (for smallest sampling level)
    maximum_level = 11
    # define training offset
    training_offset = (1,0)
    # generate training subspaces
    cqml_object.generate_uniform_ct_training_subspaces(2,scalings,regularizations,training_offset)
    # combpute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)

    # plot the result
#    plt.loglog([sample_counts_list[l][(2,)] for l in range(len(average_error_list))],average_error_list, label="MP2+CCSD(T)")
    costs = [sample_counts_list[l][(2,)] for l in range(len(average_error_list))]
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_2d_jump2_MP2_CCSDT.csv',costs,average_error_list)


    # CCSD(T)

    # define maximum learning level (for smallest sampling level)
    maximum_level = 13
    # define training offset
    training_offset = (2,0)
    # generate training subspaces
    cqml_object.generate_uniform_ct_training_subspaces(2,scalings,regularizations,training_offset)
    # combpute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)

    # plot the result
#    plt.loglog([sample_counts_list[l][(2,)] for l in range(len(average_error_list))],average_error_list, label="CCSD(T)")
    costs = [sample_counts_list[l][(2,)] for l in range(len(average_error_list))]
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_2d_jump2_CCSDT.csv',costs,average_error_list)

    
    # generate a legend for the plot
#    plt.legend()
    # add axis labels and a title
#    plt.xlabel("number of most expensive molecules")
#    plt.ylabel("MAE [kcal/mol] wrt. CCSD(T) solution")
#    plt.title("Combination technique: Plot wrt. number of most expensive molecules")

#    plt.show()



def main():

    # select whether to use already precomputed data
    use_precomputed_data = False

    # initialize combination technique object
    cqml_object = cq.cqml()
   
   
    # define energy subspaces to learn from
    cqml_object.subspaces = [(0,),(1,),(2,)]
        
    if (not use_precomputed_data):
        # set training data files
        cqml_object.data_files[(0,)] = "DataSet_cqml_QM7b/HF_ccpvdz.txt"
        cqml_object.data_files[(1,)] = "DataSet_cqml_QM7b/MP2_ccpvdz.txt"
        cqml_object.data_files[(2,)] = "DataSet_cqml_QM7b/CCSDT_ccpvdz.txt"

        # set molecule data directories
        # WARNING: "cqml" (not "cqml2d") currently only supports one single geometry for all subspaces !
        cqml_object.xyz_directories[(0,)] = "DataSet_cqml_QM7b/geometry"
        cqml_object.xyz_directories[(1,)] = "DataSet_cqml_QM7b/geometry"
        cqml_object.xyz_directories[(2,)] = "DataSet_cqml_QM7b/geometry"

        # set number of data (to be loaded) per energy subspace
        cqml_object.N_load[(0,)] = 7211
        cqml_object.N_load[(1,)] = 7211
        cqml_object.N_load[(2,)] = 7211
    
        # fix costs per energy subspace
        cqml_object.subspace_costs[(0,)] = 1
        cqml_object.subspace_costs[(1,)] = 1
        cqml_object.subspace_costs[(2,)] = 1
    
        # load the data
        cqml_object.load_data()


    # define scalings for kernel ridge regression per energy subpspace
    scalings = dict();
    scalings[(0,)] = 400
    scalings[(1,)] = 400
    scalings[(2,)] = 400
    # define regularizations per energy subspace
    regularizations = dict()
    regularizations[(0,)] = 10**(-10)
    regularizations[(1,)] = 10**(-10)
    regularizations[(2,)] = 10**(-10)

    if (not use_precomputed_data):
        # construct all training kernel matrices
        cqml_object.generate_full_kernel_matrices(scalings, regularizations)
    
        # save CT data
        cqml_object.save_binary("DataSet_cqml_QM7b/cqml_QM7b_data_2d.dat")

    if (use_precomputed_data):
        # load precomputed CT object
        print("Loading data ...")
        cqml_object = cq.cqml.load("DataSet_cqml_QM7b/cqml_QM7b_data_2d.dat")
        print("Done!")

    # set number of dimensions of CT. Here (theory_level,sampling_level) -> 2 
    cqml_object.ct_dim=2
 
    test_cqml_in_2d_jump_factor_2(cqml_object, scalings, regularizations)
    test_cqml_in_2d_jump_factor_4(cqml_object, scalings, regularizations)

if __name__ == '__main__':
    main()
