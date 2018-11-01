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

# this test compares 3D cqml, 2D cqml and conventional QML for the QM7b data set "with shift" and "s=1", cf. Figure 5 in the paper
# "Boosting quantum machine learning models with multi-level cqml_objectnation technique: Pople diagrams revisited"
def test_cqml_with_shift(cqml_object, scalings, regularizations):

    ###############################
    # testing combination technique
    ###############################


    # set level jump size
    level_jump_size = 2

    # set evaluation subspace
    evaluation_subspace = (2,2,2)

    # set number of evaluations to do for a single trial
    evaluation_count = 400



    # create a figure
#    f = plt.figure(figsize=(12,9))

    # HF + MP2 + CCSD(T)  (3D CT)
 
    # define maximum learning level (for smallest sampling level)
    maximum_level = 9
    # define training offset
    training_offset = (0,0,0)
    # generate training subspaces
    cqml_object.generate_uniform_ct_training_subspaces_with_shift(0,scalings,regularizations,training_offset, 2)

    # compute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)


    # get subspaces
    subspaces = []
    for s in sample_counts_list[0]:
        subspaces.append(s)

    # construct dictionary which will associate the subspaces to a sample count list for that subspace
    sample_count_list_per_subspace = dict()
    for s in subspaces:
        sample_count_list_per_subspace[s] = []

    # fill dictionary
    for sl in sample_counts_list:
        for s in sl:
            sample_count_list_per_subspace[s].append(sl[s])

    # plot the contributions for all subspaces
#    plt.xscale("log")
#    plt.yscale("log")
#    plt.plot(sample_count_list_per_subspace[(2,2)],average_error_list, label="3d CQML: CCSD(T) + cc-pVDZ contribution")
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_3DCT_jump1_all_levels.csv',sample_count_list_per_subspace[(2,2)],average_error_list)
   
    # CCSD(T)

    # define maximum learning level (for smallest sampling level)
    maximum_level = 13
    # define training offset
    training_offset = (2,2,0)
    # generate training subspaces
    cqml_object.generate_uniform_ct_training_subspaces(2,scalings,regularizations,training_offset)
    # combpute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)

    # get subspaces
    subspaces = []
    for s in sample_counts_list[0]:
        subspaces.append(s)

    # construct dictionary which will associate the subspaces to a sample count list for that subspace
    sample_count_list_per_subspace = dict()
    for s in subspaces:
        sample_count_list_per_subspace[s] = []

    # fill dictionary
    for sl in sample_counts_list:
        for s in sl:
            sample_count_list_per_subspace[s].append(sl[s])


    # plot the result
#    plt.yscale('log')
#    plt.xscale('log')
#    plt.plot(sample_count_list_per_subspace[(2,2)],average_error_list, label="standard ML: CCSD(T) + cc-pVDZ")
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_standard_QML.csv',sample_count_list_per_subspace[(2,2)],average_error_list)
 
   
    # fixing ccpvdz basis set, variation in theory (2D CT)

    # define maximum learning level (for smallest sampling level)
    maximum_level = 11
    # define training offset
    training_offset = (0,0,0)
    # generate training subspaces
    cqml_object.generate_uniform_ct_training_subspaces_with_one_fixed_dimension(1, scalings, regularizations, training_offset, 1, 2)

    # combpute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)

    # get subspaces
    subspaces = []
    for s in sample_counts_list[0]:
        subspaces.append(s)

    # construct dictionary which will associate the subspaces to a sample count list for that subspace
    sample_count_list_per_subspace = dict()
    for s in subspaces:
        sample_count_list_per_subspace[s] = []

    # fill dictionary
    for sl in sample_counts_list:
        for s in sl:
            sample_count_list_per_subspace[s].append(sl[s])



    # plot the result
#    plt.plot(sample_count_list_per_subspace[(2,2)],average_error_list, label="2d CQML wrt. theories: CCSD(T) + cc-pVDZ contribution")
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_2DCT_jump1_all_levels.csv',sample_count_list_per_subspace[(2,2)],average_error_list)
 

    # fixing CCSD(T) theory, variation in basis set size (2D CT)

    # define maximum learning level (for smallest sampling level)
    maximum_level = 11
    # define training offset
    training_offset = (0,0,0)
    # generate training subspaces
    cqml_object.generate_uniform_ct_training_subspaces_with_one_fixed_dimension(1, scalings, regularizations, training_offset, 0, 2)

    # combpute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)

    # get subspaces
    subspaces = []
    for s in sample_counts_list[0]:
        subspaces.append(s)

    # construct dictionary which will associate the subspaces to a sample count list for that subspace
    sample_count_list_per_subspace = dict()
    for s in subspaces:
        sample_count_list_per_subspace[s] = []

    # fill dictionary
    for sl in sample_counts_list:
        for s in sl:
            sample_count_list_per_subspace[s].append(sl[s])


    # plot the result
#    plt.plot(sample_count_list_per_subspace[(2,2)],average_error_list, label="2d CQML wrt. basis set size: CCSD(T) + cc-pVDZ contribution")
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_2DCTbasis_jump1_all_levels.csv',sample_count_list_per_subspace[(2,2)],average_error_list)


    # generate a legend for the plot
#    plt.legend()
    # add axis labels and a title
#    plt.xlabel("# most expensive training data (in terms of molecules)")
#    plt.ylabel("MAE [kcal/mol] wrt. CCSD(T) + cc-pVDZ solution")
#    plt.title("cqml comparison (s=1)")

#    plt.show()


# this test compares 3D cqml, 2D cqml and conventional QML for the QM7b data set "with shift" and "s=2", cf. Figure 5 in the paper
# "Boosting quantum machine learning models with multi-level combination technique: Pople diagrams revisited"
def test_cqml_with_shift_jump4(cqml_object, scalings, regularizations):

    ###############################
    # testing combination technique
    ###############################


    # set level jump size
    level_jump_size = 4

    # set evaluation subspace
    evaluation_subspace = (2,2,2)

    # set number of evaluations to do for a single trial
    evaluation_count = 400


    # create a figure
#    f = plt.figure(figsize=(12,9))

    # HF + MP2 + CCSD(T)  (3D CT)
 
    # define maximum learning level (for smallest sampling level)
    maximum_level = 5
    # define training offset
    training_offset = (0,0,0)
    # generate training subspaces
    cqml_object.generate_uniform_ct_training_subspaces_with_shift(0,scalings,regularizations,training_offset, 2)

    # combpute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)


    # get subspaces
    subspaces = []
    for s in sample_counts_list[0]:
        subspaces.append(s)

    # construct dictionary which will associate the subspaces to a sample count list for that subspace
    sample_count_list_per_subspace = dict()
    for s in subspaces:
        sample_count_list_per_subspace[s] = []

    # fill dictionary
    for sl in sample_counts_list:
        for s in sl:
            sample_count_list_per_subspace[s].append(sl[s])


    # plot the contributions for all subspaces
#    plt.xscale("log")
#    plt.yscale("log")
#    plt.plot(sample_count_list_per_subspace[(2,2)],average_error_list, label="3d CQML: CCSD(T) + cc-pVDZ contribution")
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_3DCT_jump2_all_levels.csv',sample_count_list_per_subspace[(2,2)],average_error_list)
   
    # CCSD(T)

    # define maximum learning level (for smallest sampling level)
    maximum_level = 13
    # define training offset
    training_offset = (2,2,0)
    # generate training subspaces
    cqml_object.generate_uniform_ct_training_subspaces(2,scalings,regularizations,training_offset)
    # combpute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)

    # get subspaces
    subspaces = []
    for s in sample_counts_list[0]:
        subspaces.append(s)

    # construct dictionary which will associate the subspaces to a sample count list for that subspace
    sample_count_list_per_subspace = dict()
    for s in subspaces:
        sample_count_list_per_subspace[s] = []

    # fill dictionary
    for sl in sample_counts_list:
        for s in sl:
            sample_count_list_per_subspace[s].append(sl[s])


    # plot the result
#    plt.yscale('log')
#    plt.xscale('log')
#    plt.plot(sample_count_list_per_subspace[(2,2)],average_error_list, label="standard ML: CCSD(T) + cc-pVDZ")
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_jump2_standard_QML.csv',sample_count_list_per_subspace[(2,2)],average_error_list)
 
   
    # fixing ccpvdz basis set, variation in theory (2D CT)

    # define maximum learning level (for smallest sampling level)
    maximum_level = 9
    # define training offset
    training_offset = (0,0,0)
    # generate traning subspaces
    cqml_object.generate_uniform_ct_training_subspaces_with_one_fixed_dimension(1, scalings, regularizations, training_offset, 1, 2)

    # compute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)

    # get subspaces
    subspaces = []
    for s in sample_counts_list[0]:
        subspaces.append(s)

    # construct dictionary which will associate the subspaces to a sample count list for that subspace
    sample_count_list_per_subspace = dict()
    for s in subspaces:
        sample_count_list_per_subspace[s] = []

    # fill dictionary
    for sl in sample_counts_list:
        for s in sl:
            sample_count_list_per_subspace[s].append(sl[s])



    # plot the result
#    plt.plot(sample_count_list_per_subspace[(2,2)],average_error_list, label="2d CQML wrt. theories: CCSD(T) + cc-pVDZ contribution")
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_2DCT_jump2_all_levels.csv',sample_count_list_per_subspace[(2,2)],average_error_list)



    # fixing CCSD(T) theory, variation in basis set size (2D CT)

    # define maximum learning level (for smallest sampling level)
    maximum_level = 9
    # define training offset
    training_offset = (0,0,0)
    # generate training subspaces
    cqml_object.generate_uniform_ct_training_subspaces_with_one_fixed_dimension(1, scalings, regularizations, training_offset, 0, 2)

    # combpute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)

    # get subspaces
    subspaces = []
    for s in sample_counts_list[0]:
        subspaces.append(s)

    # construct dictionary which will associate the subspaces to a sample count list for that subspace
    sample_count_list_per_subspace = dict()
    for s in subspaces:
        sample_count_list_per_subspace[s] = []

    # fill dictionary
    for sl in sample_counts_list:
        for s in sl:
            sample_count_list_per_subspace[s].append(sl[s])


    # plot the result
#    plt.plot(sample_count_list_per_subspace[(2,2)],average_error_list, label="2d CQML wrt. basis set size: CCSD(T) + cc-pVDZ contribution")
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_2DCTbasis_jump2_all_levels.csv',sample_count_list_per_subspace[(2,2)],average_error_list)


    # generate a legend for the plot
#    plt.legend()
    # add axis labels and a title
#    plt.xlabel("# most expensive training data (in terms of molecules)")
#    plt.ylabel("MAE [kcal/mol] wrt. CCSD(T) + cc-pVDZ solution")
#    plt.title("3d CQML")

#    plt.show()


# this test compares 3D cqml, 2D cqml and conventional QML for the QM7b data set "with shift" and "s=2" with respect to the root mean
# square error. This data is used in Figure 5 of the paper
# "Boosting quantum machine learning models with multi-level combination technique: Pople diagrams revisited"
def test_cqml_with_shift_jump4_rmse(cqml_object, scalings, regularizations):

    ###############################
    # testing combination technique
    ###############################


    # set level jump size
    level_jump_size = 4

    # set evaluation subspace
    evaluation_subspace = (2,2,2)

    # set number of evaluations to do for a single trial
    evaluation_count = 400


    # create a figure
#    f = plt.figure(figsize=(12,9))

    # HF + MP2 + CCSD(T)  (3D CT)
 
    # define maximum learning level (for smallest sampling level)
    maximum_level = 5
    # define training offset
    training_offset = (0,0,0)
    # generate training subspaces
    cqml_object.generate_uniform_ct_training_subspaces_with_shift(0,scalings,regularizations,training_offset, 2)

    # combpute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence_rmse(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)


    # get subspaces
    subspaces = []
    for s in sample_counts_list[0]:
        subspaces.append(s)

    # construct dictionary which will associate the subspaces to a sample count list for that subspace
    sample_count_list_per_subspace = dict()
    for s in subspaces:
        sample_count_list_per_subspace[s] = []

    # fill dictionary
    for sl in sample_counts_list:
        for s in sl:
            sample_count_list_per_subspace[s].append(sl[s])


    # plot the contributions for all subspaces
#    plt.xscale("log")
#    plt.yscale("log")
#    plt.plot(sample_count_list_per_subspace[(2,2)],average_error_list, label="3d CQML: CCSD(T) + cc-pVDZ contribution")
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_3DCT_jump2_all_levels_rmse.csv',sample_count_list_per_subspace[(2,2)],average_error_list)
   
    # CCSD(T)

    # define maximum learning level (for smallest sampling level)
    maximum_level = 13
    # define training offset
    training_offset = (2,2,0)
    # generate training subspaces
    cqml_object.generate_uniform_ct_training_subspaces(2,scalings,regularizations,training_offset)
    # combpute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence_rmse(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)

    # get subspaces
    subspaces = []
    for s in sample_counts_list[0]:
        subspaces.append(s)

    # construct dictionary which will associate the subspaces to a sample count list for that subspace
    sample_count_list_per_subspace = dict()
    for s in subspaces:
        sample_count_list_per_subspace[s] = []

    # fill dictionary
    for sl in sample_counts_list:
        for s in sl:
            sample_count_list_per_subspace[s].append(sl[s])


    # plot the result
#    plt.yscale('log')
#    plt.xscale('log')
#    plt.plot(sample_count_list_per_subspace[(2,2)],average_error_list, label="standard ML: CCSD(T) + cc-pVDZ")
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_jump2_standard_QML_rmse.csv',sample_count_list_per_subspace[(2,2)],average_error_list)
 
   
    # fixing ccpvdz basis set, variation in theory (2D CT)

    # define maximum learning level (for smallest sampling level)
    maximum_level = 9
    # define training offset
    training_offset = (0,0,0)
    # generate traning subspaces
    cqml_object.generate_uniform_ct_training_subspaces_with_one_fixed_dimension(1, scalings, regularizations, training_offset, 1, 2)

    # compute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence_rmse(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)

    # get subspaces
    subspaces = []
    for s in sample_counts_list[0]:
        subspaces.append(s)

    # construct dictionary which will associate the subspaces to a sample count list for that subspace
    sample_count_list_per_subspace = dict()
    for s in subspaces:
        sample_count_list_per_subspace[s] = []

    # fill dictionary
    for sl in sample_counts_list:
        for s in sl:
            sample_count_list_per_subspace[s].append(sl[s])



    # plot the result
#    plt.plot(sample_count_list_per_subspace[(2,2)],average_error_list, label="2d CQML wrt. theories: CCSD(T) + cc-pVDZ contribution")
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_2DCT_jump2_all_levels_rmse.csv',sample_count_list_per_subspace[(2,2)],average_error_list)



    # fixing CCSD(T) theory, variation in basis set size (2D CT)

    # define maximum learning level (for smallest sampling level)
    maximum_level = 9
    # define training offset
    training_offset = (0,0,0)
    # generate training subspaces
    cqml_object.generate_uniform_ct_training_subspaces_with_one_fixed_dimension(1, scalings, regularizations, training_offset, 0, 2)

    # combpute convergence
    (average_error_list, sample_counts_list) = cqml_object.check_nested_convergence_rmse(maximum_level, level_jump_size, evaluation_subspace, evaluation_count, trials)

    # get subspaces
    subspaces = []
    for s in sample_counts_list[0]:
        subspaces.append(s)

    # construct dictionary which will associate the subspaces to a sample count list for that subspace
    sample_count_list_per_subspace = dict()
    for s in subspaces:
        sample_count_list_per_subspace[s] = []

    # fill dictionary
    for sl in sample_counts_list:
        for s in sl:
            sample_count_list_per_subspace[s].append(sl[s])


    # plot the result
#    plt.plot(sample_count_list_per_subspace[(2,2)],average_error_list, label="2d CQML wrt. basis set size: CCSD(T) + cc-pVDZ contribution")
    write_data('DataSet_cqml_QM7b/DataSet_cqml_QM7b_2DCTbasis_jump2_all_levels_rmse.csv',sample_count_list_per_subspace[(2,2)],average_error_list)


    # generate a legend for the plot
#    plt.legend()
    # add axis labels and a title
#    plt.xlabel("# most expensive training data (in terms of molecules)")
#    plt.ylabel("MAE [kcal/mol] wrt. CCSD(T) + cc-pVDZ solution")
#    plt.title("3d CQML")

#    plt.show()




def main():

    # select whether to use already precomputed data
    use_precomputed_data = False

    # initialize combination technique object
    cqml_object = cq.cqml()
   
    # define energy subspaces to learn from
    cqml_object.subspaces = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    
    if (not use_precomputed_data):
        # set training data files
        cqml_object.data_files[(0,0)] = "DataSet_cqml_QM7b/HF_sto3g.txt"
        cqml_object.data_files[(0,1)] = "DataSet_cqml_QM7b/HF_631g.txt"
        cqml_object.data_files[(0,2)] = "DataSet_cqml_QM7b/HF_ccpvdz.txt"
        cqml_object.data_files[(1,0)] = "DataSet_cqml_QM7b/MP2_sto3g.txt"
        cqml_object.data_files[(1,1)] = "DataSet_cqml_QM7b/MP2_631g.txt"
        cqml_object.data_files[(1,2)] = "DataSet_cqml_QM7b/MP2_ccpvdz.txt"
        cqml_object.data_files[(2,0)] = "DataSet_cqml_QM7b/CCSDT_sto3g.txt"
        cqml_object.data_files[(2,1)] = "DataSet_cqml_QM7b/CCSDT_631g.txt"
        cqml_object.data_files[(2,2)] = "DataSet_cqml_QM7b/CCSDT_ccpvdz.txt"
     
        # set molecule data directories
        # WARNING: "cqml" (not "cqml2d") currently only supports one single geometry for all subspaces !
        cqml_object.xyz_directories[(0,0)] = "DataSet_cqml_QM7b/geometry"
        cqml_object.xyz_directories[(0,1)] = "DataSet_cqml_QM7b/geometry"
        cqml_object.xyz_directories[(0,2)] = "DataSet_cqml_QM7b/geometry"
        cqml_object.xyz_directories[(1,0)] = "DataSet_cqml_QM7b/geometry"
        cqml_object.xyz_directories[(1,1)] = "DataSet_cqml_QM7b/geometry"
        cqml_object.xyz_directories[(1,2)] = "DataSet_cqml_QM7b/geometry"
        cqml_object.xyz_directories[(2,0)] = "DataSet_cqml_QM7b/geometry"
        cqml_object.xyz_directories[(2,1)] = "DataSet_cqml_QM7b/geometry"
        cqml_object.xyz_directories[(2,2)] = "DataSet_cqml_QM7b/geometry"
     
        # set number of molecules (to be loaded) per energy subspace
        N = 7211
        cqml_object.N_load[(0,0)] = N
        cqml_object.N_load[(0,1)] = N
        cqml_object.N_load[(0,2)] = N
        cqml_object.N_load[(1,0)] = N
        cqml_object.N_load[(1,1)] = N
        cqml_object.N_load[(1,2)] = N
        cqml_object.N_load[(2,0)] = N
        cqml_object.N_load[(2,1)] = N
        cqml_object.N_load[(2,2)] = N
    
        # fix costs per energy subspace
        cqml_object.subspace_costs[(0,0)] = 1
        cqml_object.subspace_costs[(0,1)] = 1
        cqml_object.subspace_costs[(0,2)] = 1
        cqml_object.subspace_costs[(1,0)] = 1
        cqml_object.subspace_costs[(1,1)] = 1
        cqml_object.subspace_costs[(1,2)] = 1
        cqml_object.subspace_costs[(2,0)] = 1
        cqml_object.subspace_costs[(2,1)] = 1
        cqml_object.subspace_costs[(2,2)] = 1
   
        # load the data
        cqml_object.load_data()


    # define scalings for kernel ridge regression per energy subpspace
    scalings = dict();
    scalings[(0,0)] = 400
    scalings[(0,1)] = 400
    scalings[(0,2)] = 400
    scalings[(1,0)] = 400
    scalings[(1,1)] = 400
    scalings[(1,2)] = 400
    scalings[(2,0)] = 400
    scalings[(2,1)] = 400
    scalings[(2,2)] = 400
    
    
    # define regularizations per energy subspace
    regularizations = dict()
    regularizations[(0,0)] = 10**(-10)
    regularizations[(0,1)] = 10**(-10)
    regularizations[(0,2)] = 10**(-10)
    regularizations[(1,0)] = 10**(-10)
    regularizations[(1,1)] = 10**(-10)
    regularizations[(1,2)] = 10**(-10)
    regularizations[(2,0)] = 10**(-10)
    regularizations[(2,1)] = 10**(-10)
    regularizations[(2,2)] = 10**(-10)


    if (not use_precomputed_data):

        # construct all training kernel matrices
        cqml_object.generate_full_kernel_matrices(scalings, regularizations)
    
        # save CT data
        cqml_object.save_binary("DataSet_cqml_QM7b/cqml_QM7b_data.dat")
     
    if (use_precomputed_data):
        # load precomputed CT object
        print("Loading data ...")
        cqml_object = cq.cqml.load("DataSet_cqml_QM7b/cqml_QM7b_data.dat")
        print("Done!")


    # set number of dimensions of CT. Here (theory_level,basis_set_size,sampling_level) -> 3
    cqml_object.ct_dim=3


    test_cqml_with_shift(cqml_object, scalings, regularizations)
    test_cqml_with_shift_jump4(cqml_object, scalings, regularizations)
    test_cqml_with_shift_jump4_rmse(cqml_object, scalings, regularizations)



if __name__ == '__main__':
    main()
