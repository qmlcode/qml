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
#import matplotlib.pyplot as plt

import qml
from qml.models.cqml2d import *


trials = 20


def write_data(filename, costs, errors):
    outfile = open(filename,'w')
    outfile.write("N error\n")
    for i in range(len(errors)):
        outfile.write("%d %e\n" % (costs[i],errors[i]))
    outfile.close()

# this test compares cqml2d with different numbers of level (PM7 + DFT + G4MP2 vs. DFT + G4MP2 vs. G4MP2) for the CI9 data set
# with Coulomb matrix representation, cf. Figure 6 in the paper
# "Boosting quantum machine learning models with multi-level cqml_objectnation technique: Pople diagrams revisited"
def test_cqml2d_coulomb():

    #########################
    # loading multilevel data
    #########################

    # getting structure for config of multilevel data     
    ml_data_cfg = multilevel_data_config()

    # setting data files per level for data which is to be learned
    ml_data_cfg.data_files = ["DataSet_cqml2d_CI9/Y1.dat","DataSet_cqml2d_CI9/Y2.dat","DataSet_cqml2d_CI9/Y3.dat"]

    # setting directories per level for the xyz files
    ml_data_cfg.xyz_directories = ["DataSet_cqml2d_CI9/db/ci6k_PM7","DataSet_cqml2d_CI9/db/ci6k_B3LYP631G2DFP","DataSet_cqml2d_CI9/db/ci6k_G4MP2"]

    # choose representation
    ml_data_cfg.representation_type = "coulomb_matrix"

    # choose the number of compounds that shall be loaded
    ml_data_cfg.N_load = 6095

    # choose the number of levels to load
    ml_data_cfg.level_count_load = 3

    # load the multilevel data
    ml_data = load_multilevel_data(ml_data_cfg)
    
    # artificially imposint costs for the level (in the future this shall be delivered with the data)
    ml_data.level_costs = [0,0,1]

    ###############################
    # testing combination technique
    ###############################

    # getting structure for the configuraiton of the combination technique test
    ct_cfg = cqml2d_config()

    # choose the maximum resolution level (i.e. number of learning samples = 2^max_resolution_level) for which the convergence shall be checked
    ct_cfg.max_resolution_level = 12 # 9

    # set the scaling parameters for each level (this is the global level as in the ml_data structure)
    ct_cfg.scalings = [400,400,400]

    # set the regularization parameter for each level (this is the clobal level as in the ml_data structure)
    ct_cfg.regularizations = [10**(-10),10**(-10),10**(-10)];
    
    # set the jump size between each individual level
    ct_cfg.level_jump_size = [1]

    # set the (global!) level on which the error shall be evaluated (starting from 0!)
    ct_cfg.error_level = 2

    
    # set the current (global!) base level from which the combination technique shall be started (starting from 0)
    # here: start on level 0
    ct_cfg.base_level = 0
    # set the current (local!) number of levels for which the combination technique shall be computed
    # here: use all levels
    ct_cfg.level_count = 3

    # => this is the full combination technique on all levels

    # create a figure
#    f = plt.figure()

    # do the computation with averaging over "trials" trials
    (errors, costs) = compute_cqml2d_convergence(ml_data_cfg, ml_data, ct_cfg, trials);
    # plot the result
#    plt.loglog(costs,errors, label="PM7+DFT+G4MP2")
    write_data('DataSet_cqml2d_CI9/DataSet_cqml2d_CI9_PM7_DFT_G4MP2_coulomb.csv',costs,errors)
    
    # here: start on level 1
    ct_cfg.base_level = 1
    # here: use only DFT+G4MP2
    ct_cfg.level_count = 2

    # => this corresponds to a modified version of Delta-ML

    # compute
    (errors, costs) = compute_cqml2d_convergence(ml_data_cfg, ml_data, ct_cfg, trials);
    # plot
#    plt.loglog(costs,errors, label="DFT+G4MP2")
    write_data('DataSet_cqml2d_CI9/DataSet_cqml2d_CI9_DFT_G4MP2_coulomb.csv',costs,errors)

    # here: start on level 2
    ct_cfg.base_level = 2
    # here: use only one level
    ct_cfg.level_count = 1

    # => this corresponds to standard learning on level 2

    # compute
    (errors, costs) = compute_cqml2d_convergence(ml_data_cfg, ml_data, ct_cfg, trials);
    # plot
#    plt.loglog(costs,errors, label="G4MP2")
    write_data('DataSet_cqml2d_CI9/DataSet_cqml2d_CI9_G4MP2_coulomb.csv',costs,errors)

    # generate a legend for the plot
#    plt.legend()
    # add axis labels and a title
#    plt.xlabel("number of most expensive molecules")
#    plt.ylabel("MAE [kcal/mol] wrt. G4MP2 solution")
#    plt.title("CQML: Plot wrt. number of most expensive molecules")

#    plt.show()



# this test compares cqml2d with different numbers of level (PM7 + DFT + G4MP2 vs. DFT + G4MP2 vs. G4MP2) for the CI9 data set
# with SLATM representation, cf. Figure 6 in the paper
# "Boosting quantum machine learning models with multi-level cqml_objectnation technique: Pople diagrams revisited"
def test_cqml2d_slatm():

    #########################
    # loading multilevel data
    #########################

    # getting structure for config of multilevel data     
    ml_data_cfg = multilevel_data_config()

    # setting data files per level for data which is to be learned
    ml_data_cfg.data_files = ["DataSet_cqml2d_CI9/Y1.dat","DataSet_cqml2d_CI9/Y2.dat","DataSet_cqml2d_CI9/Y3.dat"]

    # setting directories per level for the xyz files
    ml_data_cfg.xyz_directories = ["DataSet_cqml2d_CI9/db/ci6k_PM7","DataSet_cqml2d_CI9/db/ci6k_B3LYP631G2DFP","DataSet_cqml2d_CI9/db/ci6k_G4MP2"]

    # choose representation
    ml_data_cfg.representation_type = "slatm"

    # choose the number of compounds that shall be loaded
    ml_data_cfg.N_load = 6095

    # choose the number of levels to load
    ml_data_cfg.level_count_load = 3

    # load the multilevel data
    ml_data = load_multilevel_data(ml_data_cfg)
    
    # artificially imposint costs for the level (in the future this shall be delivered with the data)
    ml_data.level_costs = [0,0,1]

    ###############################
    # testing combination technique
    ###############################

    # getting structure for the configuraiton of the combination technique test
    ct_cfg = cqml2d_config()

    # choose the maximum resolution level (i.e. number of learning samples = 2^max_resolution_level) for which the convergence shall be checked
    ct_cfg.max_resolution_level = 12 # 9

    # set the scaling parameters for each level (this is the global level as in the ml_data structure)
    ct_cfg.scalings = [400,400,400]

    # set the regularization parameter for each level (this is the clobal level as in the ml_data structure)
    ct_cfg.regularizations = [10**(-10),10**(-10),10**(-10)];
    
    # set the jump size between each individual level
    ct_cfg.level_jump_size = [1]

    # set the (global!) level on which the error shall be evaluated (starting from 0!)
    ct_cfg.error_level = 2

    
    # set the current (global!) base level from which the combination technique shall be started (starting from 0)
    # here: start on level 0
    ct_cfg.base_level = 0
    # set the current (local!) number of levels for which the combination technique shall be computed
    # here: use all levels
    ct_cfg.level_count = 3

    # => this is the full combination technique on all levels

    # create a figure
#    f = plt.figure()

    # do the computation with averaging over "trials" trials
    (errors, costs) = compute_cqml2d_convergence(ml_data_cfg, ml_data, ct_cfg, trials);
    # plot the result
#    plt.loglog(costs,errors, label="PM7+DFT+G4MP2")
    write_data('DataSet_cqml2d_CI9/DataSet_cqml2d_CI9_PM7_DFT_G4MP2_slatm.csv',costs,errors)
    
    # here: start on level 1
    ct_cfg.base_level = 1
    # here: use only DFT+G4MP2
    ct_cfg.level_count = 2

    # => this corresponds to a modified version of Delta-ML

    # compute
    (errors, costs) = compute_cqml2d_convergence(ml_data_cfg, ml_data, ct_cfg, trials);
    # plot
#    plt.loglog(costs,errors, label="DFT+G4MP2")
    write_data('DataSet_cqml2d_CI9/DataSet_cqml2d_CI9_DFT_G4MP2_slatm.csv',costs,errors)

    # here: start on level 2
    ct_cfg.base_level = 2
    # here: use only one level
    ct_cfg.level_count = 1

    # => this corresponds to standard learning on level 2

    # compute
    (errors, costs) = compute_cqml2d_convergence(ml_data_cfg, ml_data, ct_cfg, trials);
    # plot
#    plt.loglog(costs,errors, label="G4MP2")
    write_data('DataSet_cqml2d_CI9/DataSet_cqml2d_CI9_G4MP2_slatm.csv',costs,errors)

    # generate a legend for the plot
#    plt.legend()
    # add axis labels and a title
#    plt.xlabel("number of most expensive molecules")
#    plt.ylabel("MAE [kcal/mol] wrt. G4MP2 solution")
#    plt.title("CQML: Plot wrt. number of most expensive molecules")

#    plt.show()



def main():

    test_cqml2d_coulomb()
    test_cqml2d_slatm()

if __name__ == '__main__':
    main()
