#!/usr/bin/env python
# MIT License
#
# Copyright (c) 2017 Anders Steen Christensen
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

import os

def mkl_exists(verbose=False):

    # Get environment variables
    __MKLROOT__ = os.environ.get('MKLROOT')
    __LD_LIBRARY_PATH__ = os.environ.get('LD_LIBRARY_PATH')

    # Check if $MKLROOT is set

    if __MKLROOT__ is None:

        if verbose: 
            print("MKL-discover: MKLROOT was not set")

        return False

    else:

        if verbose: 
            print("MKL-discover: MKLROOT was set to")
            print(__MKLROOT__)

    # Check if path exists
    mklroot_exists = os.path.isdir(__MKLROOT__)
    
    if not mklroot_exists:
        if verbose: 
            print("MKL-discover: MKLROOT path does not exist")

        return False

    found_libmkl_rt = False

    # Check if libmkl_rt.so exists below $MKLROOT
    for dirpath, dirnames, filenames in os.walk(__MKLROOT__, followlinks=True):

        if "libmkl_rt.so" in filenames:

            if verbose:
                print("MKL-discover: Found libmkl_rt.so at ", dirpath)

            # Check that the dirpath where libmkl_rt.so is in $LD_LIBRARY_PATH
            if dirpath in __LD_LIBRARY_PATH__:

                if verbose:
                    print("MKL-discover: Found %s in $LD_LIBRARY_PATH" % dirpath)
                    print("MKL-discover: Concluding that MKL can be used.")
                found_libmkl_rt = True


    return found_libmkl_rt

if __name__ == "__main__":

    mkl_present = mkl_exists(verbose=False)

    if mkl_present:
        print("MKL could be found")
    else:
        print("MKL could NOT be found")
