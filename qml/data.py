# MIT License
#
# Copyright (c) 2016 Anders Steen Christensen
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

NUCLEAR_CHARGE = {
 'H'  :     1, 
 'He' :     2, 
 'Li' :     3, 
 'Be' :     4, 
 'B'  :     5, 
 'C'  :     6, 
 'N'  :     7, 
 'O'  :     8, 
 'F'  :     9, 
 'Ne' :    10, 
 'Na' :    11, 
 'Mg' :    12, 
 'Al' :    13, 
 'Si' :    14, 
 'P'  :    15, 
 'S'  :    16, 
 'Cl' :    17, 
 'Ar' :    18, 
 'K'  :    19, 
 'Ca' :    20, 
 'Sc' :    21, 
 'Ti' :    22, 
 'V'  :    23, 
 'Cr' :    24, 
 'Mn' :    25, 
 'Fe' :    26, 
 'Co' :    27, 
 'Ni' :    28, 
 'Cu' :    29, 
 'Zn' :    30, 
 'Ga' :    31, 
 'Ge' :    32, 
 'As' :    33, 
 'Se' :    34, 
 'Br' :    35, 
 'Kr' :    36, 
 'Rb' :    37, 
 'Sr' :    38, 
 'Y'  :    39, 
 'Zr' :    40, 
 'Nb' :    41, 
 'Mo' :    42, 
 'Tc' :    43, 
 'Ru' :    44, 
 'Rh' :    45, 
 'Pd' :    46, 
 'Ag' :    47, 
 'Cd' :    48, 
 'In' :    49, 
 'Sn' :    50, 
 'Sb' :    51, 
 'Te' :    52, 
 'I'  :    53, 
 'Xe' :    54, 
 'Cs' :    55, 
 'Ba' :    56, 
 'La' :    57, 
 'Ce' :    58, 
 'Pr' :    59, 
 'Nd' :    60, 
 'Pm' :    61, 
 'Sm' :    62, 
 'Eu' :    63, 
 'Gd' :    64, 
 'Tb' :    65, 
 'Dy' :    66, 
 'Ho' :    67, 
 'Er' :    68, 
 'Tm' :    69, 
 'Yb' :    70, 
 'Lu' :    71, 
 'Hf' :    72, 
 'Ta' :    73, 
 'W'  :    74, 
 'Re' :    75, 
 'Os' :    76, 
 'Ir' :    77, 
 'Pt' :    78, 
 'Au' :    79, 
 'Hg' :    80, 
 'Tl' :    81, 
 'Pb' :    82, 
 'Bi' :    83, 
 'Po' :    84, 
 'At' :    85, 
 'Rn' :    86, 
 'Fr' :    87, 
 'Ra' :    88, 
 'Ac' :    89, 
 'Th' :    90, 
 'Pa' :    91, 
 'U'  :    92, 
 'Np' :    93, 
 'Pu' :    94, 
 'Am' :    95, 
 'Cm' :    96, 
 'Bk' :    97, 
 'Cf' :    98, 
 'Es' :    99, 
 'Fm' :   100, 
 'Md' :   101, 
 'No' :   102, 
 'Lr' :   103, 
 'Rf' :   104, 
 'Db' :   105, 
 'Sg' :   106, 
 'Bh' :   107, 
 'Hs' :   108, 
 'Mt' :   109, 
 'Ds' :   110, 
 'Rg' :   111, 
 'Cn' :   112, 
 'Uuq':   114, 
 'Uuh':   116}

# Periodic table indexes
PTP = {
         1  :[1,1] ,2:  [1,8] #Row1

        ,3  :[2,1] ,4:  [2,2] #Row2
        ,5  :[2,3] ,6:  [2,4] ,7  :[2,5] ,8  :[2,6] ,9  :[2,7] ,10 :[2,8]

        ,11 :[3,1] ,12: [3,2] #Row3
        ,13 :[3,3] ,14: [3,4] ,15 :[3,5] ,16 :[3,6] ,17 :[3,7] ,18 :[3,8]

        ,19 :[4,1] ,20: [4,2] #Row4
        ,31 :[4,3] ,32: [4,4] ,33 :[4,5] ,34 :[4,6] ,35 :[4,7] ,36 :[4,8]
        ,21 :[4,9] ,22: [4,10],23 :[4,11],24 :[4,12],25 :[4,13],26 :[4,14],27 :[4,15],28 :[4,16],29 :[4,17],30 :[4,18]

        ,37 :[5,1] ,38: [5,2] #Row5
        ,49 :[5,3] ,50: [5,4] ,51 :[5,5] ,52 :[5,6] ,53 :[5,7] ,54 :[5,8]
        ,39 :[5,9] ,40: [5,10],41 :[5,11],42 :[5,12],43 :[5,13],44 :[5,14],45 :[5,15],46 :[5,16],47 :[5,17],48 :[5,18]

        ,55 :[6,1] ,56: [6,2] #Row6
        ,81 :[6,3] ,82: [6,4] ,83 :[6,5] ,84 :[6,6] ,85 :[6,7] ,86 :[6,8]
               ,72: [6,10],73 :[6,11],74 :[6,12],75 :[6,13],76 :[6,14],77 :[6,15],78 :[6,16],79 :[6,17],80 :[6,18]
        ,57 :[6,19],58: [6,20],59 :[6,21],60 :[6,22],61 :[6,23],62 :[6,24],63 :[6,25],64 :[6,26],65 :[6,27],66 :[6,28],67 :[6,29],68 :[6,30],69 :[6,31],70 :[6,32],71 :[6,33]

        ,87 :[7,1] ,88: [7,2] #Row7
        ,113:[7,3] ,114:[7,4] ,115:[7,5] ,116:[7,6] ,117:[7,7] ,118:[7,8]
               ,104:[7,10],105:[7,11],106:[7,12],107:[7,13],108:[7,14],109:[7,15],110:[7,16],111:[7,17],112:[7,18]
        ,89 :[7,19],90: [7,20],91 :[7,21],92 :[7,22],93 :[7,23],94 :[7,24],95 :[7,25],96 :[7,26],97 :[7,27],98 :[7,28],99 :[7,29],100:[7,30],101:[7,31],101:[7,32],102:[7,14],103:[7,33]}
