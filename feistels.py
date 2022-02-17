#!/usr/bin/python3
# -*- coding: utf-8 -*-

#=========================================================================
#Copyright (c) 2022

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#=========================================================================

#This project has been supported by ERC-ADG-ALGSTRONGCRYPTO (project 740972).

#=========================================================================

# REQUIREMENT LIST
#- Python 3.x with x >= 2
#- the SCIP solver, see https://scip.zib.de/
#- pyscipopt, see https://github.com/SCIP-Interfaces/PySCIPOpt
#- (for Sparkle) CryptominiSAT and pycryptosat

#=========================================================================

# Author: André Schrottenloher & Marc Stevens
# Date: January 2022
# Version: 1

#=========================================================================
"""
Several use cases of our generic MILP tool, on Feistel-like designs. Here
we use the 4-list program in generic4.py, since some of the results need
4 lists.

We only use full wrapping constraints here, which are quite strong, but better
attacks can be obtained with Guess-and-Determine (see the "feistelsgad" folder).
"""

from generic4 import find_mitm_attack, EXTENDED_SETTING, CLASSICAL_COMPUTATION, SINGLE_SOLUTION
from util import PresentConstraints


def simpira4(nrounds=4):
    """
    Simpira-4 constraints.
    """
    actual_nrounds = 2 * nrounds
    cons = PresentConstraints(nrounds=actual_nrounds)

    for r in range(actual_nrounds):
        if r % 2 == 0:
            for i in range(4):
                cons.add_cell(r, w=1)
        elif r % 4 == 1:
            cons.add_cell(r, w=1)
            cons.add_cell(r, w=2)
            cons.add_cell(r, w=1)
            cons.add_cell(r, w=2)
        elif r % 4 == 3:
            cons.add_cell(r, w=2)
            cons.add_cell(r, w=1)
            cons.add_cell(r, w=2)
            cons.add_cell(r, w=1)

    for r in range(actual_nrounds):
        for i in range(4):
            cons.add_edge_2(r, i, i, w=1)
        if r % 4 == 0:
            cons.add_edge_2(r, 0, 1, w=1)
            cons.add_edge_2(r, 2, 3, w=1)
        elif r % 4 == 2:
            cons.add_edge_2(r, 1, 2, w=1)
            cons.add_edge_2(r, 3, 0, w=1)

    return cons


def simpira3(nrounds=4):
    """
    Simpira-3 constraints.
    """
    actual_nrounds = 2 * nrounds
    cons = PresentConstraints(nrounds=actual_nrounds)

    for r in range(actual_nrounds):
        if r % 2 == 0:
            for i in range(3):
                cons.add_cell(r, w=1)
        elif r % 6 == 1:
            cons.add_cell(r, w=1)
            cons.add_cell(r, w=2)
            cons.add_cell(r, w=1)
        elif r % 6 == 3:
            cons.add_cell(r, w=1)
            cons.add_cell(r, w=1)
            cons.add_cell(r, w=2)
        elif r % 6 == 5:
            cons.add_cell(r, w=2)
            cons.add_cell(r, w=1)
            cons.add_cell(r, w=1)

    for r in range(actual_nrounds):
        for i in range(3):
            cons.add_edge_2(r, i, i, w=1)
        if r % 6 == 0:
            cons.add_edge_2(r, 0, 1, w=1)
        elif r % 6 == 2:
            cons.add_edge_2(r, 1, 2, w=1)
        elif r % 6 == 4:
            cons.add_edge_2(r, 2, 0, w=1)

    return cons


# basic Feistel. We don't find any attack on that.
#def simpira2(nrounds=2):
#    actual_nrounds = 2*nrounds
#    cons = PresentConstraints(nrounds=actual_nrounds)

#    for r in range(actual_nrounds):
#        if r % 2 == 0:
#            for i in range(2):
#                cons.add_cell(r, w=1)
#        elif r % 4 == 1:
#            cons.add_cell(r, w=1)
#            cons.add_cell(r, w=2)
#        elif r % 4 == 3:
#            cons.add_cell(r, w=2)
#            cons.add_cell(r, w=1)
#
#    for r in range(actual_nrounds):
#        for i in range(2):
#            cons.add_edge_2(r, i, i, w=1)
#        if r % 4 == 0:
#            cons.add_edge_2(r, 0, 1, w=1)
#        elif r % 4 == 2:
#            cons.add_edge_2(r, 1, 0, w=1)

#    return cons


def simpira8(nrounds=4):
    """
    Simpira-8 constraints.
    """
    actual_nrounds = 2 * nrounds
    cons = PresentConstraints(nrounds=actual_nrounds)
    s = [0, 1, 6, 5, 4, 3]
    t = [2, 7]

    for actualr in range(actual_nrounds):
        if actualr % 2 == 0:
            for i in range(8):
                cons.add_cell(actualr, w=1)
        else:
            r = actualr // 2
            # positions of receivers
            l = [
                s[(r + 1) % 6], s[(r + 5) % 6], s[(r + 3) % 6], t[(r + 1) % 2]
            ]
            for i in range(8):
                if i in l:
                    cons.add_cell(actualr, w=2)
                else:
                    cons.add_cell(actualr, w=1)

    for actualr in range(actual_nrounds):
        for i in range(8):
            cons.add_edge_2(actualr, i, i, w=1)
        if actualr % 2 == 0:
            r = actualr // 2
            # connect receiver nodes to the ones that sent the branches
            l = [(s[(r + 1) % 6], s[(r) % 6]), (s[(r + 5) % 6], t[(r) % 2]),
                 (s[(r + 3) % 6], s[(r + 4) % 6]),
                 (t[(r + 1) % 2], s[(r + 2) % 6])]
            for (a, b) in l:
                cons.add_edge_2(actualr, b, a, w=1)
    return cons


def simpira6(nrounds=4):
    """
    Simpira-6 constraints.
    """
    actual_nrounds = 2 * nrounds
    cons = PresentConstraints(nrounds=actual_nrounds)
    s = [0, 1, 2, 5, 4, 3]

    for actualr in range(actual_nrounds):
        if actualr % 2 == 0:
            for i in range(8):
                cons.add_cell(actualr, w=1)
        else:
            r = actualr // 2
            # positions of receivers
            l = [s[(r + 1) % 6], s[(r + 5) % 6], s[(r + 3) % 6]]
            for i in range(8):
                if i in l:
                    cons.add_cell(actualr, w=2)
                else:
                    cons.add_cell(actualr, w=1)

    for actualr in range(actual_nrounds):
        for i in range(6):
            cons.add_edge_2(actualr, i, i, w=1)
        if actualr % 2 == 0:
            r = actualr // 2
            # connect receiver nodes to the ones that sent the branches
            l = [(s[(r + 1) % 6], s[(r) % 6]),
                 (s[(r + 5) % 6], s[(r + 2) % 2]),
                 (s[(r + 3) % 6], s[(r + 4) % 6])]
            for (a, b) in l:
                cons.add_edge_2(actualr, b, a, w=1)
    return cons


def sparkle(b=3, nrds=4):
    """
    All versions of Sparkle. Here the number of branches is actually 2*b.
    The naming of cells is the same as in the drawings given in the paper.
    
    The structure is always a full wrapping. Each round has 4 layers:
    * layer 0: b 3-branch cells on the left, b dummy cells on the right
    * layer 1: b dummy cells on the left, one (b)-XOR cell middle (M layer), XOR cells on the right (branch addition)
    * layer 2: dummy cells, one b-branch cell in the middle (below M)
    * layer 3: XOR cells on the left (M addition)
    """
    # permutation of branches after a round: branch i is connected to
    # permutation[i]
    if b == 3:
        permutation = [3, 4, 5, 2, 0, 1]
    elif b == 2:
        permutation = [2, 3, 1, 0]
    elif b == 4:
        permutation = [4, 5, 6, 7, 3, 0, 1, 2]
    else:
        raise ValueError("not implemented")

    cons = PresentConstraints(nrounds=4 * nrds)

    infos = {}  # cell type. Allows some checking.
    for i in range(nrds):
        # populate the cells (with names, that'll make easier)
        # level 0 cells
        for j in range(b):
            _n = ("S%i_%i" % (i, j))
            infos[_n] = "3-branch"
            cons.add_cell(r=4 * i, w=1, name=_n)
        for j in range(b, 2 * b):
            _n = ("S%i_%i" % (i, j))
            infos[_n] = "dummy"
            cons.add_cell(r=4 * i, w=1, name=_n)
        # level 1 cells
        for j in range(b):
            _n = "T%i_%i" % (i, j)
            infos[_n] = "dummy"
            cons.add_cell(r=4 * i + 1, w=1, name=_n)
        _n = "MX%i" % (i)
        infos[_n] = "%i-XOR" % b
        cons.add_cell(r=4 * i + 1, w=b, name=_n)
        for j in range(b, 2 * b):
            _n = "T%i_%i" % (i, j)
            infos[_n] = "2-XOR"
            cons.add_cell(r=4 * i + 1, w=2, name=_n)
        # connections between level 0 and level 1
        # simple connections
        for j in range(2 * b):
            cons.add_edge(c1=("S%i_%i" % (i, j)), c2="T%i_%i" % (i, j), w=1)
        # connect branches to M, and to XOR on the right
        for j in range(b):
            _n = ("S%i_%i" % (i, j))
            cons.add_edge(c1=_n, c2="T%i_%i" % (i, j + b), w=1)
            cons.add_edge(c1=_n, c2="MX%i" % (i), w=1)

        # level 2 cells: only dummies, except below MX, whee we put MB (branch)
        for j in range(2 * b):
            _n = "U%i_%i" % (i, j)
            infos[_n] = "dummy"
            cons.add_cell(r=4 * i + 2, w=1, name=_n)
        _n = "MB%i" % i
        infos[_n] = "%i-branch" % b
        cons.add_cell(r=4 * i + 2, w=1, name=_n)
        # connections between level 1 and level 2
        for j in range(2 * b):
            cons.add_edge(c1=("T%i_%i" % (i, j)), c2="U%i_%i" % (i, j), w=1)
        cons.add_edge(c1="MX%i" % (i), c2="MB%i" % i, w=1)

        # level 3 cells: dummies on the left, XORs on the right
        for j in range(b):
            _n = ("V%i_%i" % (i, j))
            infos[_n] = "dummy"
            cons.add_cell(r=4 * i + 3, w=1, name=_n)
        for j in range(b, 2 * b):
            _n = ("V%i_%i" % (i, j))
            infos[_n] = "2-XOR"
            cons.add_cell(r=4 * i + 3, w=2, name=_n)
        # connections between level 2 and level 3
        for j in range(2 * b):
            cons.add_edge(c1=("U%i_%i" % (i, j)), c2=("V%i_%i" % (i, j)), w=1)
        for j in range(b):
            cons.add_edge(c1="MB%i" % i, c2="V%i_%i" % (i, j + b), w=1)

    # now for each round, we must connect
    # "V%i_%i" to S%i_%i
    for i in range(nrds - 1):
        for j in range(2 * b):
            cons.add_edge(c1="V%i_%i" % (i, j),
                          c2="S%i_%i" % ((i + 1) % nrds, permutation[j]),
                          w=1)
    # for last round, skip the permutation of branches
    i = nrds - 1
    for j in range(2 * b):
        cons.add_edge(c1="V%i_%i" % (i, j),
                      c2="S%i_%i" % ((i + 1) % nrds, j),
                      w=1)

    # check
    for c in cons.cell_name_to_data:
        if infos[c] == "dummy":
            assert cons.fwd_edges_width(c) == 1
            assert cons.bwd_edges_width(c) == 1
            assert cons.get_cell_width(c) == 1
        elif infos[c] == "2-XOR":
            assert cons.fwd_edges_width(c) == 1
            assert cons.bwd_edges_width(c) == 2
            assert cons.get_cell_width(c) == 2

    return cons


_HELP = """
Usage : python3 feistels.py attack width

Demonstrates some attacks (full-wrapping distinguishers on Feistel networks). 
Parameters (number of rounds...) are in the script.
attack:
- simpira : the Simpira permutations
- sparkle : the Sparkle permutations

width: specifies the variant for the given design (e.g. 3,4,6,8 for Simpira and 2,3,4
for Sparkle).

All examples here are classical.
"""

if __name__ == "__main__":
    import sys

    argc = len(sys.argv)
    if argc < 2:
        print(_HELP)
        sys.exit(0)
    attack = sys.argv[1]

    if argc < 3:
        width = 3
    else:
        width = int(sys.argv[2])

    computation_model = CLASSICAL_COMPUTATION

    cut_forward = []
    cut_backward = []
    covered_round = None
    optimize_with_mem = False
    time_target = None

    if attack == "simpira":
        if width not in [3, 4, 6, 8]:
            raise ValueError("Invalid variant")
        # demonstrates the max. number of rounds on which the solver finds a distinguisher
        max_nrounds = {3: 8, 4: 7, 6: 9, 8: 9}
        nrounds = max_nrounds[width]
        time_target = width - 1  # 3
        cell_nbr = width
        if width == 8:
            cons = simpira8(nrounds=nrounds)
        elif width == 4:
            cons = simpira4(nrounds=nrounds)
        elif width == 3:
            cons = simpira3(nrounds=nrounds)
        elif width == 6:
            cons = simpira6(nrounds=nrounds)

    elif attack == "sparkle":
        if width not in [2, 3, 4]:
            raise ValueError("Invalid variant")
        nrounds = 4
        # do not always converge, but we get some results:
        # 4 round attacks always (for full wrapping: time 3, 5 and 6)
        # no 5 round attacks, or so it seems.
        # This is quite coherent with the analysis by hand & similar to the
        # results on Simpira.
        cons = sparkle(b=width, nrds=nrounds)
    else:
        raise ValueError("Invalid attack")

    cell_var_covered, global_lincons = find_mitm_attack(
        cons,
        time_target=time_target,
        setting=EXTENDED_SETTING,
        computation_model=CLASSICAL_COMPUTATION,
        flag=SINGLE_SOLUTION,
        optimize_with_mem=optimize_with_mem,
        covered_round=covered_round,
        cut_forward=cut_forward,
        cut_backward=cut_backward)

    #=================================
    # picture conversion. Not supported in the distributed code.
    try:
        from tikz_util import convert_to_present_pic
        TIKZ_MODULE_IMPORTED = True
    except ImportError:
        # means that the tikz_util module does not exist
        TIKZ_MODULE_IMPORTED = False

    if TIKZ_MODULE_IMPORTED and attack == "simpira":
        str_pic = convert_to_present_pic(cons,
                                         cell_var_covered,
                                         global_lincons,
                                         flag="simpira",
                                         cell_nbr=cell_nbr,
                                         display_cell_names=True,
                                         only_cells=False)
        print(str_pic)
