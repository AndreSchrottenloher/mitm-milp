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
# Date: June 2022
# Version: 2

#=========================================================================
"""
Several use cases of our generic MILP tool on the Gimli permutation. Here
we use the 4-list program in generic4.py, since the results need 4 lists.
"""

from generic4 import find_mitm_attack, CLASSICAL_COMPUTATION, SINGLE_SOLUTION, ALL_SOLUTIONS, PRESENT_SETTING
from util import PresentConstraints


def make_gimli_constraints(n_double_rounds=6, flag="full-wrapping"):
    """
    Creates the constraints for Gimli attacks.
    Parameters:
    n_double_rounds -- the actual number of "rounds" attacked is twice this amount
    flag -- "full-wrapping" for afull wrapping distinguisher, or "fix-rate" for
            a state recovery attack on Gimli-Cipher.
    """
    cons = PresentConstraints(nrounds=n_double_rounds)

    for r in range(n_double_rounds):
        for i in range(4):
            cons.add_cell(r, w=1)

    for r in range(n_double_rounds - 1):
        if r % 2 == 0:
            # small swap
            cons.add_edge_2(r, 0, 1, 1. / 3.)
            cons.add_edge_2(r, 1, 0, 1. / 3.)
            cons.add_edge_2(r, 2, 3, 1. / 3.)
            cons.add_edge_2(r, 3, 2, 1. / 3.)
        else:
            # big swap
            cons.add_edge_2(r, 0, 2, 1. / 3.)
            cons.add_edge_2(r, 2, 0, 1. / 3.)
            cons.add_edge_2(r, 1, 3, 1. / 3.)
            cons.add_edge_2(r, 3, 1, 1. / 3.)
        for i in range(4):
            cons.add_edge_2(r, i, i, 2. / 3.)

    r = n_double_rounds - 1
    if flag == "full-wrapping":
        # do the last round
        if r % 2 == 0:
            # small swap
            cons.add_edge_2(r, 0, 1, 1. / 3.)
            cons.add_edge_2(r, 1, 0, 1. / 3.)
            cons.add_edge_2(r, 2, 3, 1. / 3.)
            cons.add_edge_2(r, 3, 2, 1. / 3.)
        else:
            # big swap
            cons.add_edge_2(r, 0, 2, 1. / 3.)
            cons.add_edge_2(r, 2, 0, 1. / 3.)
            cons.add_edge_2(r, 1, 3, 1. / 3.)
            cons.add_edge_2(r, 3, 1, 1. / 3.)
        for i in range(4):
            cons.add_edge_2(r, i, i, 2. / 3.)

        # merge cells 2 by 2
        for r in range(n_double_rounds):
            for i in range(2):
                cons.merge_cells_2(r, [(2 * i + k) % 4 for k in range(2)],
                                   merge_edges=True)

    elif flag == "fix-rate":
        for i in range(4):
            n = cons.add_edge_2(n_double_rounds - 1, i, i, 1. / 3.)
            cons.set_global(n)

    return cons


_HELP = """
Usage : python3 gimli.py attack

Demonstrates some attacks. Parameters (number of rounds...) are in the script.
attack:
- state : state-recovery
- wrapping : wrapping distinguisher

All examples here are classical.
"""

if __name__ == "__main__":
    import sys

    argc = len(sys.argv)
    if argc < 2:
        print(_HELP)
        sys.exit(0)
    attack = sys.argv[1]

    computation_model = CLASSICAL_COMPUTATION
    optimize_with_mem = True
    rounds_without_global_cons = []
    cut_forward, cut_backward = [], []
    backward_hint, forward_hint = [], []
    backward_zero, forward_zero = [], []
    covered_round = None

    if attack == "state":
        n_double_rounds = 6
        cons = make_gimli_constraints(n_double_rounds=n_double_rounds,
                                      flag="fix-rate")
        solution_flag = ALL_SOLUTIONS

    elif attack == "wrapping":
        n_double_rounds = 11
        covered_round = 6
        cons = make_gimli_constraints(n_double_rounds=n_double_rounds,
                                      flag="full-wrapping")
        solution_flag = SINGLE_SOLUTION

    else:
        raise ValueError(attack)

    cell_var_covered, global_lincons = find_mitm_attack(
        cons,
        flag=solution_flag,
        computation_model=computation_model,
        optimize_with_mem=optimize_with_mem,
        cut_forward=cut_forward,
        cut_backward=cut_backward,
        backward_hint=backward_hint,
        forward_hint=forward_hint,
        backward_zero=backward_zero,
        forward_zero=forward_zero,
        setting=PRESENT_SETTING,
        covered_round=covered_round)

    #=====================================
    # picture conversion. Not supported in the distributed code.
    try:
        from tikz_util import convert_to_present_pic
        TIKZ_MODULE_IMPORTED = True
    except ImportError:
        # means that the tikz_util module does not exist
        TIKZ_MODULE_IMPORTED = False

    if TIKZ_MODULE_IMPORTED:
        str_pic = convert_to_present_pic(cons,
                                         cell_var_covered,
                                         global_lincons,
                                         flag="gimli",
                                         display_cell_names=True)

        print(str_pic)
