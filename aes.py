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
Several use cases of our generic MILP method for
finding MITM attacks, on AES-like designs.
"""
from generic import (find_mitm_attack, AES_SETTING, SINGLE_SOLUTION,
                     CLASSICAL_COMPUTATION, QUANTUM_COMPUTATION)
from util import PresentConstraints


def make_aes_constraints(d=4,
                         nrounds=5,
                         final_mc=False,
                         structure_flag="full-wrapping"):
    """
    Arguments:
    - d -- size of the AES state if it is a square
    - final_mc -- set to False if there is no final MixColumns (can only be used
            with d = 4)
    - structure_flag --
        full-wrapping: wrap the entire final state on the input (with or without final MC)
        half-wrapping: wrap only the first or second half of the state (depending on
                a flag set in the code)

    """
    if structure_flag not in [
            "full-wrapping", "half-wrapping", "half-fixed", "single-col-fixed",
            "single-col-wrapping"
    ]:
        raise ValueError("Wrong flag: " + str(structure_flag))

    real_nrounds = (nrounds - 1 if not final_mc else nrounds)
    cons = PresentConstraints(nrounds=real_nrounds)
    for r in range(real_nrounds):
        for i in range(d):
            cons.add_cell(r, w=1)

    for r in range(real_nrounds - 1):
        for i in range(d**2):
            cons.add_edge_2(r, i // d, i % d, w=1. / d)

    if final_mc:
        # if there is a final MC, we must connect the cells from rounds "nrounds-1" to 0
        r = nrounds - 1
        if structure_flag in ["full-wrapping"]:
            for i in range(d**2):
                cons.add_edge_2(nrounds - 1, i // d, i % d, w=1. / d)

        elif structure_flag in ["half-wrapping", "half-fixed"]:
            # we wrap only one half: what corresponds to a couple cells in round 0
            first_half = False
            for i in (range(0, d**2 //
                            2) if first_half else range(d**2 // 2, d**2)):
                n = cons.add_edge_2(r, i % d, i // d, w=1. / d)
                if structure_flag == "half-fixed":
                    cons.set_global(n)

        elif structure_flag in ["single-col-fixed", "single-col-wrapping"]:
            for i in range(0, d):
                n = cons.add_edge_2(r, i % d, i // d, w=1. / d)
                if structure_flag == "single-col-fixed":
                    cons.set_global(n)

    else:
        # if there is no final MC, we must connect the cells from rounds "nrounds-2" to 0 directly
        r = nrounds - 2
        if structure_flag == "full-wrapping":
            if d != 4:
                raise Exception("Unsupported")
            cons.add_edge_2(r, 0, 0, 0.5)
            cons.add_edge_2(r, 0, 2, 0.5)
            cons.add_edge_2(r, 2, 0, 0.5)
            cons.add_edge_2(r, 2, 2, 0.5)
            cons.add_edge_2(r, 1, 3, 0.5)
            cons.add_edge_2(r, 1, 1, 0.5)
            cons.add_edge_2(r, 3, 3, 0.5)
            cons.add_edge_2(r, 3, 1, 0.5)
        elif structure_flag == "single-col-wrapping":
            if d != 4:
                raise Exception("Unsupported")
            cons.add_edge_2(r, 0, 0, 0.5)
            cons.add_edge_2(r, 0, 2, 0.5)

        else:
            raise Exception("Unsupported")
    cons.simplify()

    return cons


def make_grostl512_constraints(nrounds=5, merge=None):
    """
    In grostl-512, the state is a 8*16 rectangle (16 cells), and instead of
    0,1,2,3,4,5,6,7, the shifts are 0,1,2,3,4,5,6,11.
    """
    cons = PresentConstraints(nrounds=nrounds)
    d = 16
    edge_width = 1. / 8. if merge is None else 1. / 8. / merge
    cell_width = 1 if merge is None else 1 / merge
    for r in range(nrounds):
        for i in range(d):
            cons.add_cell(r, w=cell_width)

    # due to the shifts, cell i at round r+1 has links with
    # i, i+1, i+2, i+3, i+4, i+5, i+6, i+11 at round r
    for r in range(nrounds - 1):
        for i in range(d):
            for s in [0, 1, 2, 3, 4, 5, 6, 11]:
                cons.add_edge_2(r, (i + s) % d, i, w=edge_width)
    # then half wrapping
    r = nrounds - 1
    for i in range(d // 2, d):
        for s in [0, 1, 2, 3, 4, 5, 6, 11]:
            cons.add_edge_2(r, (i + s) % d, i, w=edge_width)
    # merge
    if merge:
        for r in range(nrounds):
            for i in range(d // merge):
                cons.merge_cells_2(r, [merge * i + j for j in range(merge)],
                                   merge_edges=True)
    cons.simplify()
    return cons


def make_saturnin_constraints(nrounds=5, merge=None):
    cons = PresentConstraints(nrounds=nrounds)  # half-rounds actually
    # cells = columns of the cube.
    # columns = constant (x,z)
    # nibble (x,y,z) in the cube is mapped to y + 4x + 16z
    # SR slice: (x,y,z) -> (x+y mod4, y, z)
    # SR sheet: (x,y,z) -> (x,y, z+y mod z)

    cell_w = 1. / merge if merge else 1
    edge_w = 0.25 / merge if merge else 0.25

    for r in range(nrounds):  # two MC layers by round
        for i in range(16):
            cons.add_cell(r, w=cell_w)
    # column: (x,z) -> x + 4*z

    for r in range(nrounds):
        if r % 4 == 0:
            # next round is going to have SRslice
            # (x,y,z) goes to  (x+y mod4, y, z)
            # connect nibble (x,y,z) to (x+y mod4, y, z)
            # connect column (x,z) to all (x',z)
            for x in range(4):
                for z in range(4):
                    for i in range(4):
                        cons.add_edge_2(r, x + 4 * z, i + 4 * z, w=edge_w)
        elif r % 4 == 1:
            # do SRslice inverse: same connection
            for x in range(4):
                for z in range(4):
                    for i in range(4):
                        cons.add_edge_2(r, x + 4 * z, i + 4 * z, w=edge_w)
        elif r % 4 == 2:
            # SRsheets: connect column (x,z) to all (x,z')
            for x in range(4):
                for z in range(4):
                    for i in range(4):
                        cons.add_edge_2(r, x + 4 * z, x + 4 * i, w=edge_w)
        elif r % 4 == 3:
            # SRsheets inv: connect column (x,z) to all (x,z')
            for x in range(4):
                for z in range(4):
                    for i in range(4):
                        cons.add_edge_2(r, x + 4 * z, x + 4 * i, w=edge_w)

    if merge:
        for r in range(nrounds):
            for i in range(16 // merge):
                cons.merge_cells_2(r, [merge * i + j for j in range(merge)],
                                   merge_edges=True)
    cons.simplify()
    return cons


def haraka256_mix(i):
    # 4 goes to 1
    colperm = [0, 2, 4, 6, 1, 3, 5, 7]
    return colperm[i]


def haraka256_invmix(i):
    colperm = [0, 4, 1, 5, 2, 6, 3, 7]
    return colperm[i]


def haraka512_mix(i):
    colperm = [3, 11, 7, 15, 8, 0, 12, 4, 9, 1, 13, 5, 2, 10, 6, 14]
    return colperm.index(i)


def haraka512_invmix(i):
    colperm = [3, 11, 7, 15, 8, 0, 12, 4, 9, 1, 13, 5, 2, 10, 6, 14]
    return colperm[i]


def make_haraka256_constraints(nrounds=5):
    """
    nrounds: number of AES rounds (corresponds to half-rounds in Haraka)
    """
    # two AES states updated in parallel, but every two rounds, there is a
    # MIX operation
    # there is also a final MC
    # the cells are supposed to represent columns at the beginning of each round

    cons = PresentConstraints(nrounds=nrounds)
    for r in range(nrounds):
        for i in range(8):
            cons.add_cell(r, w=1)

    for r in range(nrounds):
        # connect round r and r +1
        if r % 2 == 0:
            # no MIX: two parallel AES rounds
            for i in range(16):
                cons.add_edge_2(r, i // 4, i % 4, w=0.25)
                cons.add_edge_2(r, (i // 4) + 4, (i % 4) + 4, w=0.25)
        else:
            for i in range(16):
                cons.add_edge_2(r, (i // 4), haraka256_mix(i % 4), w=0.25)
                cons.add_edge_2(r, ((i // 4) + 4),
                                haraka256_mix((i % 4) + 4),
                                w=0.25)

    return cons


def make_haraka512_constraints(nrounds=5, flag="partial-wrapping"):
    """
    nrounds: number of AES rounds (corresponds to half-rounds in Haraka)

    Flag: "partial-wrapping" for the standard Haraka-512 feedforward
    "partial-io" for the same, but with IO constraints instead of wrapping.
    "sponge-wrapping" for a feedforward which extracts the 2 first AES states
      instead of the columns specified by Haraka (this is the sponge-based use
      proposed in SPHINCS+)
    "sponge-io" for the same with IO constraints.
    """
    # as in haraka-256, every two rounds, there is a MIX
    # there is also a final MC
    # flag: partial-wrapping for the Haraka v2 original proposal
    # (half of the state, corresponding to columns
    # 2,3,6,7,8,9,12,13 of the next state
    if flag not in [
            "partial-wrapping", "partial-io", "sponge-wrapping", "sponge-io",
            "sponge-io-2"
    ]:
        raise ValueError("Invalid flag: " + str(flag))

    cons = PresentConstraints(nrounds=nrounds)
    for r in range(nrounds):
        for i in range(16):
            # 16 columns
            cons.add_cell(r, w=1)

    for r in range(nrounds - 1):
        # connect round r and r +1
        if r % 2 == 0:
            # no MIX: four parallel AES rounds
            for i in range(16):
                for j in range(4):
                    cons.add_edge_2(r, i // 4 + 4 * j, (i % 4) + 4 * j, w=0.25)
        else:
            for i in range(16):
                for j in range(4):
                    cons.add_edge_2(r, (i // 4) + 4 * j,
                                    haraka512_mix(i % 4 + 4 * j),
                                    w=0.25)

    # final round
    if flag == "partial-wrapping" or flag == "partial-io":
        wrapping_columns = [2, 3, 6, 7, 8, 9, 12, 13]
    else:
        wrapping_columns = [0, 1, 2, 3, 4, 5, 6, 7]
    r = nrounds - 1
    if r % 2 == 0:
        # no MIX: four parallel AES rounds
        for i in range(16):
            for j in range(4):
                if flag == "sponge-io-2":
                    if (i % 4) + 4 * j in [8, 9, 10, 11, 12, 13, 14, 15]:
                        # capacity constraint
                        # output rate - input capacity
                        n = cons.add_edge_2(r,
                                            i // 4 + 4 * j - 8,
                                            (i % 4) + 4 * j,
                                            w=0.25)
                        cons.set_global(n)
                else:
                    if (i % 4) + 4 * j in wrapping_columns:
                        n = cons.add_edge_2(r,
                                            i // 4 + 4 * j, (i % 4) + 4 * j,
                                            w=0.25)
                        if flag == "partial-io" or flag == "sponge-io":
                            cons.set_global(n)
    else:
        for i in range(16):
            for j in range(4):
                if haraka512_mix((i % 4) + 4 * j) in wrapping_columns:
                    n = cons.add_edge_2(r, (i // 4) + 4 * j,
                                        haraka512_mix(i % 4 + 4 * j),
                                        w=0.25)
                    if flag == "partial-io" or flag == "sponge-io":
                        cons.set_global(n)
    cons.simplify()
    return cons


_HELP = """
Usage : python3 aes.py attack computation_model

Demonstrates some attacks. Parameters (number of rounds...) are in the script.
attack:
- aes : aes permutation, full wrapping, last MC omitted
- haraka256 : haraka-256 v2 attack
- haraka512 : haraka-512 v2 attack
- grostl256 : grostl-256 ot attack
- grostl512 : grostl-512 ot attack
- haraka-sponge : haraka-512 in sponge mode

If computation_model not given, default is classical.
computation_model :
- classical
- quantum
"""

if __name__ == "__main__":
    import sys

    argc = len(sys.argv)
    if argc < 2:
        print(_HELP)
        sys.exit(0)
    attack = sys.argv[1]

    if argc == 2:
        computation_model = CLASSICAL_COMPUTATION
    else:
        computation_model = sys.argv[2]
    if computation_model not in [CLASSICAL_COMPUTATION, QUANTUM_COMPUTATION]:
        raise ValueError("Invalid computation model: " +
                         str(computation_model))

    covered_round = None
    time_target = None
    cut_forward, cut_backward = [], []
    backward_hint, forward_hint = [], []
    backward_zero, forward_zero = [], []
    d = 4  # state size parameter for AES-like square states
    optimize_with_mem = True
    generic_flag = SINGLE_SOLUTION

    if attack == "test":
        nrounds, d, final_mc, structure_flag = 6, 4, True, "full-wrapping"
        cons = make_aes_constraints(d=4,
                                    nrounds=nrounds,
                                    final_mc=True,
                                    structure_flag=structure_flag)

    elif attack == "aes":
        nrounds, d, final_mc, structure_flag = 7, 4, False, "full-wrapping"
        cons = make_aes_constraints(d=d,
                                    nrounds=nrounds,
                                    final_mc=final_mc,
                                    structure_flag=structure_flag)

    elif attack == "grostl256":
        nrounds, d, final_mc, structure_flag = 6, 8, True, "half-wrapping"

        # should run in a few days without the covered round
        # finishes in less than 20 min with the covered round (without time target)
        covered_round = 1
        if computation_model == CLASSICAL_COMPUTATION:
            time_target = 3.5
        cons = make_aes_constraints(d=d,
                                    nrounds=nrounds,
                                    final_mc=final_mc,
                                    structure_flag=structure_flag)

    elif attack == "grostl512":
        nrounds = 8
        covered_round = 1
        cons = make_grostl512_constraints(nrounds=nrounds, merge=None)
        optimize_with_mem = True
        # these hints were obtained by running the optimization with merge = 4
        # (merging cells by groups of 4)
        if computation_model == CLASSICAL_COMPUTATION:
            cut_backward = [4, 5, 6]
            cut_forward = [0, 1, 6, 7]
            backward_hint = (
                ['x^0_%i' % i for i in range(8, 16)] +
                ['x^7_%i' % i for i in [12, 13, 14, 15]] +
                ['x^1_%i' % i for i in [4, 5, 6, 7, 8, 9, 10, 11]] +
                ['x^2_%i' % i for i in [0, 1, 2, 3, 4, 5, 6, 7]] +
                ['x^3_%i' % i for i in [0, 1, 2, 3]])
        elif computation_model == QUANTUM_COMPUTATION:
            cut_backward = [4, 5, 6]
            cut_forward = [0, 1, 6, 7]
            backward_hint = (
                ['x^0_%i' % i
                 for i in range(8, 16)] + ['x^7_%i' % i for i in [14, 15]] +
                ['x^1_%i' % i for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]] +
                ['x^2_%i' % i for i in [0, 1, 2, 3, 4, 5, 6, 7, 14, 15]] +
                ['x^3_%i' % i for i in [0, 1, 14, 15]])
            backward_zero = (['x^0_%i' % i for i in range(0, 8)] +
                             ['x^7_%i' % i for i in range(0, 14)] +
                             ['x^1_%i' % i for i in [0, 1, 12, 13, 14, 15]] +
                             ['x^2_%i' % i for i in [8, 9, 10, 11]] +
                             ['x^3_%i' % i for i in [2, 3, 4, 5, 6, 7, 8]])

    elif attack == "haraka-sponge":
        if computation_model == QUANTUM_COMPUTATION:
            nrounds = 7
        else:
            nrounds = 9
        cons = make_haraka512_constraints(nrounds=nrounds, flag="sponge-io")

#    elif attack == "saturnin":
#        nrounds = 12
#        computation_model = CLASSICAL_COMPUTATION

#        backward_hint = ['x^0_%i' for i in [0,4,8,12]] + ['x^11_%i' for i in [0,4,8,12]]
#        cons = make_saturnin_constraints(nrounds=nrounds, merge=None)
#        optimize_with_mem = True

    elif attack == "haraka256":
        # we'll obtain: 3.5 time and 1 memory (quantum)
        nrounds = 9
        cons = make_haraka256_constraints(nrounds=nrounds)
        time_target = 7 if computation_model == CLASSICAL_COMPUTATION else 3.5
        generic_flag = "single-solution"
        optimize_with_mem = True
        # some hints to make the computation faster: we expect a fully active state
        # here, some rounds will be cut, and a round in the middle will be covered
        backward_hint = ['x^0_4', 'x^0_5', 'x^0_6', 'x^0_7']
        backward_zero = ['x^0_0', 'x^0_1', 'x^0_2', 'x^0_3']
        cut_backward, cut_forward = [], [0, 1, 2, nrounds - 2, nrounds - 1]
        covered_round = 4

    elif attack == "haraka512":
        # we'll obtain: 7.5 and 0.5
        nrounds = 11
        cons = make_haraka512_constraints(nrounds=nrounds)
        # these hints are here to recover an attack similar to Bao et al.
        backward_zero = (['x^%i_%i' % (0, i) for i in range(16) if i >= 4] +
                         ['x^%i_%i' % (1, i) for i in range(16) if i >= 4])  #
        forward_zero = (['x^%i_%i' % (8, i) for i in range(16) if i >= 4] +
                        ['x^%i_%i' % (9, i) for i in range(16) if i >= 4])  #
        cut_backward, cut_forward = [
            nrounds - 1, nrounds - 2, nrounds - 3, nrounds - 4
        ], [0, 1, 2, 3]
        covered_round = 4

    else:
        raise ValueError("Invalid attack: " + str(attack))

    if True:
        cell_var_covered, global_lincons = find_mitm_attack(
            cons,
            time_target=time_target,
            flag=generic_flag,
            computation_model=computation_model,
            optimize_with_mem=optimize_with_mem,
            cut_forward=cut_forward,
            cut_backward=cut_backward,
            backward_hint=backward_hint,
            forward_hint=forward_hint,
            backward_zero=backward_zero,
            forward_zero=forward_zero,
            setting=AES_SETTING,
            covered_round=covered_round)

    #===============================================================
    # picture conversion. Not supported in the distributed code.
    try:
        from tikz_util import (convert_to_present_pic,
                               convert_to_haraka256_pic,
                               convert_to_haraka512_pic, convert_to_aes_pic,
                               convert_to_grostl512_pic)
        TIKZ_MODULE_IMPORTED = True
    except ImportError:
        # means that the tikz_util module does not exist
        TIKZ_MODULE_IMPORTED = False

    if TIKZ_MODULE_IMPORTED:
        # outputs directly to the console.
        if attack == "haraka-256v2":
            str_pic = convert_to_haraka256_pic(cons, cell_var_covered,
                                               global_lincons)
        elif attack in ["haraka-512v2", "haraka-sponge"]:
            str_pic = convert_to_present_pic(cons,
                                             cell_var_covered,
                                             global_lincons,
                                             flag="haraka512",
                                             cell_nbr=d,
                                             edge_nbr=d,
                                             display_cell_names=False,
                                             only_cells=False)
            print(str_pic)
            str_pic = convert_to_haraka512_pic(cons, cell_var_covered,
                                               global_lincons)
        elif attack == "grostl512-ot":
            str_pic = convert_to_grostl512_pic(cons, cell_var_covered,
                                               global_lincons)

#            print(str_pic)
#            str_pic = convert_to_present_pic(cons,
#                                             cell_var_covered,
#                                             global_lincons,
#                                             flag="grostl512",
#                                             cell_nbr=16, edge_nbr=8,
#                                             display_cell_names=True,
#                                             only_cells=True)

        elif attack == "saturnin":
            str_pic = convert_to_present_pic(cons,
                                             cell_var_covered,
                                             global_lincons,
                                             flag="aes",
                                             cell_nbr=4,
                                             edge_nbr=4,
                                             display_cell_names=True,
                                             only_cells=True)

        else:
            str_pic = convert_to_present_pic(cons,
                                             cell_var_covered,
                                             global_lincons,
                                             flag="aes",
                                             cell_nbr=d,
                                             edge_nbr=d,
                                             display_cell_names=True,
                                             only_cells=True)
        print(str_pic)
