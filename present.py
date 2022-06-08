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
Several use cases of our generic MILP method for
finding MITM attacks, on Present-like designs.

"""

from generic import find_mitm_attack, PRESENT_SETTING, SINGLE_SOLUTION, ALL_SOLUTIONS, CLASSICAL_COMPUTATION, QUANTUM_COMPUTATION
from util import PresentConstraints
import math


def make_present_constraints(nrounds=6,
                             width=16,
                             structure_flag="full-wrapping",
                             pairwise=False):
    """
    Creates constraints for Present or Spongent.
    
    Parameters:
    width -- number of cells. Actual Present has 16 cells. Spongent has more.
    structure_flag -- "full-wrapping" (as usual), "single-sbox" with a fixed S-Box at
        position 0 in input and output, or "single-sbox-13" for the same constraint,
        but fixed at position 13
    pairwise -- set to True in order to merge some cells pairwise in the middle rounds
        (it reduces the solving time for our 8-round attack on Present, but we might
        end up losing some solutions)
    """

    cons = PresentConstraints(nrounds=nrounds)

    b = 4 * width

    def next_bit(i):
        if i == b - 1:
            return b - 1
        else:
            return (i * (b // 4)) % (b - 1)

    def next_boxes(j):
        # next boxes of S-Box number j
        res = []
        for i in range(4):
            bit = j * 4 + i
            res.append((next_bit(bit) // 4))
        return res

    def previous_boxes(j):
        res = []
        for k in range(width):
            if j in next_boxes(k):
                res.append(k)
        return res

    for r in range(nrounds):
        for i in range(width):
            cons.add_cell(r, w=1)

    for r in range(nrounds if structure_flag == "full-wrapping" else nrounds -
                   1):
        # connect the wires between S-Boxes at round r and r+1
        # bit i of the state is moved to i * (b/4) mod (b-1)
        # and b-1 if i = b-1 , where b is the bit-size of the state
        # thus there are b linear constraints of 1 bit each between 2 cells
        for i in range(b):
            # linear cons connecting bit i to next_bit(i)
            sb1 = (i // 4)
            sb2 = (next_bit(i) // 4)
            cons.add_edge(cons.get_cell_name(r, sb1),
                          cons.get_cell_name((r + 1) % nrounds, sb2),
                          w=0.25)

    if pairwise:
        # merge the cells pairwise in middle rounds
        diff_steps = math.ceil(math.log(width, 4) + 0.1)
        for r in range(diff_steps, nrounds - diff_steps):
            for i in range(width // 2):
                cons.merge_cells_2(r, [(2 * i + k) % width for k in range(2)],
                                   merge_edges=True)

    if structure_flag == "full-wrapping":
        pass

    elif structure_flag == "single-sbox":
        edge_name = cons.add_edge(cons.get_cell_name(nrounds - 1, 0),
                                  cons.get_cell_name(0, 0),
                                  w=1)
        cons.set_global(edge_name)

    elif structure_flag == "single-sbox-13":
        edge_name = cons.add_edge(cons.get_cell_name(nrounds - 1, 13),
                                  cons.get_cell_name(0, 13),
                                  w=1)
        cons.set_global(edge_name)

    else:
        raise ValueError("Invalid structure flag: " + str(flag))

    cons.simplify()
    return cons


_HELP = """
Usage : python3 present.py attack version

Demonstrates some attacks. Parameters (number of rounds...) are in the script.
attack:
- present7 : 7-round mitm on present
- present8 : 8-round mitm on present
- spongent : mitm on spongent

If spongent is selected, a second parameter is expected which gives the 
size in cells: it must belong to the list:
22, 34, 44, 60, 66, 68, 84, 96, 120, 168, 192. Default is 22.

All examples here are classical.
"""

if __name__ == "__main__":
    import sys

    argc = len(sys.argv)
    if argc < 2:
        print(_HELP)
        sys.exit(0)
    attack = sys.argv[1]

    if attack == "spongent":
        if argc == 2:
            width = 22
        else:
            width = int(sys.argv[2])
            if width not in [22, 34, 44, 60, 66, 68, 84, 96, 120, 168, 192]:
                raise ValueError(
                    "Invalid width: " + str(width) +
                    ". The valid widths are: 22, 34, 44, 60, 66, 68, 84, 96, 120, 168, 192"
                )
    else:
        if argc == 3:
            raise ValueError("Unexpected parameter 'width' for this example")

    computation_model = CLASSICAL_COMPUTATION
    generic_flag = ALL_SOLUTIONS
    time_target = None
    cut_forward = []
    cut_backward = []
    forward_hint = []
    optimize_with_mem = True
    covered_round = None
    verb = True

    if attack == "toy":
        # normal result: 2.75, list size = 2.25
        nrounds, width = 4, 4
        structure_flag = "full-wrapping"
        present_cons = make_present_constraints(nrounds=nrounds,
                                                width=width,
                                                structure_flag=structure_flag)

    elif attack == "present7":
        nrounds, width = 8, 16
        structure_flag = "single-sbox-13"
        present_cons = make_present_constraints(nrounds=nrounds,
                                                width=width,
                                                structure_flag=structure_flag)

        cut_forward = [nrounds - 1]
        cut_backward = [0]
        covered_round = ((nrounds - 1) // 2)
        optimize_with_mem = True

    elif attack == "present8":
        nrounds, width = 9, 16
        structure_flag = "single-sbox-13"
        present_cons = make_present_constraints(nrounds=nrounds,
                                                width=width,
                                                structure_flag=structure_flag,
                                                pairwise=True)
        cut_forward = [nrounds - 1]
        cut_backward = [0]
        covered_round = ((nrounds - 1) // 2)
        optimize_with_mem = True

    elif attack == "spongent":
        # 22, 34, 44, 60, 66, 68, 84, 96, 120, 168, 192
        # in cells
        #        width = 22

        # max nbr. of rounds where solutions of minimal time complexity are found
        # (see paper)
        # The inexistence of solutions for larger number of rounds is not proven,
        # so we might have missed some.
        new_nrounds = {
            16: 7,  # 20 seconds
            22: 8,  # 8 rds max
            34: 8,  # 8 rds max
            40: 9,  # 9 rds max
            44: 9,  # 9 rds max
            60: 10,  # 10 rds max
            66: 10,  # 10 rds max, 11 rds infeasible with these constraints
            68: 10,  # 10 rds max
            84: 10,  # 10 rds max
            96: 11,  # 11 rds max
            120: 12,  # 11 rds in <= 100 s
            168: 12,  # 11 rds in <= 100 s
            192: 12  # <= 500s
        }

        nrounds = new_nrounds[width] + 1
        time_target = width - 2 if width != 22 else width - 1.5
        optimize_with_mem = False
        print("Nbr of rounds: ", new_nrounds[width])

        structure_flag = "single-sbox"
        present_cons = make_present_constraints(nrounds=nrounds,
                                                width=width,
                                                structure_flag=structure_flag)

        # during the first and last diffusion steps, only a single list represented
        diff_steps = math.ceil(math.log(width, 4))
        forward_hint = sum([
            list(present_cons.cell_names_by_round[i])
            for i in range(diff_steps)
        ], [])
        cut_forward = [nrounds - 1,
                       nrounds - 2]  # nrounds-i for i in range(1, diff_steps)]
        covered_round = ((nrounds - 1) // 2)

        verb = False

    else:
        raise ValueError("")

    func = find_mitm_attack
    cell_var_covered, global_lincons = func(
        present_cons,
        computation_model=computation_model,
        flag=generic_flag,
        optimize_with_mem=optimize_with_mem,
        setting=PRESENT_SETTING,
        time_target=time_target,
        covered_round=covered_round,
        cut_forward=cut_forward,
        cut_backward=cut_backward,
        verb=verb)

    #========================
    # picture conversion. Not supported in the distributed code.
    try:
        from tikz_util import convert_to_present_pic
        TIKZ_MODULE_IMPORTED = True
    except ImportError:
        # means that the tikz_util module does not exist
        TIKZ_MODULE_IMPORTED = False

    if TIKZ_MODULE_IMPORTED:
        str_pic = convert_to_present_pic(present_cons,
                                         cell_var_covered,
                                         global_lincons,
                                         flag="present",
                                         display_cell_names=True,
                                         only_cells=True)
        print(str_pic)
