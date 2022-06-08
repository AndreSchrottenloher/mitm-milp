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
Generic MITM solver for finding two-list merging-based MITM attacks.
It uses the SCIP solver and its python interface pyscipopt.

This 4-list solver can only be used for "present" and "extended" settings,
and supports only classical computations. We used it for Feistel-like examples.
"""

from pyscipopt import Model, quicksum
import math

#========================================

CLASSICAL_COMPUTATION = "classical"
QUANTUM_COMPUTATION = "quantum"
SINGLE_SOLUTION = "single-solution"
ALL_SOLUTIONS = "all-solutions"

AES_SETTING = "aes"
PRESENT_SETTING = "present"
EXTENDED_SETTING = "extended"

EPSILON = 0.01


def find_mitm_attack(present_constraints,
                     time_target=None,
                     flag="single-solution",
                     computation_model="classical",
                     setting="present",
                     optimize_with_mem=True,
                     cut_forward=[],
                     cut_backward=[],
                     backward_hint=[],
                     forward_hint=[],
                     backward_zero=[],
                     forward_zero=[],
                     covered_round=None):
    """
    Finds the best complexity of a 4-list merging MITM attack as specified in the
    paper, only in Present and Extended settings.
    
    Returns a dictionary of cell colorings, and of global linear constraints.
    """

    nrounds = present_constraints.nrounds
    (cells, cells_by_round, linear_constraints, linear_constraints_by_round,
     global_fixed) = present_constraints.get_data()

    related_cells_atnextr = present_constraints.fwd_graph
    related_cells_atprevr = present_constraints.bwd_graph

    #=======================
    # Checks on the parameters

    if computation_model not in [CLASSICAL_COMPUTATION, QUANTUM_COMPUTATION]:
        raise ValueError("Invalid computation model flag: " +
                         str(computation_model))
    if flag not in [SINGLE_SOLUTION, ALL_SOLUTIONS]:
        raise ValueError("Invalid flag: " + str(flag))

    if setting == AES_SETTING:
        raise ValueError("AES setting unsupported")
    if computation_model == QUANTUM_COMPUTATION:
        raise ValueError("Quantum computations unsupported")

    state_size = present_constraints.state_size()
    possible_middle_rounds = present_constraints.possible_middle_rounds()
    # check that the colored round corresponds to a complete state

    # check that the "cut forward" and "cut backward" values are OK
    if cut_forward != []:
        for r in cut_forward:
            if r < 0 or r >= nrounds:
                raise ValueError("Bad cut-forward value")
    if cut_backward != []:
        for r in cut_backward:
            if r < 0 or r >= nrounds:
                raise ValueError("Bad cut-backward value")

    # sort the lists of cut rounds forwards and backwards
    cut_fwd = sorted(
        cut_forward)  # the first cut fwd round is first in increasing order
    cut_bwd = sorted(cut_backward)  # the first cut bwd round is last
    cut_bwd.reverse()

    #================================
    # generic time to find a solution: amount of wrapping constraint between input and output
    generic_time_one = (sum([
        linear_constraints[s][2]
        for s in linear_constraints_by_round[nrounds - 1]
    ]))

    # number of solutions in the path
    path_solutions = (
        sum([cells[c] for c in cells]) -
        sum([linear_constraints[s][2] for s in linear_constraints]) -
        sum([linear_constraints[s][2] for s in global_fixed]))

    #==============================
    # now we start to define the model
    m = Model("Linear_merging")

    labels = ["f1", "b1", "f2", "b2", "m1", "m2", "m"]

    # these variables MUST be boolean
    cell_var_colored = {}
    for l in labels:
        cell_var_colored[l] = {}
        for c in cells:
            cell_var_colored[l][c] = m.addVar(vtype="B")

    #===================
    # automatic simplification of dummy cells in Feistel-like permutations
    _count = 0
    for c in cells:
        # list of fwd links
        fwdlist = list(related_cells_atnextr[c])
        bwdlist = list(related_cells_atprevr[c])
        if (len(fwdlist) == 1 and len(bwdlist) == 1
                and cells[c] == related_cells_atnextr[c][fwdlist[0]]
                and cells[c] == related_cells_atprevr[c][bwdlist[0]]):
            # width of cell equal to unique edge fwd and bwd:
            # set colored variables equal to the cell above
            _count += 1
            for l in labels:
                m.addCons(
                    cell_var_colored[l][c] == cell_var_colored[l][bwdlist[0]])
    print("==== Simplified:", _count, "dummy cells ==== ")
    #====================

    # Alernative implementation of global linear constraints in the Present-like
    # case, using variables for each linear constraint.
    global_lincons = {}
    for s in linear_constraints:
        global_lincons[s] = m.addVar(vtype="C", lb=0, ub=1)

    for s in global_fixed:
        m.addCons(global_lincons[s] == 1)

    # reduction of global constraints
    global_cons_reduction = m.addVar(vtype="C", lb=0)

    m.addCons(global_cons_reduction == quicksum([
        global_lincons[s] * linear_constraints[s][2]
        for s in linear_constraints
    ]))

    fixed_additional = m.addVar(vtype="C", lb=0)
    # additional constraints that are not from the path
    m.addCons(fixed_additional == global_cons_reduction -
              sum([linear_constraints[s][2] for s in global_fixed]))

    # number of times we repeat the merging
    repetitions = m.addVar(vtype="C", lb=0)
    m.addCons(repetitions <= fixed_additional)

    # number of solutions found by solving the MITM problem with the current path
    # (including additional constraints and repetitions)
    number_of_solutions = m.addVar(vtype="C", lb=0)
    m.addCons(number_of_solutions == path_solutions - fixed_additional +
              repetitions)

    # repeating only improves the memory
    if not optimize_with_mem:
        # in particular, if path_solutions = 0 then this imposes number_of_solutions = -fixed_additional
        # thus fixed_additional = 0 and there are no global lincons
        m.addCons(repetitions == 0)

    # generic time complexity
    generic_time_total = m.addVar(vtype="C", lb=0)
    m.addCons(generic_time_total == number_of_solutions + generic_time_one)
    # we are always looking for a time complexity below the generic

    max_list_size = m.addVar(vtype="C", lb=0)
    list_sizes = {}
    for label in labels:
        list_sizes[label] = m.addVar(vtype="C", lb=0)
        m.addCons(max_list_size >= list_sizes[label])

    time_comp = m.addVar(vtype="C", lb=0)
    memory_comp = m.addVar(vtype="C", lb=0)

    if time_target is not None:
        m.addCons(time_comp == time_target)

    # memory comp
    switch = m.addVar(vtype="B")
    m.addCons(memory_comp >= list_sizes["f1"] - 100 * switch)
    m.addCons(memory_comp >= list_sizes["b1"] - 100 * (1 - switch))

    switch = m.addVar(vtype="B")
    m.addCons(memory_comp >= list_sizes["f2"] - 100 * switch)
    m.addCons(memory_comp >= list_sizes["b2"] - 100 * (1 - switch))

    switch = m.addVar(vtype="B")
    m.addCons(memory_comp >= list_sizes["m1"] - 100 * switch)
    m.addCons(memory_comp >= list_sizes["m2"] - 100 * (1 - switch))

    if flag == SINGLE_SOLUTION:
        # search for the smallest time comp to obtain a single solution from the path
        m.addCons(number_of_solutions == 0)
    elif flag == ALL_SOLUTIONS:
        # search for the smallest time comp to obtain all solutions from the path
        m.addCons(number_of_solutions == path_solutions)

    if computation_model == CLASSICAL_COMPUTATION:
        # classical setting: repetition loop + merging time
        m.addCons(time_comp >= max_list_size + repetitions)
    else:
        raise ValueError(
            "An error should already have been raised at this point")

    if not optimize_with_mem:
        m.setObjective(time_comp, sense="minimize")
    else:
        # time first, and for a given time, find the minimal memory
        m.setObjective(1000 * (time_comp) + memory_comp, sense="minimize")

    #=============================================
    # variables that say if a round is cut
    cut_fwd_rounds = {}
    cut_bwd_rounds = {}
    for r in range(nrounds):
        cut_bwd_rounds[r] = m.addVar(vtype="B")
        cut_fwd_rounds[r] = m.addVar(vtype="B")
    m.addCons(quicksum([cut_fwd_rounds[r] for r in range(nrounds)]) >= 1)
    m.addCons(quicksum([cut_bwd_rounds[r] for r in range(nrounds)]) >= 1)

    # no cell var colored at the cut round(s)
    for r in range(nrounds):
        for c in cells_by_round[r]:
            m.addCons(cell_var_colored["f1"][c] <= 1 - cut_fwd_rounds[r])
            m.addCons(cell_var_colored["b1"][c] <= 1 - cut_bwd_rounds[r])
            m.addCons(cell_var_colored["f2"][c] <= 1 - cut_fwd_rounds[r])
            m.addCons(cell_var_colored["b2"][c] <= 1 - cut_bwd_rounds[r])

    # we can set the cut rounds manually
    if cut_fwd != []:
        for r in range(nrounds):
            m.addCons(cut_fwd_rounds[r] == (1 if r in cut_fwd else 0))
    if cut_bwd != []:
        for r in range(nrounds):
            m.addCons(cut_bwd_rounds[r] == (1 if r in cut_bwd else 0))

    #================

    # no shared cells between both lists, but only in present setting
    if setting == PRESENT_SETTING:
        for c in cells:
            m.addCons(
                cell_var_colored["f1"][c] + cell_var_colored["b1"][c] +
                cell_var_colored["b2"][c] + cell_var_colored["f2"][c] <= 1)
            m.addCons(
                cell_var_colored["m1"][c] + cell_var_colored["m2"][c] <= 1)

    #===========
    for s in linear_constraints:
        c1, c2, w = tuple(linear_constraints[s])
        m.addCons(cell_var_colored["f1"][c1] + cell_var_colored["f2"][c1] +
                  global_lincons[s] <= 1)
        m.addCons(cell_var_colored["b1"][c2] + cell_var_colored["b2"][c2] +
                  global_lincons[s] <= 1)
        m.addCons(cell_var_colored["f1"][c1] + cell_var_colored["f2"][c1] +
                  cell_var_colored["b1"][c1] +
                  cell_var_colored["b2"][c1] >= global_lincons[s])

    # variables that give the reduction from global linear constraints that
    # we have in each list.
    # There is such a reduction as long as one of the cells is in the list.
    global_lincons_active = {}
    for label in labels:
        global_lincons_active[label] = {}
        for s in linear_constraints:
            global_lincons_active[label][s] = m.addVar(vtype="C", lb=0)
            m.addCons(global_lincons_active[label][s] <= global_lincons[s] *
                      linear_constraints[s][2])
            m.addCons(global_lincons_active[label][s] <=
                      cell_var_colored[label][linear_constraints[s][0]] +
                      cell_var_colored[label][linear_constraints[s][1]])

    cell_contrib = {}
    for label in labels:
        cell_contrib[label] = {}
        for r in cells_by_round:
            for c in cells_by_round[r]:
                # contribution of cell. Maximum is the width of this cell.
                cell_contribution = m.addVar(vtype="C", lb=0, ub=cells[c])

                nextorprev = (quicksum([
                    related_cells_atnextr[c][cc] * cell_var_colored[label][cc]
                    for cc in related_cells_atnextr[c]
                ]) if label in ["b1", "b2"] else quicksum([
                    related_cells_atprevr[c][cc] * cell_var_colored[label][cc]
                    for cc in related_cells_atprevr[c]
                ]))

                m.addCons(cell_contribution >= cell_var_colored[label][c] *
                          cells[c] - nextorprev)
                cell_contrib[label][c] = cell_contribution

        m.addCons(
            list_sizes[label] >=
            quicksum([cell_contrib[label][c] for c in cells]) - quicksum(
                [global_lincons_active[label][s] for s in linear_constraints]))

    for c in cells:
        m.addCons(cell_var_colored["m1"][c] <= cell_var_colored["f1"][c] +
                  cell_var_colored["b1"][c])
        m.addCons(cell_var_colored["m2"][c] <= cell_var_colored["f2"][c] +
                  cell_var_colored["b2"][c])
        m.addCons(cell_var_colored["m"][c] <= cell_var_colored["m1"][c] +
                  cell_var_colored["m2"][c])
        # unnecessary
        # however, without these constraints there can be less cells
        # in m1 than in the union of f1 and b1, which can seem strange


#        m.addCons( cell_var_colored["m1"][c] >= cell_var_colored["f1"][c] )
#        m.addCons( cell_var_colored["m1"][c] >= cell_var_colored["b1"][c] )
#        m.addCons( cell_var_colored["m2"][c] >= cell_var_colored["f2"][c] )
#        m.addCons( cell_var_colored["m2"][c] >= cell_var_colored["b2"][c] )

# in merged list, one round in the middle must be completely colored
    if covered_round is not None:
        for c in cells_by_round[covered_round]:
            m.addCons(cell_var_colored["m"][c] == 1)
    else:
        l = []
        for r in possible_middle_rounds:
            tmpb = m.addVar(vtype="B")
            for c in cells_by_round[r]:
                m.addCons(tmpb <= cell_var_colored["m"][c])
            l.append(tmpb)
        m.addCons(quicksum(l) >= 1)
    # the linear constraints of one round in the middle must be completely colored

    #=====================================================================

    m.optimize()

    print("Max list size: ", m.getVal(max_list_size))
    print("Memory comp:", m.getVal(memory_comp))
    print("Solutions of the path:", path_solutions)
    print("Fixed additional constraints:", m.getVal(fixed_additional))
    print("Repetitions:", m.getVal(repetitions))
    print("Generic time one:", generic_time_one)
    print("Generic time total:", m.getVal(generic_time_total))
    print("Time complexity:", m.getVal(time_comp))
    print("Number of solutions:", m.getVal(number_of_solutions))

    if cut_fwd == []:
        # cut forward rounds found by the model
        cut_fwd = [
            r for r in range(nrounds) if m.getVal(cut_fwd_rounds[r]) > 0.5
        ]

    if cut_bwd == []:
        # cut backward rounds found by the model
        cut_bwd = [
            r for r in range(nrounds) if m.getVal(cut_bwd_rounds[r]) > 0.5
        ]

    for label in cell_var_colored:
        for c in cells:
            # 0 or 1
            cell_var_colored[label][c] = int(
                round(m.getVal(cell_var_colored[label][c]), 5))
    for s in global_lincons:
        global_lincons[s] = m.getVal(global_lincons[s])

    for label in labels:
        print("-----------", label)
        list_cells = [c for c in cells if (cell_var_colored[label][c]) > 0.5]
        print("   List size: ", m.getVal(list_sizes[label]))
        print("   Cells: ", list_cells)

        print("   Contributions (without global reduction): ")
        for c in list_cells:
            cell_contrib[label][c] = m.getVal(cell_contrib[label][c])
            print(c, cell_contrib[label][c])

    print("----------- Global lincons")
    print(
        sum([
            linear_constraints[s][2] for s in global_lincons
            if global_lincons[s] > EPSILON
        ]))
    result_lincons = [s for s in global_lincons if global_lincons[s] > EPSILON]
    for s in result_lincons:
        print(linear_constraints[s], global_lincons[s])

    cell_var_colored["forward"] = {c: 0 for c in cells}
    cell_var_colored["backward"] = {c: 0 for c in cells}
    cell_var_colored["merged"] = {c: 0 for c in cells}

    return cell_var_colored, global_lincons
