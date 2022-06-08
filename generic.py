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

See "generic4.py" for an implementation of a 4-list strategy. We did not
implement larger strategies.

The main function defined below works for the three settings defined in the paper:
- Present-like setting (the basic setting)
- AES-like setting (matching through MixColumns enabled)
- Extended setting (XOR and branching cells enabled)

The computation model is either classical, or quantum (QRAQM); see the paper. It
finds either one or all the solutions of the specified path.
The basic constraints are wrapping constraints. IO constraints are also
available (e.g., in the Present / Spongent examples), but in a somewhat restricted
manner.

The most technical part of the implementation is the definition of "global
reduction" variables for each cell in the AES-like setting, which can be found
around line 400 of this file. This implementation differs from the view of 
"global edges" with which we started in the paper. In the Present-like case
it simply aggregates for each cell the total amount of "global edges" that this
cell belongs to, and in the AES-like case it includes the reduction through MC.
"""

from pyscipopt import Model, quicksum
import math

#========================================

FORWARD = "forward"
BACKWARD = "backward"
MERGED = "merged"

CLASSICAL_COMPUTATION = "classical"
QUANTUM_COMPUTATION = "quantum"
SINGLE_SOLUTION = "single-solution"
ALL_SOLUTIONS = "all-solutions"

AES_SETTING = "aes"
PRESENT_SETTING = "present"
EXTENDED_SETTING = "extended"
# used inside the optimization
EPSILON = 0.01


def find_mitm_attack(present_constraints,
                     time_target=None,
                     flag=SINGLE_SOLUTION,
                     computation_model=CLASSICAL_COMPUTATION,
                     setting=PRESENT_SETTING,
                     optimize_with_mem=True,
                     memory_limit=None,
                     cut_forward=[],
                     cut_backward=[],
                     backward_hint=[],
                     forward_hint=[],
                     backward_zero=[],
                     forward_zero=[],
                     covered_round=None,
                     verb=True):
    """
    Finds the best complexity of a 2-list merging MITM attack as specified in the
    paper. Arguments:
    
    - time_target -- set only if we want to reach a particular time complexity
    - computation_model -- either CLASSICAL_COMPUTATION or QUANTUM_COMPUTATION (QRAQM)
    - optimize_with_mem -- True if we include the memory in optimization goal
    - setting -- one of the 3 flags AES_SETTING, PRESENT_SETTING, EXTENDED_SETTING
    - flag -- either SINGLE_SOLUTION or ALL_SOLUTIONS

    The combination of ALL_SOLUTIONS and QUANTUM_COMPUTATION will yield an error (in the
    quantum setting, we study single-solution cases).

    - cut_forward -- rounds at which the forward list has no cells. If not specified,
      it will be searched.
    - cut_backward -- same for backward list. If not specified, it will be searched as well.
    - covered_round -- a round that will be colored. If not specified, it will also be searched.
    - forward_hint, backward_hint, forward_zero, backward_zero: set manually the coloring
            of some cells. This can greatly help reduce the solving time if we have an
            idea of what the path should look like.

    Returns a dictionary of cell colorings, and of global linear constraints.
    Note that these global constraints are actually recomputed from the internal
    "global_reduction" variables attached to each cell, and in the AES case, they
    do not include reduction through MC.
    """

    # take the data from the present-like constraints
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
        # if AES-like case, check that all cells have same width 1
        for c in cells:
            if cells[c] != 1:
                raise ValueError("Unsupported: cells must have width 1")
        if global_fixed:
            print(
                "Warning: IO constraints in AES setting may not work correctly"
            )

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
    if computation_model == QUANTUM_COMPUTATION:
        generic_time_one = 0.5 * (sum([
            linear_constraints[s][2]
            for s in linear_constraints_by_round[nrounds - 1]
        ]))
    else:
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

    labels = [FORWARD, BACKWARD, MERGED]

    # main variables: the colored cells. These variables MUST be boolean
    cell_var_colored = {}
    for l in labels:
        cell_var_colored[l] = {}
        for c in cells:
            cell_var_colored[l][c] = m.addVar(vtype="B")

    # ======= global reduction variables: reduction by global constraints and /
    # or through MC (in AES case). Without matching through MC, we simply count
    # the amount of links to bwd cells for each fwd cells, which can be used as
    # global constraints.
    global_reduction = m.addVar(vtype="C", lb=0)

    # global reduction for each cell
    # this variable counts either: the amount of global edges incoming (i.e. edges
    # backward -> forward), or: the amount of matching through MC for a cell
    # that does not belong to backward or forward, in the AES case
    global_reduction_vars = {}
    for c in cells:
        global_reduction_vars[c] = m.addVar(vtype="C", lb=0, ub=cells[c])

    # these lines concern the input-output case, where we have specified
    # global constraints. In that case, to make things simple, we simply force
    # pairs of cells of global constraints to be respectively in the forward
    # and backward lists. Then this constraint will be automatically satisfied.

    cells_in_global_cons = set()
    # all cells that belong to some global constraint
    for s in global_fixed:
        #        missed_global_cons[s] = m.addVar(vtype="C", lb=0)
        c1, c2, w = linear_constraints[s]
        cells_in_global_cons.add(c1)
        cells_in_global_cons.add(c2)
        # since the link c1 - c2 is a global constraint, c1 should not be active in
        # forward list, and c2 should not be active in backward
        m.addCons(cell_var_colored[FORWARD][c1] == 0)
        m.addCons(cell_var_colored[BACKWARD][c2] == 0)
        m.addCons(cell_var_colored[FORWARD][c2] == 1)
        m.addCons(cell_var_colored[BACKWARD][c1] == 1)

    fixed_additional = m.addVar(vtype="C", lb=0)
    # additional constraints that are not from the path
    m.addCons(fixed_additional == global_reduction -
              sum([linear_constraints[s][2] for s in global_fixed]))

    # number of times we repeat the merging
    repetitions = m.addVar(vtype="C", lb=0)
    m.addCons(repetitions <= fixed_additional)

    # number of solutions found by solving the MITM problem with the current path
    # (including additional constraints and repetitions)
    number_of_solutions = m.addVar(vtype="C", lb=0)
    m.addCons(number_of_solutions == path_solutions - fixed_additional +
              repetitions)

    # generic time complexity
    generic_time_total = m.addVar(vtype="C", lb=0)
    m.addCons(generic_time_total == number_of_solutions + generic_time_one)
    # we are always looking for a time complexity below the generic

    #======= constraints on list sizes, time comp, memory comp & objective function
    max_list_size = m.addVar(vtype="C", lb=0)
    list_sizes = {}
    for label in labels:
        list_sizes[label] = m.addVar(vtype="C", lb=0)
        m.addCons(max_list_size >= list_sizes[label])

    time_comp = m.addVar(vtype="C", lb=0)
    memory_comp = m.addVar(vtype="C", lb=0)

    # memory comp = min(list size forward, list size backward)
    switch = m.addVar(vtype="B")
    m.addCons(memory_comp >= list_sizes[FORWARD] - 100 * switch)
    m.addCons(memory_comp >= list_sizes[BACKWARD] - 100 * (1 - switch))

    # additional constraints due to the type of solution we want
    if flag == SINGLE_SOLUTION:
        # search for the smallest time comp to obtain a single solution from the path
        m.addCons(number_of_solutions == 0)
    elif flag == ALL_SOLUTIONS:
        # search for the smallest time comp to obtain all solutions from the path
        m.addCons(number_of_solutions == path_solutions)

    if computation_model == CLASSICAL_COMPUTATION:
        # classical setting: repetition loop + merging time
        m.addCons(time_comp >= max_list_size + repetitions)
    elif computation_model == QUANTUM_COMPUTATION:
        m.addCons(time_comp >= 0.5 * repetitions + 0.5 * max_list_size)
        m.addCons(time_comp >= 0.5 * repetitions + memory_comp)

    # definition of the objective
    if not optimize_with_mem:
        # time only
        m.setObjective(time_comp, sense="minimize")
        m.addCons(repetitions == 0)
    else:
        # time first, and for a given time, find the minimal memory
        # this will be much more costly than time alone
        m.setObjective(1000 * (time_comp) + memory_comp, sense="minimize")

    if memory_limit is not None:
        m.addCons(memory_comp <= memory_limit)

    #======= constraints to simplify the path
    # "hints"
    for c in backward_hint:
        if c in cell_var_colored[BACKWARD]:
            m.addCons(cell_var_colored[BACKWARD][c] == 1)

    for c in forward_hint:
        if c in cell_var_colored[FORWARD]:
            m.addCons(cell_var_colored[FORWARD][c] == 1)

    for c in backward_zero:
        if c in cell_var_colored[BACKWARD]:
            m.addCons(cell_var_colored[BACKWARD][c] == 0)

    for c in forward_zero:
        if c in cell_var_colored[FORWARD]:
            m.addCons(cell_var_colored[FORWARD][c] == 0)

    cells_in_global_cons = set()
    for s in global_fixed:
        c1, c2, w = linear_constraints[s]
        cells_in_global_cons.add(c1)
        cells_in_global_cons.add(c2)
    # constraints by symmetry: if two cells have the same fwd and bwd graph,
    # then we can exchange them. The first cell is preferably put on forward
    # and the second on backward (ordering is by cell names).
    count = 0
    for r in range(nrounds):
        cells_tmp = present_constraints.cell_names_by_round[r]
        pairs_tmp = []
        for c1 in cells_tmp:
            for c2 in cells_tmp:
                if c1 != c2 and c1 not in cells_in_global_cons and c2 not in cells_in_global_cons:
                    if (present_constraints.fwd_graph[c1]
                            == present_constraints.fwd_graph[c2]
                            and present_constraints.bwd_graph[c1]
                            == present_constraints.bwd_graph[c2]):
                        pairs_tmp.append(sorted([c1, c2]))
        count += len(pairs_tmp)
        # then order
        for t in pairs_tmp:
            m.addCons(cell_var_colored[FORWARD][t[0]] >=
                      cell_var_colored[FORWARD][t[1]])
            m.addCons(cell_var_colored[BACKWARD][t[0]] <=
                      cell_var_colored[BACKWARD][t[1]])
    print("============= ")
    print("Detected", count, "symmetric pairs of cells")
    print("============= ")

    # target for the time complexity
    if time_target is not None:
        m.addCons(time_comp == time_target)

    #====== cut rounds and colored round
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
            m.addCons(cell_var_colored[FORWARD][c] <= 1 - cut_fwd_rounds[r])
            m.addCons(cell_var_colored[BACKWARD][c] <= 1 - cut_bwd_rounds[r])

    # we can set the cut rounds manually
    if cut_fwd != []:
        for r in range(nrounds):
            m.addCons(cut_fwd_rounds[r] == (1 if r in cut_fwd else 0))
    if cut_bwd != []:
        for r in range(nrounds):
            m.addCons(cut_bwd_rounds[r] == (1 if r in cut_bwd else 0))

    # in merged list, one round in the middle must be completely colored
    if covered_round is not None:
        for c in cells_by_round[covered_round]:
            m.addCons(cell_var_colored[MERGED][c] == 1)
    else:
        l = []
        for r in possible_middle_rounds:
            tmpb = m.addVar(vtype="B")
            for c in cells_by_round[r]:
                m.addCons(tmpb <= cell_var_colored[MERGED][c])
            l.append(tmpb)
        m.addCons(quicksum(l) >= 1)

    #======================================
    if setting in [AES_SETTING, PRESENT_SETTING]:
        # simplification of cells, but not in extended case
        for c in cells:
            m.addCons(cell_var_colored[FORWARD][c] +
                      cell_var_colored[BACKWARD][c] <= 1)

    if setting in [PRESENT_SETTING, EXTENDED_SETTING]:
        # no new cells in the merged list: this is not true in AES-like mode
        for c in cells:
            m.addCons(
                cell_var_colored[MERGED][c] <= cell_var_colored[FORWARD][c] +
                cell_var_colored[BACKWARD][c])

    # In AES setting, constraints on new cells of the merged list: no such
    # cells at two successive rounds
    if setting == AES_SETTING:
        hasnewmgd = {}
        for r in range(nrounds):
            hasnewmgd[r] = m.addVar(vtype="B")
            for c in cells_by_round[r]:
                m.addCons(hasnewmgd[r] >= cell_var_colored[MERGED][c] -
                          cell_var_colored[BACKWARD][c] -
                          cell_var_colored[FORWARD][c])
        for r in range(nrounds):
            m.addCons(hasnewmgd[r] + hasnewmgd[(r + 1) % nrounds] <= 1)

    for c in cells:
        col_fwd_next = m.addVar(vtype="C")
        col_bwd_next = m.addVar(vtype="C")
        col_fwd_prev = m.addVar(vtype="C")
        col_bwd_prev = m.addVar(vtype="C")
        # total weight of edges between cell c and cells that belong to the forward
        # list at the next round
        m.addCons(col_fwd_next == quicksum([
            related_cells_atnextr[c][cc] * cell_var_colored[FORWARD][cc]
            for cc in related_cells_atnextr[c]
        ]))
        # total weight of edges between cell c and cells that belong to the backward
        # list at the next round
        m.addCons(col_bwd_next == quicksum([
            related_cells_atnextr[c][cc] * cell_var_colored[BACKWARD][cc]
            for cc in related_cells_atnextr[c]
        ]))
        # total weight of edges between cell c and cells that belong to the forward
        # list at the previous round
        m.addCons(col_fwd_prev == quicksum([
            related_cells_atprevr[c][cc] * cell_var_colored[FORWARD][cc]
            for cc in related_cells_atprevr[c]
        ]))
        # total weight of edges between cell c and cells that belong to the backward
        # list at the previous round
        m.addCons(col_bwd_prev == quicksum([
            related_cells_atprevr[c][cc] * cell_var_colored[BACKWARD][cc]
            for cc in related_cells_atprevr[c]
        ]))

        if setting == AES_SETTING:
            # our implementation of "global reduction" variables for each cell.
            # These constraints are an optimization of the following:
            # - if the cell belongs to the merged list, but neither the backwards
            # nor the forwards, then the global reduction can be up to:
            # min( col_bwd_prev, col_fwd_next, col_fwd + col_bwd - 1 )
            #   (this corresponds to the reduction of memory through MC)
            # - if the cell belongs to the forward list, we have a global reduction up to col_bwd_prev
            #   (this corresponds to all the edges that we can set "global")
            # - if the cell belongs to the backward list, we don't have any global reduction
            #   (it would have already been counted in the other cases)

            # In particular this implies:
            # - global_reduction[c] <= col_bwd_prev
            # - global_reduction[c] <= cell_var_colored[MERGED][c] - cell_var_colored[BACKWARD][c]
            # (which also includes the constraint
            #   cell_var_colored[MERGED][c] >= cell_var_colored[BACKWARD][c])

            m.addCons(
                cell_var_colored[MERGED][c] >= cell_var_colored[FORWARD][c])
            m.addCons(global_reduction_vars[c] <= cell_var_colored[MERGED][c] -
                      cell_var_colored[BACKWARD][c])
            m.addCons(
                global_reduction_vars[c] <= cell_var_colored[FORWARD][c] +
                cell_var_colored[BACKWARD][c] - cell_var_colored[MERGED][c] +
                col_fwd_next + col_bwd_next + col_fwd_prev + col_bwd_prev)

            # always up to col_bwd_prev
            m.addCons(global_reduction_vars[c] <= col_bwd_prev)

            # if not fwd, then up to col_fwd_next
            m.addCons(
                global_reduction_vars[c] <= cell_var_colored[FORWARD][c] +
                col_fwd_next)
        else:
            # if col in forward, then reduce up to col_bwd_prev: this corresponds
            # to global constraints that we can enforce between fwd and bwd cells
            m.addCons(global_reduction_vars[c] <= cell_var_colored[FORWARD][c])
            m.addCons(global_reduction_vars[c] <= col_bwd_prev)

    m.addCons(
        global_reduction == quicksum([global_reduction_vars[c]
                                      for c in cells]))
    #=============================

    #============ computation of list sizes by summing cell contributions.
    cell_contrib = {}
    for label in [FORWARD, BACKWARD, MERGED]:
        cell_contrib[label] = {}
        for c in cells:
            # contribution of cell.
            # if we go backwards, minimum is = width - sum(width of edges forwards)
            # if we go forwards, minimum is = width - sum(width of edges backwards)
            # this is to take into account XOR and branching cells in the extended case
            # Note that the direction for merge does not matter, so we take the backward
            # direction in this code, but we could go forwards equivalently.
            if label == BACKWARD or label == MERGED:
                lower_bound = min(
                    0, cells[c] - present_constraints.fwd_edges_width(c))
            elif label == FORWARD:
                lower_bound = min(
                    0, cells[c] - present_constraints.bwd_edges_width(c))
            # So, in Present or AES case, this lower_bound is 0 anyway

            nextorprev = (quicksum([
                related_cells_atnextr[c][cc] * cell_var_colored[label][cc]
                for cc in related_cells_atnextr[c]
            ]) if label == BACKWARD or label == MERGED else quicksum([
                related_cells_atprevr[c][cc] * cell_var_colored[label][cc]
                for cc in related_cells_atprevr[c]
            ]))

            # basic contribution of cell
            cell_contrib[label][c] = m.addVar(vtype="C",
                                              lb=lower_bound,
                                              ub=cells[c])
            m.addCons(cell_contrib[label][c] >= cell_var_colored[label][c] *
                      cells[c] - nextorprev)
            # in the extended setting, branching cells can have a negative contribution.
            # But we must ensure that they are in the list.
            # Otherwise (Present and AES) we don't need this constraint
            if setting == EXTENDED_SETTING:
                m.addCons(cell_contrib[label][c] >= -100 *
                          (cell_var_colored[label][c]))

        m.addCons(list_sizes[label] >=
                  quicksum([cell_contrib[label][c]
                            for c in cells]) - global_reduction)


#=====================================================================

    m.optimize()

    #================= Interpret the results of the optimizer

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

    # From the global reduction variables, find the global linear constraints
    # between pairs of cells.
    # In the AES-like case,
    # the reductions through MC cannot always be turned into global linear constraints
    # between pairs of cells.
    global_lincons = {}
    for s in linear_constraints:
        global_lincons[s] = 0

    global_reduction_vars = {
        c: m.getVal(global_reduction_vars[c])
        for c in cells
    }
    # set the global constraints
    for r in range(nrounds):
        for c in cells_by_round[r]:
            tmp = global_reduction_vars[c]
            if cell_var_colored[FORWARD][c] and tmp > EPSILON:
                # set global constraints (randomly on the edges that go backwards
                # from the current cell)
                for s in linear_constraints_by_round[(r - 1) % nrounds]:
                    c1, c2, w = tuple(linear_constraints[s])
                    if c2 == c and (
                            cell_var_colored[BACKWARD][c1]) and tmp > EPSILON:
                        global_lincons[s] = round(min(tmp, w) / w, 4)
                        tmp -= w

    if verb:
        # The output here is only indicative; in particular in the AES setting
        # not all global constraints are correctly taken into account when
        # printing the contribution of each cell.
        for label in labels:
            print("-----------", label)
            list_cells = [
                c for c in cells if (cell_var_colored[label][c]) > 0.5
            ]
            print("   List size: ", m.getVal(list_sizes[label]))
            print("   Cells: ", list_cells)
            for c in cells:
                cell_contrib[label][c] = round(
                    m.getVal(cell_contrib[label][c]), 9)

            print(
                "    Recomputed list size: ",
                sum([cell_contrib[label][c]
                     for c in list_cells]) - m.getVal(global_reduction))
            print("   Contributions (without global reduction): ")
            for c in list_cells:
                print(c, cell_contrib[label][c])

        print(
            "----------- Additional reductions (either through MC, or global constraints)"
        )
        for c in cells:
            if global_reduction_vars[c] > EPSILON:
                print(c, global_reduction_vars[c])

        print("----------- Global linear constraints in the solution path")
        print(
            sum([
                linear_constraints[s][2] for s in global_lincons
                if global_lincons[s] > EPSILON
            ]))
        result_lincons = [
            s for s in global_lincons if global_lincons[s] > EPSILON
        ]
        for s in result_lincons:
            print(linear_constraints[s], global_lincons[s])

    return cell_var_colored, global_lincons
