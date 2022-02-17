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
Demonstrations of some attacks on Simpira variants (which, more generally,
apply to GFN-based permutations).
"""

from simpira_util import *


def simpira_attack(b=3, nrounds=None, tikz=False, expand=True):
    """
    Demonstrates an attack on small Simpira, with a wrapping of the first branch
    on itself.

    b: number of branches
    nrounds: number of rounds to attack. If None, will simply demonstrate the
        attack with the maximal number of rounds that we obtained.

    tikz: tikz output
    expand: if False, will not attempt to expand the system
    """
    # number of rounds that we manage to attack
    b_to_nrounds = {
        2: 5,
        3: 11,
        4: 9,
        6: 9,
        8: 9,
        5: 12,
        7: 21,
    }
    if b not in [2, 3, 4, 6, 8]:
        print("Total nbr of double-F: ", nbr_of_df(b))
        print("Nbr of double-F:", b_to_nrounds[b])

    _nrounds = nrounds if nrounds is not None else b_to_nrounds[b]
    print("=== b=", b, " === Attacking", _nrounds, "rounds")
    eq_system = SimpiraEquationSystem(b=b, nrounds=_nrounds)
    eq_system.wrap(0, 0)

    if tikz:
        print(eq_system.tikz)

    print("=== Not simplified:")
    print(eq_system)
    eq_system.simplify(verb=False)
    print("=== Simplified:")
    print(eq_system)
    print(eq_system.variables)
    if eq_system.is_trivial():
        print("=== Simplified system is trivial, no MILP needed")
        solutions = list(eq_system.variables)
    else:
        if expand:
            eq_system.expand()
            print("=== Expanded:")
            print(eq_system)
        #     now use gad
        solutions = gad_solving(eq_system, nb_steps=10, goal=b - 1)
        # there are many alternative solutions. Here are examples that we took in the paper:
        #    b_to_solutions = {
        #       6 : [8, 13, 14, 16, 18],
        #       3 : [4, 6],
        #       2 : [2],
        #       4 : [11,14,16],
        #       8 : []
        #    }
    if len(solutions) >= b:
        # There was no valid GAD strategy
        raise Exception("Failed!")

    # Re-create an Equation system object to compute actual values
    eq_full = SimpiraEquationSystem(b=b, nrounds=_nrounds)
    # expand in order to deduce properly
    eq_full.wrap(0, 0)
    eq_full.expand()
    # take a guess: 0 for all the states that we want to guess
    d = {i: [0] * 16 for i in solutions}
    x = eq_full.solve_gad(d)
    # print result
    print(simpira_state_to_str(x))
    simpira(x, nrounds=_nrounds)
    print(simpira_state_to_str(x))
    if b not in [2, 3, 4, 6, 8]:
        print("Total nbr of double-F: ", nbr_of_df(b))
        print("Nbr of double-F:", b_to_nrounds[b])


def larger_attack(b):
    """
    Demonstration of the attack on (slightly less than) 2/3 of the rounds of larger Simpira versions.
    """
    nbr_df = nbr_of_df(b)
    # we do 2/3 of this, minus 1
    # then it's trivial (no MILP required)
    if b % 2 == 0:
        nrounds = int(2 / 3 * nbr_df) - 1
    else:
        nrounds = int(2 / 3 * nbr_df) - 2

    print(b, nbr_df, nbr_df / 3, nrounds)
    eq_system = SimpiraEquationSystem(b=b, nrounds=nrounds)
    eq_system.wrap(0, 0)
    eq_system.simplify(verb=False)
    if eq_system.eqs != []:
        raise Exception("failure!")


if __name__ == "__main__":

    simpira_attack(b=8, nrounds=9, expand=True)
