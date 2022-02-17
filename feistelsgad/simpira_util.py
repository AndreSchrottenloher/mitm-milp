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
Generic representation of an equation system resulting from a MITM problem
on a GFN, and some methods to find a GAD strategy to solve the system in 
simple cases. These functions are used to demonstrate our attacks on Simpira
variants.

Usually the proper course of action will be:
- create the equation system object that corresponds to the design, and the
  wrapping constraint that we want to enforce (usually a single branch)
- simplify the system by removing useless variables
- if the system becomes trivial, we're done: a GAD strategy can solve it
- otherwise, expand the system by taking all non-trivial sums of equations
- then use our MILP program to find if there is a GAD strategy

"""

from copy import copy
from pyscipopt import Model, quicksum
from simpira_implementation import *

# used for tikz export
STANDALONE_HEADER = """
\\documentclass{standalone}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage{tikz}
\\usepackage{xcolor}
\\usepackage{../style/feistel}

\\colorlet{darkgreen}{green!50!black} \colorlet{darkblue}{blue!50!black}
\\colorlet{darkred}{red!50!black} \colorlet{darkorange}{orange!70!black}

"""


class SimpiraEquationSystem:
    """
    This class represents a system of equations in Simpira (but actually,
    any Feistel-like scheme that we consider without swapping branches).
    
    In particular, we can: write the equations, expand them (there are not
    many non-trivial equations that we can deduce, actually), or simplify
    them. Simplifying means that we will remove states that appear in a single
    equation. Sometimes this empties the system. Then the 
    remaining variables give immediately the states to guess, in order to deduce
    everything else, and there is no need for MILP.
    
    Otherwise we can give these equations to the GAD solver, based on MILP, 
    to find if there is a GAD procedure. And finally, we can deduce the 
    whole state given enough internal guesses.
    
    Internally an equation of the form:
    S1 + S2 = Pi_3(S4) + Pi_4(S5)
    is represented by two lists: "sum" = [1,2] and "pi" = [(3,4), (4,5)]
    """

    def __init__(self, b=4, nrounds=8):
        self.b = b
        self.eqs = []
        self.init_state = None
        self.final_state = None
        self.variables = set()
        self._init_tikz()
        if b == 3:
            self._init_3(nrounds=nrounds)
        elif b == 2:
            self._init_2(nrounds=nrounds)
        elif b == 4:
            self._init_4(nrounds=nrounds)
        elif b == 6:
            self._init_6(nrounds=nrounds)
        elif b == 8:
            self._init_8(nrounds=nrounds)
        else:
            self._init_gen(ndf=nrounds)
        self._finalize_tikz()

    def __getitem__(self, i):
        return self.eqs[i]

    def is_trivial(self):
        """
        Check if system has become trivial.
        """
        return self.eqs == []

    def has_eq(self, sum_set, pi_set, full=True):
        """
        Finds if an equation already exists.
        """
        tmp = self.eqs
        for e in tmp:
            if (e["sum"]) == (sum_set) and (e["pi"]) == (pi_set):
                return True
        return False

    def add_eq(self, sum_list, pi_list, full=True):
        """
        Adds a new equation to the system (e.g., obtained by a sum of two equations).
        """
        tmp = self.eqs
        sum_set = set(sum_list)
        pi_set = set(pi_list)
        for c in sum_set:
            # states
            assert type(c) == int
        for p in pi_set:
            # pairs (const, state)
            assert type(p) == tuple
            assert len(p) == 2
            assert type(p[0]) == int
            assert type(p[1]) == int
        tmp.append({"sum": sum_set, "pi": pi_set, "value": [0] * 16})
        for i in sum_set:
            self.variables.add(i)
        for p in pi_set:
            self.variables.add(p[1])

    def xor_eqs(self, e1, e2):
        """
        Returns the "sum" and "pi" lists of an equation obtained by XORing two
        equations of the system.
        """
        new_sum_set = copy(e1["sum"]) ^ copy(e2["sum"])
        new_pi_set = copy(e1["pi"]) ^ copy(e2["pi"])
        return new_sum_set, new_pi_set

    def replace(self, i, j):
        """
        Replaces state variable i by j in the system.
        """
        for e in self.eqs:
            if i in e["sum"]:
                e["sum"].remove(i)
                e["sum"].add(j)
            for p in e["pi"]:
                if p[1] == i:
                    e["pi"].remove(p)
                    e["pi"].add((p[0], j))
        self.variables.remove(i)

    def wrap(self, pos_init, pos_final):
        """
        Wrap position i of initial state on position j of final state: this replaces
        one state variable.
        """
        self.replace(self.final_state[pos_final], self.init_state[pos_init])

    def __str__(self):
        """
        Print this equation system in a LateX-ish way.
        """
        res = ""
        for e in self.eqs:
            if e["sum"] or e["pi"]:
                sum_list = " + ".join(["S_{%i}" % i for i in list(e["sum"])])
                if not e["sum"]:
                    sum_list = "0"
                pi_list = " + ".join(
                    ["\Pi_{%i}(S_{%i})" % p for p in list(e["pi"])])
                res += sum_list + " = " + pi_list + " \\\\\n"
        return res

    def print_eq(self, i):
        e = self.eqs[i]
        sum_list = " + ".join(["S_{%i}" % i for i in list(e["sum"])])
        if not e["sum"]:
            sum_list = "0"
        pi_list = " + ".join(["\Pi_{%i}(S_{%i})" % p for p in list(e["pi"])])
        res = sum_list + " = " + pi_list + " \\\\"
        print(res)

    #============================
    # Simplifying the system in a trivial way

    def find_var_with_single_eq(self, full=False):
        """
        Finds a state variable that appears only in a single equation, and can
        thus be removed from the system. It returns the variable and its
        corresponding equation.
        """
        tmp = self.eqs
        counts = {}
        for e in tmp:
            for i in e["sum"]:
                if i not in counts:
                    counts[i] = 0
                counts[i] += 1
            for p in e["pi"]:
                if p[1] not in counts:
                    counts[p[1]] = 0
                counts[p[1]] += 1
        var_single = None
        for c in counts:
            if counts[c] == 1:
                var_single = c
        eq_single = None
        for e in tmp:
            for i in e["sum"]:
                if i == var_single:
                    eq_single = e
            for p in e["pi"]:
                if p[1] == var_single:
                    eq_single = e
        return var_single, eq_single

    def find_eq_with_single_var(self):
        """
        Finds an equation that contains a single variable, if there is one, whose
        value can then be deduced.
        """
        for e in self.eqs:
            if len(e["sum"]) + len(e["pi"]) == 1:
                return e
        return None

    def simplify(self, verb=False):
        """
        Simplifies the system by removing all trivial equations and state variables.
        """
        simplified = True
        while simplified:
            simplified = False
            var_single, eq_single = self.find_var_with_single_eq()
            if var_single is not None:
                simplified = True
                if verb:
                    print("Removing ", var_single)
                self.variables.remove(var_single)
                self.eqs.remove(eq_single)
                if verb:
                    print("Removing equation: ", eq_single)

    def expand(self, full=False):
        """
        Expands the system by taking all possible sums between equations, and
        adding all new non-trivial equations obtained with such sums.
        """
        # toggle between the full system and the reduced system
        to_expand = True
        tmp = self.eqs
        while to_expand:
            to_expand = False
            pairs = []
            for e1 in tmp:
                for e2 in tmp:
                    if e2 != e1:
                        if (e1["sum"]).intersection((e2["sum"])):
                            pairs.append((e1, e2))
            # check all overlapping pairs
            for (e1, e2) in pairs:
                new_sum_set, new_pi_set = self.xor_eqs(e1, e2)
                if not self.has_eq(new_sum_set, new_pi_set, full=full):
                    to_expand = True
                    self.add_eq(new_sum_set, new_pi_set, full=full)

    def equations_for(self, v):
        res = []
        for i in range(len(self.eqs)):
            e = self.eqs[i]
            if v in list(e["sum"]) + [p[1] for p in e["pi"]]:
                res.append(i)
        return res

    #==========================
    # solving the system by simple GAD

    def replace_by_value(self, i, value):
        """
        Replaces a state variable by a given value in all the equations.
        """
        # replace var i by value v
        for e in self.eqs:
            if i in e["sum"]:
                e["sum"].remove(i)
                e["value"] = xor(e["value"], value)
            to_remove = None
            for p in e["pi"]:
                if p[1] == i:
                    to_remove = p
            if to_remove is not None:
                e["pi"].remove(to_remove)
                e["value"] = xor(e["value"], pi(value, to_remove[0], b=self.b))

    def solve_gad(self, d):
        """
        Given a dictionary of guessed state values, and assuming that everything
        can be deduced from them, performs the deduction automatically (so we
        don't have to write this down by hand). If the guesses are not sufficient,
        an error will be raised at some point.
        """
        resd = d
        for i in resd:
            self.replace_by_value(i, resd[i])

        # wait until we have completed the initial state
        while not (set(self.init_state)).issubset(set([i for i in resd])):
            e = self.find_eq_with_single_var()
            if e is None:
                print(self)
                raise Exception("cannot deduce here")
            if len(e["sum"]) == 1:
                v = list(e["sum"])[0]
                resd[v] = e["value"]
                self.replace_by_value(v, resd[v])
            elif len(e["pi"]) == 1:
                p = list(e["pi"])[0]
                v = p[1]
                resd[v] = invpi(e["value"], p[0], b=self.b)
                self.replace_by_value(v, resd[v])
            else:
                raise ValueError("not possible")

        return [resd[i] for i in self.init_state]

    #=========================
    # these functions initialize the Simpira equation system for a given variant.

    def _init_tikz(self):
        self.tikz = STANDALONE_HEADER + (
            """\\begin{document}\n \\begin{feistelpic}{%i}\n""" % self.b)
        self._occupied = [False] * self.b
        self.tikz += """\\begin{feistelrd}"""
        for j in range(self.b):
            self.tikz += """\\labelstart{%i}{$S_{%i}$}""" % (j, j)

    # xor b on a, going through Pi_i, and add a new name
    def _tikz_add_branch(self, a, b, i, new):
        newr = False
        for j in range(min(a, b), max(a, b) + 1):
            if self._occupied[j]:
                newr = True
            else:
                self._occupied[j] = True
        if newr:
            self._occupied = [False] * self.b
            for j in range(min(a, b), max(a, b) + 1):
                self._occupied[j] = True
            self.tikz += """\\end{feistelrd}\n"""
            self.tikz += """\\begin{feistelrd}"""

        self.tikz += """\\xorbranch{%i}{%i}{$\Pi_{%i}$}""" % (b, a, i)
        self.tikz += """\\labelend{%i}{$S_{%i}$}""" % (a, new)

    def _finalize_tikz(self):
        self.tikz += """\\end{feistelrd}\n"""
        self.tikz += """\\end{feistelpic}\n"""
        self.tikz += """\\end{document}\n%=============================\n"""

    def _init_2(self, nrounds=6):
        i = 1
        self.init_state = [0, 1]
        current_state = [0, 1]
        numbering = 2
        for r in range(nrounds):
            self._tikz_add_branch((r + 1) % 2, r % 2, i, numbering)
            self.add_eq([current_state[(r + 1) % 2], numbering],
                        [(i, current_state[r % 2])])
            current_state[(r + 1) % 2] = numbering
            numbering += 1
            i += 1
        self.final_state = copy(current_state)

    def _init_3(self, nrounds=6):
        i = 1
        self.init_state = [0, 1, 2]
        current_state = [0, 1, 2]
        numbering = 3
        for r in range(nrounds):
            self._tikz_add_branch((r + 1) % 3, r % 3, i, numbering)
            self.add_eq([current_state[(r + 1) % 3], numbering],
                        [(i, current_state[r % 3])])
            current_state[(r + 1) % 3] = numbering
            numbering += 1
            i += 1
        self.final_state = copy(current_state)

    def _init_4(self, nrounds=7):
        i = 1
        self.init_state = [0, 1, 2, 3]
        current_state = [0, 1, 2, 3]
        numbering = 4
        for r in range(nrounds):
            self._tikz_add_branch((r + 1) % 4, r % 4, i, numbering)
            self.add_eq([current_state[(r + 1) % 4], numbering],
                        [(i, current_state[r % 4])])
            current_state[(r + 1) % 4] = numbering
            numbering += 1
            i += 1
            self._tikz_add_branch((r + 3) % 4, (r + 2) % 4, i, numbering)
            self.add_eq([current_state[(r + 3) % 4], numbering],
                        [(i, current_state[(r + 2) % 4])])
            current_state[(r + 3) % 4] = numbering
            numbering += 1
            i += 1

        self.final_state = copy(current_state)

    def _init_6(self, nrounds=6):
        i = 1  # constant numbering
        current_state = [0, 1, 2, 3, 4, 5]  # starting state
        self.init_state = copy(current_state)
        numbering = 6
        s = [0, 1, 2, 5, 4, 3]
        for r in range(nrounds):
            for (a, b) in [(s[(r + 1) % 6], s[(r) % 6]),
                           (s[(r + 5) % 6], s[(r + 2) % 6]),
                           (s[(r + 3) % 6], s[(r + 4) % 6])]:
                # we XOR branch b on branch a
                self._tikz_add_branch(a, b, i, numbering)
                self.add_eq([current_state[a], numbering],
                            [(i, current_state[b])])
                current_state[a] = numbering
                numbering += 1
                i += 1

        self.final_state = copy(current_state)

    def _init_8(self, nrounds=6):
        i = 1
        current_state = [0, 1, 2, 3, 4, 5, 6, 7]
        self.init_state = copy(current_state)
        numbering = 8
        s = [0, 1, 6, 5, 4, 3]
        t = [2, 7]
        for r in range(nrounds):
            for (a, b) in [(s[(r + 1) % 6], s[(r) % 6]),
                           (s[(r + 5) % 6], t[(r) % 2]),
                           (s[(r + 3) % 6], s[(r + 4) % 6]),
                           (t[(r + 1) % 2], s[(r + 2) % 6])]:
                self._tikz_add_branch(a, b, i, numbering)
                self.add_eq([current_state[a], numbering],
                            [(i, current_state[b])])
                current_state[a] = numbering
                numbering += 1
                i += 1
        self.final_state = copy(current_state)

    def _init_gen(self, ndf=100):
        b = self.b
        i = 0
        current_state = [j for j in range(b)]
        self.init_state = copy(current_state)
        numbering = [b]

        def doubleF(r, i, numbering):
            if (r % 2):
                tmp1, tmp2 = (r % b), (r + 1) % b
                self._tikz_add_branch(tmp1, tmp2, 2 * i + 1, numbering[0])
                self.add_eq([current_state[tmp1], numbering[0]],
                            [(2 * i + 1, current_state[tmp2])])
                current_state[tmp1] = numbering[0]
                numbering[0] += 1
                tmp1, tmp2 = (r + 1) % b, (r) % b
                self._tikz_add_branch(tmp1, tmp2, 2 * i + 2, numbering[0])
                self.add_eq([current_state[tmp1], numbering[0]],
                            [(2 * i + 2, current_state[tmp2])])
                current_state[tmp1] = numbering[0]
                numbering[0] += 1
            else:
                tmp1, tmp2 = (r + 1) % b, (r) % b
                self._tikz_add_branch(tmp1, tmp2, 2 * i + 1, numbering[0])
                self.add_eq([current_state[tmp1], numbering[0]],
                            [(2 * i + 1, current_state[tmp2])])
                current_state[tmp1] = numbering[0]
                numbering[0] += 1
                tmp1, tmp2 = (r % b), (r + 1) % b
                self._tikz_add_branch(tmp1, tmp2, 2 * i + 2, numbering[0])
                self.add_eq([current_state[tmp1], numbering[0]],
                            [(2 * i + 2, current_state[tmp2])])
                current_state[tmp1] = numbering[0]
                numbering[0] += 1

        #==

        d = (b // 2) * 2

        for j in range(3):
            if d != b:
                doubleF(b - 2, i, numbering)
                i += 1
                if i >= ndf:
                    break
            for r in range(d - 1):
                doubleF(r, i, numbering)
                i += 1
                if i >= ndf:
                    break
                if (r != d - r - 2):
                    doubleF(d - r - 2, i, numbering)
                    i += 1
                if i >= ndf:
                    break
            if i >= ndf:
                break

            if d != b:
                doubleF(b - 2, i, numbering)
                i += 1
            if i >= ndf:
                break

        self.final_state = copy(current_state)


#===================================


def gad_solving(eq_system, nb_steps=5, goal=None):
    """
    Given a Simpira equation system, searches for a GAD strategy using MILP.
    The MILP program tries to minimize the number of variables that
    need to be guessed in order to deduce all the others in the system. These
    deductions are performed trivially, i.e., if we know all variables of an equation
    except one, then this variable is deduced at the next step. We have 
    to specify a certain number of steps.
    """
    m = Model("GAD")
    # find if there is a guessing strategy to deduce everything from these equations
    # in several steps
    nb_eqs = len(eq_system.eqs)
    deduced = {}
    for n in range(nb_steps):
        deduced[n] = {}
        for v in eq_system.variables:
            # 1 if variable is deduced at this step
            if n == 0:
                deduced[n][v] = m.addVar(vtype="B")
            else:
                deduced[n][v] = m.addVar(vtype="C", lb=0, ub=1)

    # active equations at this round (all vars but one are known)
    active_eqs = {}

    for n in range(nb_steps - 1):
        # deduced at next step <=> already known, OR:
        # there exists a single eq such that all vars are known except this one
        active_eqs[n] = {}
        for i in range(nb_eqs):
            active_eqs[n][i] = m.addVar(vtype="B")
            # active if at most one unknow var
            e = eq_system[i]
            all_eq_vars = set(list(e["sum"]) + [p[1] for p in e["pi"]])
            m.addCons((len(all_eq_vars) - 1) * active_eqs[n][i] <= quicksum(
                [deduced[n][v] for v in all_eq_vars]))

        for v in eq_system.variables:
            # since all equations are written down  + deduced[n][v] is not necessary (or does not seem)
            m.addCons(deduced[n + 1][v] <= deduced[n][v] + quicksum(
                [active_eqs[n][i] for i in eq_system.equations_for(v)]))

    # initial state must be deduced!
    for v in eq_system.variables:
        m.addCons(deduced[nb_steps - 1][v] == 1)

    m.setObjective(quicksum([deduced[0][v] for v in eq_system.variables]),
                   sense="minimize")
    if goal is not None:
        m.addCons(
            goal == quicksum([deduced[0][v] for v in eq_system.variables]))

    m.optimize()

    # then give the guesses
    for n in range(nb_steps):
        for v in eq_system.variables:
            deduced[n][v] = int(round(m.getVal(deduced[n][v]), 4))
    for n in range(nb_steps - 1):
        for i in range(nb_eqs):
            active_eqs[n][i] = int(round(m.getVal(active_eqs[n][i]), 4))

    result = [v for v in eq_system.variables if deduced[0][v] > 0.5]
    print(result)

    for n in range(nb_steps):
        print("----", n)
        print([v for v in eq_system.variables if deduced[n][v] > 0.5])
        if n < nb_steps - 1:
            for i in range(nb_eqs):
                if active_eqs[n][i] == 1 and (n == 0
                                              or active_eqs[n - 1][i] == 0):
                    eq_system.print_eq(i)  # (i, eq_system[i])
    return result


if __name__ == "__main__":

    # for MILP we want to work with a simplified system

    eqs = SimpiraEquationSystem(b=7, nrounds=21)
    eqs.wrap(0, 0)

    print("=== Not simplified:")
    print(eqs)
    eqs.simplify(verb=False)
    print("=== Simplified:")
    print(eqs)
    print(eqs.variables)
    print("=== Expanded:")
    eqs.expand()
    print(eqs)
