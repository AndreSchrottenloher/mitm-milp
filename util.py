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
Implementation of a cell-based representation of a Present-like design, as a
directed graph. The data structure supports the definition of cells and 
linear constraints between them.
"""


class PresentConstraints:
    """
    Each cell in the graph has a name, but it is also identified by its round,
    and the position it has at this round. Cells are named x^i_j where i is the
    round number and j the position. They have weights.

    Each linear constraint is an edge between two cells, which also has a weight.
    """

    def __init__(self, nrounds):
        self.merged_cells = {}
        self.cell_names_by_round = {}
        self.cell_round_pos_to_name = {}
        self.cell_name_to_data = {}  # rd, pos, width

        self.cell_name_to_fwd_edges_width = {}
        self.cell_name_to_bwd_edges_width = {}

        self.fwd_graph = {
        }  # for each cells, connected cells at the next round
        self.bwd_graph = {
        }  # for each cells, connected cells at the prev round

        self.fwd_edges = {}
        # for each cell c, edges between c and cells at the next round (going forward)
        self.bwd_edges = {}
        # for each cell c, edges between c and cells at the previous round (going backward)

        self._edge_numbering_helper = {}
        self._cell_pos_helper = {}
        # each edge has also a name
        self.edge_names_by_round = {}
        self.edge_name_to_data = {}  # c1, c2 (names), width

        self.individual_links_fwd = {
        }  # remember the individual edges between nodes
        self.cell_pos_storage = {}  # remember the position of cells
        # (before simplifying and merging)
        self.global_fixed = []
        for r in range(nrounds):
            self.cell_names_by_round[r] = []
            self.edge_names_by_round[r] = []
            self._cell_pos_helper[r] = 0
        self.nrounds = nrounds

    def add_cell(self, r, w, name=None):
        """
        Adds a cell of width w at round r.
        """
        pos = self._cell_pos_helper[r]
        if name is None:
            name = "x^%i_%i" % (r, pos)
        self.cell_names_by_round[r].append(name)
        self.cell_name_to_data[name] = (r, pos, w)
        self.cell_round_pos_to_name[(r, pos)] = name
        self.cell_pos_storage[name] = pos
        self._edge_numbering_helper[name] = {}
        self._cell_pos_helper[r] += 1
        self.fwd_edges[name] = []
        self.bwd_edges[name] = []
        self.fwd_graph[name] = {}
        self.bwd_graph[name] = {}

    def state_size(self):
        """
        Finds the state size of this design (maximal sum of widths of individual
        cells over all the rounds).
        """
        # state size for a round: sum of all cell widths for this round
        # the largest such sum defines the state size of the design
        res = 0
        for r in self.cell_names_by_round:
            tmp = sum([
                self.cell_name_to_data[c][2]
                for c in self.cell_names_by_round[r]
            ])
            res = max(tmp, res)
        return res

    def possible_middle_rounds(self):
        """
        Returns a list of rounds which have a size equal to the maximal state size.
        (Not all the rounds have the same size, only the "middle rounds" are complete,
        for ex. if the input-output conditions are enforced only on part of the state).
        """
        s = self.state_size()
        res = []
        for r in self.cell_names_by_round:
            if sum([
                    self.cell_name_to_data[c][2]
                    for c in self.cell_names_by_round[r]
            ]) == s:
                res.append(r)
        return res

    def fwd_edges_width(self, c):
        return sum([self.fwd_graph[c][cp] for cp in self.fwd_graph[c]])

    def bwd_edges_width(self, c):
        return sum([self.bwd_graph[c][cp] for cp in self.bwd_graph[c]])

    def remove_cell(self, name):
        """
        Removes a cell (by name) and all the edges to which it belongs.
        """
        rd, pos, width = self.cell_name_to_data[name]
        del self.cell_name_to_data[name]
        self.cell_names_by_round[rd].remove(name)
        del self.cell_round_pos_to_name[(rd, pos)]

        # update of the forward / backward graphs
        if name in self.fwd_graph:
            del self.fwd_graph[name]
        if name in self.bwd_graph:
            del self.bwd_graph[name]
        # remove this cell from the bwd graphs of nodes at next round
        for e in self.fwd_edges[name]:
            if e in self.edge_name_to_data:
                (c1, c2, w) = self.edge_name_to_data[e]
                if c1 in self.bwd_graph[c2]:
                    del self.bwd_graph[c2][c1]
        for e in self.bwd_edges[name]:
            if e in self.edge_name_to_data:
                (c1, c2, w) = self.edge_name_to_data[e]
                if c2 in self.fwd_graph[c1]:
                    del self.fwd_graph[c1][c2]

        edges_to_remove = self.fwd_edges[name] + self.bwd_edges[name]
        del self.fwd_edges[name]
        del self.bwd_edges[name]
        for e in edges_to_remove:
            for r in self.edge_names_by_round:
                if e in self.edge_names_by_round[r]:
                    self.edge_names_by_round[r].remove(e)
            if e in self.global_fixed:
                self.global_fixed.remove(e)
            if e in self.edge_name_to_data:
                del self.edge_name_to_data[e]

    def split_name(self, c):
        """
        Given a cell name, if the cell was the result of a merging, recovers
        the names of individual cells.    
        """
        if c in self.merged_cells:
            return list(self.merged_cells[c])
        else:
            return [c]

    def merge_cells(self, r, l, merge_edges=True):
        """
        Merges a list of individual cells, adds the merged cell and removes them.
        """
        # add new cell that is the merging of a list of cells
        # automatically merges the edges

        new = "+".join(l)  # n1 + "+" + n2
        self.merged_cells[new] = tuple(l)
        self.add_cell(r, sum([self.get_cell_width(n) for n in l]), name=new)

        new_fwd_edges = {}
        new_bwd_edges = {}

        for e in (sum([self.fwd_edges[n] for n in l],
                      [])):  # + self.fwd_edges[n2]):
            if e in self.edge_name_to_data:
                (c1, c2, w) = self.edge_name_to_data[e]
                if merge_edges:
                    if c2 not in new_fwd_edges:
                        new_fwd_edges[c2] = 0
                    new_fwd_edges[c2] += w
                else:
                    self.add_edge(new, c2, w)
                #print(new, c2)
        for e in (sum([self.bwd_edges[n] for n in l], [])):
            if e in self.edge_name_to_data:
                (c1, c2, w) = self.edge_name_to_data[e]
                if merge_edges:
                    if c1 not in new_bwd_edges:
                        new_bwd_edges[c1] = 0
                    new_bwd_edges[c1] += w
                else:
                    self.add_edge(c1, new, w)
                #print(c1, new)

        if merge_edges:
            for c in new_fwd_edges:
                self.add_edge(new, c, new_fwd_edges[c])
            for c in new_bwd_edges:
                self.add_edge(c, new, new_bwd_edges[c])

        for n in l:
            self.remove_cell(n)

    def merge_cells_2(self, r, l, merge_edges=True):
        # merge cells at a given round, identified by their number
        self.merge_cells(r, [self.cell_round_pos_to_name[(r, i)] for i in l],
                         merge_edges=merge_edges)

    def simplify(self):
        # remove all cells which do not have backward AND forward constraints
        # at the same time
        simplified = True
        while simplified:
            simplified = False
            to_remove = []
            for cname in self.cell_name_to_data:
                if ((self.get_cell_width(cname) == 1)
                        and ((cname not in self.fwd_graph) or
                             (cname not in self.bwd_graph) or
                             (not self.fwd_graph[cname]) or
                             (not self.bwd_graph[cname]))):
                    to_remove.append(cname)
            simplified = (to_remove != [])
            for cname in to_remove:
                self.remove_cell(cname)

    def add_edge(self, c1, c2, w):
        """
        Adds an edge between two cells and returns its new name.
        """
        if c1 not in self.cell_name_to_data or c2 not in self.cell_name_to_data:
            raise ValueError("Unexisting cell")
        if c1 not in self.individual_links_fwd:
            self.individual_links_fwd[c1] = []
        if c2 not in self.individual_links_fwd[c1]:
            self.individual_links_fwd[c1].append(c2)
        idx = 0
        if c2 in self._edge_numbering_helper[c1]:
            self._edge_numbering_helper[c1][c2] += 1
            idx = self._edge_numbering_helper[c1][c2]
        else:
            self._edge_numbering_helper[c1][c2] = 0
        name = str(c1) + ":" + str(c2) + ":" + str(idx)

        self.fwd_edges[c1].append(name)
        self.bwd_edges[c2].append(name)
        cur_round = self.cell_name_to_data[c1][0]
        self.edge_names_by_round[cur_round].append(name)

        # multiple edges are possible
        self.edge_name_to_data[name] = c1, c2, w
        if c2 not in self.fwd_graph[c1]:
            self.fwd_graph[c1][c2] = 0
        if c1 not in self.bwd_graph[c2]:
            self.bwd_graph[c2][c1] = 0
        self.fwd_graph[c1][c2] += w
        self.bwd_graph[c2][c1] += w
        return name

    def individual_link_exists(self, c1, c2):
        return (c2 in self.individual_links_fwd[c1])

    def add_edge_2(self, r, i1, i2, w):
        """
        Adds an edge between two cells, identified by their index.
        """
        c1 = self.get_cell_name(r, i1)
        c2 = self.get_cell_name((r + 1) % self.nrounds, i2)
        return self.add_edge(c1, c2, w)

    def get_cell_name(self, r, pos):
        """
        Returns the name of a cell at a given position.
        """
        return self.cell_round_pos_to_name[(r, pos)]

    def set_global(self, name):
        """
        Sets an edge, given by its name, as a global constraint.
        """
        self.global_fixed.append(name)

    def get_cells_by_round(self, r):
        return self.cell_names_by_round[r]

    def get_cell_width(self, n):
        return self.cell_name_to_data[n][2]

    def get_cell_pos(self, n):
        return self.cell_pos_storage[n]

    def get_edges_by_round(self, r):
        return self.edge_names_by_round[r]

    def edge_data(self, n):
        return self.edge_name_to_data[n]

    def get_data(self):
        """
        Returns the data that we need for the generic solver:
        - a dictionary of cell names : cell data (round, position, weight)
        - a dictionary of rounds : cells for this round
        - a dictionary of edge names : edge data (cell at previous round, cell at next round, weight)
        - a dictionary of rounds : edges for this round
        - a list of edge names which are globally fixed
        """
        #cells, cells_by_round, linear_constraints, linear_by_round, global_fixed
        return ({
            c: self.cell_name_to_data[c][2]
            for c in self.cell_name_to_data
        }, self.cell_names_by_round, self.edge_name_to_data,
                self.edge_names_by_round, self.global_fixed)

    def __repr__(self):
        return str(self)

    def __str__(self):
        res = "Present-like constraint set: "
        res += str(self.get_data())
        return res
