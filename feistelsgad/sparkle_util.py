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
Python implementation of Sparkle & components. Includes an implementation
of ARX-box equations into a SAT solver. The SAT solver used is Cryptominisat
(https://github.com/msoos/cryptominisat/releases), through the package pycryptosat.
"""

from random import randrange
import random
from copy import copy
from pycryptosat import Solver as InternalSolver
import time

#=======================================================

rcon = [
    0xB7E15162, 0xBF715880, 0x38B4DA56, 0x324E7738, 0xBB1185EB, 0x4F7C7B57,
    0xCFBFA1C8, 0xC2B3293D
]


def nbr_to_bits(n, w=64):
    """
    Converts an integer to a list of bits.
    """
    tmp = str(bin(n))[2:]
    tmp = "0" * (w - len(tmp)) + tmp
    return [int(c) for c in tmp]


def bits_to_nbr(b, w=64):
    """
    Converts a list of bits to an integer.
    """
    tmp = copy(b)
    tmp.reverse()
    return sum([tmp[i] * 2**i for i in range(len(tmp))])


def rotate(s, r):
    """
    Rotates a list of bits.
    """
    return [s[(i - r) % len(s)] for i in range(len(s))]


def rotatenum(x, r):
    """
    Rotates the bits of a 32-bit integer, by the amount r (rotation operation
    used in Alzette).
    """
    res = (((x) >> (r)) | ((x) << (32 - (r)))) % (2**32)
    # ((x << r) | (x >> (32-r))) & MASK_N
    return res


def ell(x):
    return (x % 2**16) * 2**16 + ((x % 2**16) ^ (x // 2**16))


def inv_ell(x):
    return ((x % 2**16) ^ (x // 2**16)) * 2**16 + (x // 2**16)


def biglfunc(s):
    # on 64 bits: returns ell(second part), ell(first part)
    x = s // (2**32)
    y = s % (2**32)
    return ell(x) + (ell(y) * 2**32)


def inv_biglfunc(s):
    x = s // (2**32)
    y = s % (2**32)
    return inv_ell(x) + (inv_ell(y) * 2**32)


def biglplus(s):
    """
    Returns biglfunc(s) XOR s
    """
    x = s // (2**32)
    y = s % (2**32)
    return ell(x) ^ y + ((ell(y) ^ x) * 2**32)


def inv_biglplus(s):
    """
    Inverse of biglplus.
    """
    bits = nbr_to_bits(s, w=64)
    s0p = bits[0:16]
    s1p = bits[16:32]
    s2p = bits[32:48]
    s3p = bits[48:]
    s1 = xorbits(s0p, s3p)
    s2 = xorbits(s1, s2p)
    s3 = xorbits(s1p, xorbits(s1, s2))
    s0 = xorbits(s0p, s3)
    res = bits_to_nbr(s0 + s1 + s2 + s3, w=64)
    return res


def xorbits(l1, l2):
    """
    XORs two lists of bits.
    """
    return [l1[i] ^ l2[i] for i in range(len(l1))]


class Solver:
    """
    Custom class to simplify the use of SAT solvers. We use it to implement
    ARX-box equations.
    """

    def __init__(self, verbose=1):
        self.solver = InternalSolver(verbose=verbose)
        self.literal_counter = 0

    def new_literal(self):
        self.literal_counter += 1
        return self.literal_counter

    def solver_xor(self, *args):
        """
        For several individual literals a,b,c..., returns a literal v such 
        that v = a XOR b XOR c ...
        Does the same bit by bit if a,b,c... are lists of literals of the same length.
        """
        list_args = list(args)
        list_mode = False
        for a in list_args:
            if type(a) == list:
                list_mode = True
            elif list_mode:
                raise ValueError("invalid type")
        nb = 0
        if list_mode:
            nb = len(list_args[0])
            for a in list_args:
                if len(a) != nb:
                    raise ValueError("Invalid list lengths")
        if list_mode:
            res = [None] * nb
            for i in range(nb):
                res[i] = self.solver_xor(*[a[i] for a in list_args])
        else:
            res = self.new_literal()
            self.solver.add_xor_clause([res] + list_args, False)

        return res

    def solver_and(self, a, b):
        """
        Returns the AND of two literals, or two lists of literals.
        """
        if type(a) == list:
            res = [None] * len(a)
            for i in range(len(a)):
                res[i] = self.solver_and(a[i], b[i])
        else:
            res = self.new_literal()
            self.solver.add_clause([-res, a])
            self.solver.add_clause([-res, b])
            self.solver.add_clause([res, -b, -a])
        return res

    def solver_or(self, a, b):
        """
        Returns the OR of two literals, or two lists of literals.
        """
        if type(a) == list:
            res = [None] * len(a)
            for i in range(len(a)):
                res[i] = self.solver_or(a[i], b[i])
        else:
            res = self.new_literal()
            self.solver.add_clause([res, -a])
            self.solver.add_clause([res, -b])
            self.solver.add_clause([-res, b, a])
        return res

    def solver_xor_bits(self, a, c):
        """
        Returns the XOR of literals with bits.
        """
        if type(a) == list:
            res = [None] * len(a)
            for i in range(len(a)):
                res[i] = self.solver_xor_bits(a[i], c[i])
        else:
            res = self.new_literal()
            # res = a XOR c
            self.solver.add_xor_clause([a, res], bool(c))
        return res

    #===========
    # Implementation of modular addition and subtraction, using a simple
    # boolean circuit with adders.

    def _half_add(self, a, b):
        return self.solver_xor(a, b), self.solver_and(a, b)

    def _full_add(self, a, b, cin):
        s, cout = self._half_add(a, b)
        ss, ccout = self._half_add(cin, s)
        return ss, self.solver_or(ccout, cout)

    def solver_add(self, ll1, ll2):
        l1 = copy(ll1)
        l1.reverse()
        l2 = copy(ll2)
        l2.reverse()
        # returns list of bits which is modular addition of that
        res = [None] * len(l1)
        tmp, c = self._half_add(l1[0], l2[0])
        res[0] = tmp
        for i in range(1, len(l1)):
            tmp, cc = self._full_add(l1[i], l2[i], c)
            c = cc
            res[i] = tmp
        res.reverse()
        return res

    def solver_simplified_add(self, a, b):
        """
        Returns an approximation of addition: a XOR b XOR ( (a&b << 1) )
        """
        res = [None] * len(a)
        res[-1] = self.solver_xor(a[-1], b[-1])
        for i in range(0, len(a) - 1):
            res[i] = self.solver_xor(a[i], b[i],
                                     self.solver_and(a[i + 1], b[i + 1]))
        return res

    #====================

    def solver_alzette(self, s, i):
        """
        Returns list of literals which represent the bits of alzette(s, i),
        where i is the rd constant index and s is the input state bits.
        """
        x = s[:32]
        y = s[32:]
        bits = nbr_to_bits(rcon[i], w=32)
        for (a, b) in [(31, 24), (17, 17), (0, 31), (24, 16)]:
            x = self.solver_add(x, rotate(y, a))
            y = self.solver_xor(y, rotate(x, b))
            x = self.solver_xor_bits(x, bits)
        return x + y

    def solver_bigell(self, s):
        """
        Returns list of literals which represent bigell(s)
        """
        # cut s into 4 parts
        s0 = s[:16]
        s1 = s[16:32]
        s2 = s[32:48]
        s3 = s[48:]
        # return s3 | s2 + s3 | s1 | s0 + s1
        return (s3 + self.solver_xor(s2, s3) + s1 + self.solver_xor(s0, s1))

    def solver_inv_bigell(self, s):
        s0 = s[:16]
        s1 = s[16:32]
        s2 = s[32:48]
        s3 = s[48:]
        # return s2 + s3 | s2 | s0 + s1 | s0
        return (self.solver_xor(s2, s3) + s2 + self.solver_xor(s0, s1) + s0)

    def solver_set_equal(self, a, c):
        """
        Constraints an equality between literals and values.
        """
        if type(a) == list:
            res = [None] * len(a)
            for i in range(len(a)):
                self.solver_set_equal(a[i], c[i])
        else:
            self.solver.add_xor_clause([a], bool(c))  # True if 1


def alzette(s, i):
    """
    Alzette ARX-Box operating on 64 bits.
    """
    x = s // (2**32)
    y = s % (2**32)
    for (a, b) in [(31, 24), (17, 17), (0, 31), (24, 16)]:
        x = (x + rotatenum(y, a)) % (2**32)
        y = (y ^ rotatenum(x, b))
        x = x ^ rcon[i]
    return y + (x * 2**32)


def simplified_add(a, b):
    return (a ^ b ^ ((a & b) << 1))


def inv_alzette(s, i):
    """
    Inverse of Alzette ARX-Box.
    """
    x = s // (2**32)
    y = s % (2**32)
    for (a, b) in [(24, 16), (0, 31), (17, 17), (31, 24)]:
        x = x ^ rcon[i]
        y = (y ^ rotatenum(x, b))
        x = (x - rotatenum(y, a) + 2**32) % (2**32)

    return y + (x * 2**32)


def sparkle(instate, start=0, end=5):
    """
    In-place implementation of Sparkle on a list of 64-bit integers.
    """
    nb = len(instate)
    state = instate
    for i in range(start, end):
        state[0] = state[0] ^ (rcon[i])
        state[1] = state[1] ^ (i)
        for j in range(0, nb):
            state[j] = alzette(state[j], j)
        tmp = biglfunc(state[0])
        s0 = state[0]
        for j in range(1, nb // 2):
            tmp ^= biglfunc(state[j])
        for j in range(1, nb // 2):
            state[j - 1] = state[j + nb // 2] ^ state[j] ^ tmp
            state[j + nb // 2] = state[j]
        state[nb // 2 - 1] = state[nb // 2] ^ s0 ^ tmp
        state[nb // 2] = s0


def inv_sparkle(instate, start=0, end=5):
    """
    In-place implementation of inverse Sparkle on a list of 64-bit integers.
    """
    nb = len(instate)
    state = instate
    for i in range(end - 1, start - 1, -1):
        tmp = 0
        tmpb = state[nb // 2 - 1]
        for j in range(nb // 2 - 1, 0, -1):
            state[j] = state[j + nb // 2]
            tmp ^= state[j]
            state[j + nb // 2] = state[j - 1]
        state[0] = state[nb // 2]
        state[nb // 2] = tmpb
        tmp ^= state[0]
        tmp = biglfunc(tmp)
        for j in range(0, nb // 2):
            state[j + nb // 2] ^= tmp ^ state[j]
        for j in range(0, nb):
            state[j] = inv_alzette(state[j], j)
        state[0] = state[0] ^ (rcon[i])
        state[1] = state[1] ^ (i)


if __name__ == "__main__":

    #=== testing some functions
    assert ell(0xB7E15162) == 0x5162e683

    for i in range(10):
        n = random.randrange(2**32)
        assert inv_ell(ell(n)) == n
        assert ell(inv_ell(n)) == n

        n = random.randrange(2**64)
        assert biglfunc(inv_biglfunc(n)) == n
        assert inv_biglfunc(biglfunc(n)) == n

        n = random.randrange(2**32)
        nn = bits_to_nbr(rotate(nbr_to_bits(n, w=32), 10), w=32)
        assert nn == rotatenum(n, 10)

    #=== tesing the solver
    tmp = 150
    assert bits_to_nbr(nbr_to_bits(tmp)) == tmp

    for i in range(10):
        s = Solver(verbose=0)
        n1 = random.randrange(2**64)
        n2 = random.randrange(2**64)

        val1 = nbr_to_bits(n1)
        val2 = nbr_to_bits(n2)
        val3 = nbr_to_bits((n1 + n2) % (2**64))

        v1 = [s.new_literal() for i in range(64)]
        v2 = [s.new_literal() for i in range(64)]
        v3 = s.solver_add(v1, v2)

        s.solver_set_equal(v1, val1)
        s.solver_set_equal(v2, val2)
        s.solver_set_equal(v3, val3)

        sat, solution = s.solver.solve()
        assert sat

    # check that the solver and the alzette implementation agree with each other
    c = 1
    print("Testing Alzette + solver")
    for j in range(20):
        s = Solver(verbose=0)
        n = random.randrange(2**64)
        val1 = nbr_to_bits(n)
        nn = alzette(n, c)
        val2 = nbr_to_bits(nn)

        v1 = [s.new_literal() for i in range(64)]
        v2 = s.solver_alzette(v1, c)

        s.solver_set_equal(v1, val1)
        s.solver_set_equal(v2, val2)
        sat, solution = s.solver.solve()
        assert sat

    # test ell
    print("Testing bigell + solver")
    for j in range(10):
        s = Solver(verbose=0)
        n = random.randrange(2**64)
        val1 = nbr_to_bits(n)
        nn = biglfunc(n)
        val2 = nbr_to_bits(nn)

        v1 = [s.new_literal() for i in range(64)]
        v2 = s.solver_bigell(v1)
        s.solver_set_equal(v1, val1)
        s.solver_set_equal(v2, val2)
        sat, solution = s.solver.solve()
        assert sat

    print("Testing inv bigell + solver")
    for j in range(10):
        s = Solver(verbose=0)
        n = random.randrange(2**64)
        val1 = nbr_to_bits(n)
        nn = inv_biglfunc(n)
        val2 = nbr_to_bits(nn)

        v1 = [s.new_literal() for i in range(64)]
        v2 = s.solver_inv_bigell(v1)
        s.solver_set_equal(v1, val1)
        s.solver_set_equal(v2, val2)
        sat, solution = s.solver.solve()
        assert sat

    #======

    # test vectors
    assert (alzette(0, 0)) == 0x44dd4de9e5581f2d
    assert alzette(0x806f124eb84c4965, 0) == 0xeaae6d21f4c0a271
    for i in range(50):
        t = random.randrange(2**64)
        assert (inv_alzette(alzette(t, t % 8), t % 8) == t)
        assert (alzette(inv_alzette(t, t % 8), t % 8) == t)

    #=== test of Sparkle
    state = [0] * 8
    sparkle(state, start=0, end=5)
    assert (state[0] == 0xb68350e0b3846db5)  # 0xb68350e0)
    #====

    # for reference, 5 steps of Sparkle-512:
    # 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
    # 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
    # ->
    # b68350e0 b3846db5 4a068b3d e74e656f e47a93a2 e4ecf9fa 59aebae1 8c290caf
    # 0a2285f3 44b0310d a411fcd1 76ff7c66 b40b306f ef238738 ebafd2ea 5266f6f7

    #===== test of inverse
    state = [0] * 8
    sparkle(state, start=0, end=5)
    inv_sparkle(state, start=0, end=5)
    assert (state[0] == 0)

    state = [0] * 4
    sparkle(state, start=0, end=5)
    inv_sparkle(state, start=0, end=5)
    assert (state[0] == 0)

    #=============
    # test of simplified adding
    for i in range(10):
        s = Solver(verbose=0)
        n1 = random.randrange(2**64)
        n2 = random.randrange(2**64)

        val1 = nbr_to_bits(n1)
        val2 = nbr_to_bits(n2)
        val3 = nbr_to_bits((n1 ^ n2 ^ ((n1 & n2) << 1)) % (2**64))

        v1 = [s.new_literal() for i in range(64)]
        v2 = [s.new_literal() for i in range(64)]
        v3 = s.solver_simplified_add(v1, v2)

        s.solver_set_equal(v1, val1)
        s.solver_set_equal(v2, val2)
        s.solver_set_equal(v3, val3)

        sat, solution = s.solver.solve()
        assert sat
