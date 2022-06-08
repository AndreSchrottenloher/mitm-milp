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
4-round MITM distinguishers on Sparkle-256 and -384, and
5-round MITM distinguisher on Sparkle-512. The latter uses Pycryptosat for
SAT solving.
Do not mind the "assert" commands, which are here only to check that everything
is going as planned. Use the "new" parameter to compute a new solution than
the ones given in the paper (it will take approx. 2 minutes for Sparkle-512).
"""

import time
from sparkle_util import *


def print_for_c(state):
    # converting to 32 bits if you want to check with the C reference implementation
    state32 = []
    for i in state:
        state32.append(i // 2**32)
        state32.append(i % 2**32)
    print(', '.join([str(hex(t)) for t in state32]))


def attack256(new=False):
    """
    Simple GAD distinguisher on 4-step Sparkle-256. There is no SAT solving here.
    """
    # input and output
    s0_0 = alzette(0 ^ rcon[0], 0)
    s3_0 = 0

    # internal
    if new:
        s1_0 = random.randrange(2**64)
    else:
        s1_0 = 0
    s1_1 = 0
    # deduce at round 1
    s1_2 = alzette(s0_0, 2)
    t1_2 = s1_0 ^ s1_2
    mb1 = biglfunc(s1_0 ^ s1_1)
    assert mb1 == 0
    s2_1 = alzette(t1_2 ^ mb1 ^ 2, 1)
    # deduce at round 2
    s2_3 = alzette(s1_1, 3)
    t2_3 = s2_3 ^ s2_1
    v2_3 = inv_alzette(s3_0, 0) ^ rcon[3]
    mb2 = v2_3 ^ t2_3
    # now we know that mb2 = bigellfunc(s2_1^s2_0) so we can deduce s2_0
    s2_0 = inv_biglfunc(mb2) ^ s2_1
    # we have a complete state at r2

    v1_0 = s1_0
    v1_1 = s1_1
    v1_2 = inv_alzette(s2_1, 1) ^ 2
    v1_3 = inv_alzette(s2_0, 0) ^ rcon[2]
    assert s1_2 == alzette(s0_0, 2)
    assert s1_2 == v1_2 ^ mb1 ^ s1_0

    state = [v1_3, v1_2, v1_0, v1_1]
    inv_sparkle(state, 0, 2)  # invert 2 first steps
    print("Initial state:")
    print(' '.join([str(hex(t))[2:] for t in state]))
    assert state[0] == 0
    print_for_c(state)
    print("After 4 rounds:")
    sparkle(state, 0, 4)
    assert state[2] == 0
    print(' '.join([str(hex(t))[2:] for t in state]))


def attack512(new=False):
    """
    GAD distinguisher on 5-step Sparkle-512. Uses SAT solving.
    """
    found = False
    start_time = time.time()

    while not found:
        print("Trying...")

        # input constraint at step 0
        s0_3 = alzette(0, 3)
        # output constraint at step 4
        v3_6 = inv_alzette(0, 1) ^ 4

        # additional guesses: mb1 = 0, mb2 = 0, and 4 guesses in the middle
        s2_4, s2_5 = 0, rcon[3]
        if not new:
            s2_2 = 0x469ae7b20e4b9a16
        else:
            s2_2 = random.randrange(2**64)

        s2_3 = s2_2
        # these guesses ensure a nice-looking ARX equation towards the end

        # now deduce as much as we can from them
        s1_7 = alzette(s0_3, 7)
        s1_0 = inv_alzette(s2_4, 4)
        s1_1 = inv_alzette(s2_5, 5)
        v1_7 = inv_alzette(s2_2, 2)
        s1_3 = v1_7 ^ s1_7

        s1_2 = s1_1 ^ s1_0 ^ s1_3  # use that mb1 = 0

        # continue deductions at rd2
        s2_6 = alzette(s1_2, 6)
        s2_7 = alzette(s1_3, 7)
        v2_6 = s2_6 ^ s2_2
        v2_7 = s2_7 ^ s2_3
        # then at rd3
        s3_1 = alzette(v2_6 ^ 3, 1)  # don't forget rd constant
        s3_2 = alzette(v2_7, 2)
        s3_6 = alzette(s2_2, 6)
        s3_7 = alzette(s2_3, 7)
        t3_6 = s3_6 ^ s3_2
        # now deduce mb3
        mb3 = t3_6 ^ v3_6
        # we have mb3 = bigell(s3_0^s3_1^s3_2^s3_3)
        C = inv_biglfunc(mb3) ^ s3_1 ^ s3_2
        # we deduce the eq s3_0^s3_3 = C where s3_0 = A0(s2_0) and s3_3 = A3(s2_0)
        # and s2_1 = s2_0

        # solve this system
        s = Solver(verbose=0)
        val = nbr_to_bits(C)
        v1 = [s.new_literal() for i in range(64)]

        term1 = s.solver_alzette(v1, 0)
        term2 = s.solver_alzette(v1, 3)
        tmp = s.solver_xor(term1, term2)
        s.solver_set_equal(tmp, val)

        sat = False
        if new:
            sat, solution = s.solver.solve()

        if sat or (not new):
            found = True
            print("Found !")
            print("s2_2 :", hex(s2_2))
            if new:
                v1bits = [int(solution[lit]) for lit in v1]
                s2_0 = bits_to_nbr(v1bits, w=64)
            else:
                s2_0 = 0x899bde2c383424e
            print("s2_0 :", hex(s2_0))
            s2_1 = s2_0
            # we now have the complete state after 2.5 rounds.
            # invert the alzette layer on that, permute, & invert 2 rounds of Sparkle

            state = [
                inv_alzette(s2_0, 0) ^ rcon[2],
                inv_alzette(s2_1, 1) ^ 2,
                inv_alzette(s2_2, 2),
                inv_alzette(s2_3, 3),
                inv_alzette(s2_4, 4),
                inv_alzette(s2_5, 5),
                inv_alzette(s2_6, 6),
                inv_alzette(s2_7, 7)
            ]

            # checking
            assert s2_0 ^ s2_1 ^ s2_2 ^ s2_3 == 0
            assert s1_0 ^ s1_1 ^ s1_2 ^ s1_3 == 0
            assert s2_5 ^ s2_1 ^ rcon[3] == s2_0
            s3_0 = alzette(s2_1, 0)
            s3_3 = alzette(s2_0, 3)
            assert s3_0 ^ s3_1 ^ s3_2 ^ s3_3 == inv_biglfunc(mb3)
            assert s3_6 == alzette(s2_2, 6)
            assert v3_6 == s3_6 ^ s3_2 ^ mb3

            # moment of truth
            inv_sparkle(state, 0, 2)
            print("Initial state:")
            print(' '.join([str(hex(t))[2:] for t in state]))
            assert state[3] == 0
            print_for_c(state)
            # some checks again
            assert state[3] == inv_alzette(s0_3, 3)
            sparkle(state, 0, 1)
            assert state[0] == inv_alzette(s1_0, 0) ^ rcon[1]
            assert state[1] == inv_alzette(s1_1, 1) ^ 1
            assert state[2] == inv_alzette(s1_2, 2)
            assert state[3] == inv_alzette(s1_3, 3)
            sparkle(state, 1, 2)
            assert state[0] == inv_alzette(s2_0, 0) ^ rcon[2]
            assert state[1] == inv_alzette(s2_1, 1) ^ 2
            assert state[2] == inv_alzette(s2_2, 2)
            assert state[3] == inv_alzette(s2_3, 3)

            print("After 5 rounds:")
            sparkle(state, 2, 5)
            assert state[5] == 0
            print(' '.join([str(hex(t))[2:] for t in state]))

            end_time = time.time()
            print("The computation took ", round(end_time - start_time, 4),
                  " seconds.")


def attack384(new=False):
    """
    GAD distinguisher on 4-step Sparkle-384. Does not use SAT solving.
    """
    # input and output constraints
    s0_0 = alzette(0 ^ rcon[0], 0)
    s3_1 = 0

    # internal guesses
    if new:
        s1_0 = random.randrange(2**64)
    else:
        s1_0 = 0
    s1_1 = 0
    s1_2 = 0
    # deductions at step 1
    s1_3 = alzette(s0_0, 3)
    v1_3 = s1_3 ^ biglfunc(s1_0 ^ s1_1 ^ s1_2) ^ s1_0
    # deductions at step 2
    s2_2 = alzette(v1_3, 2)
    s2_3 = alzette(s1_0, 3)
    s2_4 = alzette(s1_1, 4)
    s2_5 = alzette(s1_2, 5)
    v2_5 = inv_alzette(s3_1, 1) ^ 3
    mb2 = v2_5 ^ s2_5 ^ s2_2
    # make another guess
    s2_0 = 0
    # deduce s2_1
    s2_1 = inv_biglfunc(mb2) ^ s2_2 ^ s2_0

    state = [
        inv_alzette(s2_0, 0) ^ rcon[2],
        inv_alzette(s2_1, 1) ^ 2,
        inv_alzette(s2_2, 2),
        inv_alzette(s2_3, 3),
        inv_alzette(s2_4, 4),
        inv_alzette(s2_5, 5)
    ]

    # moment of truth
    inv_sparkle(state, 0, 2)
    print("Initial state:")
    assert state[0] == 0
    print(' '.join([str(hex(t))[2:] for t in state]))
    print_for_c(state)
    print("After 4 rounds:")
    sparkle(state, 0, 4)
    assert state[4] == 0
    print(' '.join([str(hex(t))[2:] for t in state]))


if __name__ == "__main__":

    # replace by "new=True" to output new solutions

    print("==== SPARKLE-256 4-STEP DISTINGUISHER")
    attack256(new=False)

    print("==== SPARKLE-384 4-STEP DISTINGUISHER")
    attack384(new=False)

    print("==== SPARKLE-512 5-STEP DISTINGUISHER")
    attack512(new=False)
