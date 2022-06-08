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
Custom implementation of Simpira v2 in python. It's based on a python native
implementation of AES components which is adapted from various web sources. 
An AES state is represented as a list of 16 bytes, with the standard AES
numbering: column 0 is bytes [0,1,2,3], column 1 is bytes [4,5,6,7], etc.

The AES operations are computed out of place. Simpira is computed in place. To
access the implementation, just use:

>>> simpira(x)

Where x is a list of AES states represented as above. It applies the applicable
variant of Simpira depending on the length of this input, in place.
"""

import sys, hashlib, string, getpass
from copy import copy

#==================================
# IMPLEMENTATION OF AES ROUNDS

# Lookup tables for s-box and inverse

sbox = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b,
    0xfe, 0xd7, 0xab, 0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26,
    0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2,
    0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed,
    0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f,
    0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec,
    0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14,
    0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d,
    0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f,
    0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1, 0xf8, 0x98, 0x11,
    0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f,
    0xb0, 0x54, 0xbb, 0x16
]

inv_sbox = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e,
    0x81, 0xf3, 0xd7, 0xfb, 0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87,
    0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb, 0x54, 0x7b, 0x94, 0x32,
    0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49,
    0x6d, 0x8b, 0xd1, 0x25, 0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16,
    0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92, 0x6c, 0x70, 0x48, 0x50,
    0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05,
    0xb8, 0xb3, 0x45, 0x06, 0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
    0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b, 0x3a, 0x91, 0x11, 0x41,
    0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8,
    0x1c, 0x75, 0xdf, 0x6e, 0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89,
    0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b, 0xfc, 0x56, 0x3e, 0x4b,
    0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59,
    0x27, 0x80, 0xec, 0x5f, 0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d,
    0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef, 0xa0, 0xe0, 0x3b, 0x4d,
    0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63,
    0x55, 0x21, 0x0c, 0x7d
]


# multiplication in the finite field
def ff_mult(a, b):
    p = 0
    tmp = 0
    for i in range(8):
        if b & 1 == 1:
            p ^= a
        tmp = a & 0x80
        a <<= 1
        if tmp == 0x80:
            a ^= 0x1b
        b >>= 1
    return p % 256


# mix columns (out of place)
def mix_columns(state):
    res = [None] * 16
    for i in range(4):
        res[0 + 4 * i] = (ff_mult(state[0 + 4 * i], 2)
                          ^ ff_mult(state[3 + 4 * i], 1)
                          ^ ff_mult(state[2 + 4 * i], 1)
                          ^ ff_mult(state[1 + 4 * i], 3))
        res[1 + 4 * i] = (ff_mult(state[1 + 4 * i], 2)
                          ^ ff_mult(state[0 + 4 * i], 1)
                          ^ ff_mult(state[3 + 4 * i], 1)
                          ^ ff_mult(state[2 + 4 * i], 3))
        res[2 + 4 * i] = (ff_mult(state[2 + 4 * i], 2)
                          ^ ff_mult(state[1 + 4 * i], 1)
                          ^ ff_mult(state[0 + 4 * i], 1)
                          ^ ff_mult(state[3 + 4 * i], 3))
        res[3 + 4 * i] = (ff_mult(state[3 + 4 * i], 2)
                          ^ ff_mult(state[2 + 4 * i], 1)
                          ^ ff_mult(state[1 + 4 * i], 1)
                          ^ ff_mult(state[0 + 4 * i], 3))
    return res


# aes round (out of place)
def aes_round(state, round_constant):
    res = [None] * 16
    # subbytes and shiftrows
    shift = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]
    for i in range(16):
        res[i] = sbox[state[shift[i]]]
    # mix columns
    res = mix_columns(res)
    for i in range(len(res)):
        res[i] = res[i] ^ round_constant[i]
    return res


# inverse aes round (out of place)
def inv_aes_round(state, round_constant):
    res = [None] * 16
    for i in range(len(state)):
        res[i] = state[i] ^ round_constant[i]
    # M^4 = I, so we don't have to implement the inverse
    for i in range(3):
        res = mix_columns(res)
    # inverse shiftrows and subbytes
    shift = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]
    tmp = [res[shift.index(i)] for i in range(16)]
    for i in range(16):
        res[i] = inv_sbox[tmp[i]]
    return res


# conversion of an AES state into 4 32-bit hex numbers representing the columns
def aes_state_to_hex(state):
    # map state to columns
    res = [0, 0, 0, 0]
    for i in range(4):
        for j in range(4):
            res[i] += state[4 * i + j] * ((256)**j)
    return [hex(t) for t in res]


#=========================================
# IMPLEMENTATION OF SIMPIRA


# Round constant depending on b (number of branches) and i (current index)
def const(i, b=4):
    tmp0 = ((i ^ b) % 2**8)
    tmp1 = ((i ^ b) // 2**8) % (2**8)
    tmp2 = ((i ^ b) // 2**16) % (2**8)
    tmp3 = ((i ^ b) // 2**24) % (2**8)
    return [
        0x00 ^ tmp0, tmp1, tmp2, tmp3, 0x10 ^ tmp0, tmp1, tmp2, tmp3,
        0x20 ^ tmp0, tmp1, tmp2, tmp3, 0x30 ^ tmp0, tmp1, tmp2, tmp3
    ]


# XOR of two states
def xor(x, y):
    return [x[i] ^ y[i] for i in range(len(x))]


# Round function pi (F in the Simpira specification), out of place
def pi(x, i, b=4):
    return aes_round(aes_round(x, const(i, b=b)), [0] * 16)


# Inverse of pi, out of place
def invpi(x, i, b=4):
    return inv_aes_round(inv_aes_round(x, [0] * 16), const(i, b=b))


# Full or reduced Simpira-2, in place
def simpira2(x, nrounds=15):
    i = 1
    for r in range(nrounds):
        x[(r + 1) % 2] = xor(x[(r + 1) % 2], pi(x[r % 2], i, b=2))
        i += 1


# Full or reduced Simpira-3, in place
def simpira3(x, nrounds=21):
    if len(x) != 3:
        raise ValueError("wrong state size!")
    i = 1
    for r in range(nrounds):
        x[(r + 1) % 3] = xor(x[(r + 1) % 3], pi(x[r % 3], i, b=3))
        i += 1


# Full or reduced Simpira-4, in place
def simpira4(x, nrounds=15):
    i = 1
    for r in range(nrounds):
        x[(r + 1) % 4] = xor(x[(r + 1) % 4], pi(x[r % 4], i, b=4))
        i += 1
        x[(r + 3) % 4] = xor(x[(r + 3) % 4], pi(x[(r + 2) % 4], i, b=4))
        i += 1


# Full or reduced Simpira-6, in place
def simpira6(x, nrounds=15):
    if len(x) != 6:
        raise ValueError("wrong state size!")
    i = 1
    s = [0, 1, 2, 5, 4, 3]
    for r in range(nrounds):
        x[s[(r + 1) % 6]] = xor(x[s[(r + 1) % 6]], pi(x[s[(r) % 6]], i, b=6))
        i += 1
        x[s[(r + 5) % 6]] = xor(x[s[(r + 5) % 6]], pi(x[s[(r + 2) % 6]],
                                                      i,
                                                      b=6))
        i += 1
        x[s[(r + 3) % 6]] = xor(x[s[(r + 3) % 6]], pi(x[s[(r + 4) % 6]],
                                                      i,
                                                      b=6))
        i += 1


# Full or reduced Simpira-8, in place
def simpira8(x, nrounds=18):
    if len(x) != 8:
        raise ValueError("wrong state size!")
    i = 1
    s = [0, 1, 6, 5, 4, 3]
    t = [2, 7]
    for r in range(nrounds):
        x[s[(r + 1) % 6]] = xor(x[s[(r + 1) % 6]], pi(x[s[(r) % 6]], i, b=8))
        i += 1
        x[s[(r + 5) % 6]] = xor(x[s[(r + 5) % 6]], pi(x[t[(r) % 2]], i, b=8))
        i += 1
        x[s[(r + 3) % 6]] = xor(x[s[(r + 3) % 6]], pi(x[s[(r + 4) % 6]],
                                                      i,
                                                      b=8))
        i += 1
        x[t[(r + 1) % 2]] = xor(x[t[(r + 1) % 2]], pi(x[s[(r + 2) % 6]],
                                                      i,
                                                      b=8))
        i += 1


# for larger designs it becomes more difficult to speak of "rounds", and rather
# we'll count the number of evaluations of f
def nbr_of_df(b):
    res = 0
    d = (b // 2) * 2
    for j in range(3):
        if d != b:
            res += 1
        for r in range(d - 1):
            res += 1
            if (r != d - r - 2):
                res += 1
        if d != b:
            res += 1
    return res


# Big version of Simpira, potentially reduced to a certain number of double-F functions.
def simpirabig(x, ndf=183):
    b = len(x)

    def doubleF(x, r, k):
        if (r % 2):
            x[(r) % b] = xor(x[(r) % b], pi(x[(r + 1) % b], 2 * k + 1, b))
            x[(r + 1) % b] = xor(x[(r + 1) % b], pi(x[(r) % b], 2 * k + 2, b))
        else:
            x[(r + 1) % b] = xor(x[(r + 1) % b], pi(x[(r) % b], 2 * k + 1, b))
            x[(r) % b] = xor(x[(r) % b], pi(x[(r + 1) % b], 2 * k + 2, b))

    k = 0
    d = (b // 2) * 2

    for j in range(3):
        if d != b:
            doubleF(x, b - 2, k)
            k += 1
            if k >= ndf:
                break
        for r in range(d - 1):
            doubleF(x, r, k)
            k += 1
            if k >= ndf:
                break
            if (r != d - r - 2):
                doubleF(x, d - r - 2, k)
                k += 1
            if k >= ndf:
                break
        if k >= ndf:
            break

        if d != b:
            doubleF(x, b - 2, k)
            k += 1
        if k >= ndf:
            break


# Generic version of Simpira
def simpira(x, nrounds=18):
    b = len(x)
    if b == 2:
        simpira2(x, nrounds)
    elif b == 3:
        simpira3(x, nrounds)
    elif b == 4:
        simpira4(x, nrounds)
    elif b == 6:
        simpira6(x, nrounds)
    elif b == 8:
        simpira8(x, nrounds)
    else:
        simpirabig(x, nrounds)


def test():

    tmp = [0] * 16
    assert invpi(pi(tmp, 0), 0) == [0] * 16
    assert pi(invpi(tmp, 0), 0) == [0] * 16
    # test vector for simpira-4
    #    x: 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
    # 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
    #y: e5275708 895695a0 29f6952f a14b53ae 50ac1683 cfa4851e 604e571b 5e211b5d
    # 5783d7bc 80c8f844 e754d0cd 1a8c43f3 c2a60c46 7907db4b 70d28e06 db9b9dcd
    tmp = [[0] * 16, [0] * 16, [0] * 16, [0] * 16]
    simpira4(tmp, 15)
    assert aes_state_to_hex(tmp[0])[0] == '0xe5275708'
    assert aes_state_to_hex(tmp[1])[1] == '0xcfa4851e'

    # test vector for simpira-2
    #x: 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
    #y: 7dca956b cf46da8c 3044ab97 c627efa8 a664b431 556a10ed 3ba8303e c2148ca0
    tmp = [[0] * 16, [0] * 16]
    simpira2(tmp, 15)
    assert aes_state_to_hex(tmp[0])[0] == '0x7dca956b'
    assert aes_state_to_hex(tmp[1])[1] == '0x556a10ed'

    # test vector for simpira-3
    #x: 00000000 00000000 00000000 00000000 00000000 00000000
    # 00000000 00000000 00000000 00000000 00000000 00000000
    #y: 1194cfd7 418145aa f1b60775 cd8ecff4 d819c52c 6172338a
    # 4a211305 94501cad 6b9e2a00 49ed6e72 0d3b3651 db842384
    tmp = [[0] * 16, [0] * 16, [0] * 16]
    simpira3(tmp, 21)
    assert aes_state_to_hex(tmp[0])[0] == '0x1194cfd7'

    # test vector for simpira-6
    tmp = [[0] * 16, [0] * 16, [0] * 16, [0] * 16, [0] * 16, [0] * 16]
    simpira6(tmp, 15)
    assert aes_state_to_hex(tmp[0])[0] == '0x8b683692'

    # test vector for simpira-8
    tmp = [[0] * 16, [0] * 16, [0] * 16, [0] * 16, [0] * 16, [0] * 16,
           [0] * 16, [0] * 16]
    simpira8(tmp, 18)
    assert aes_state_to_hex(tmp[0])[0] == '0x124c7d5a'

    # test vector for simpira-32
    tmp = [[0] * 16 for i in range(32)]
    simpirabig(tmp)
    assert aes_state_to_hex(tmp[0])[0] == '0x52a7105a'


def aes_state_to_str(aes_state):
    return ' '.join([s[2:] for s in aes_state_to_hex(aes_state)])


def simpira_state_to_str(state):
    # [['0xd1a2902f', '0x9b2b34c2', '0x9bdcf087', '0xb6396692']
    # -> 'd1a2902f 9b2b34c2 9bdcf087 b6396692 | '
    return " | ".join([
        ' '.join([s[2:] for s in aes_state_to_hex(aes_state)])
        for aes_state in state
    ])


if __name__ == "__main__":

    #test()
    import time
    t1 = time.time()
    #print(time.time())
    for i in range(2**10):
        tmp = [[0] * 16, [0] * 16, [0] * 16, [0] * 16]
        simpira4(tmp, 15)
    print(time.time() - t1)
