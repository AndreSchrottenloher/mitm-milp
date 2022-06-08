
/*
Copyright (c) 2022

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This project has been supported by ERC-ADG-ALGSTRONGCRYPTO (project 740972).

Author: André Schrottenloher & Marc Stevens
Date: June 2022
Version: 2

*/

/*
Demonstrator of our attack against full Haraka-512. This is an implementation
of the algorithms detailed in Appendix C of the full version of the paper 
(https://eprint.iacr.org/2022/189).

Some of this code comes from or was inspired by the reference implementations
of Haraka available at: https://github.com/kste/haraka
*/

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <random>
#include <ctime>
#include <chrono>
#include <thread>
#include <mutex>
#include <cmath>
#include <array>
#include <tuple>
#include <vector>

#include <emmintrin.h>
#include <smmintrin.h> 
#include <tmmintrin.h> 
#include <wmmintrin.h>
#include <xmmintrin.h>

using namespace std;

//========================================================================
//================ CONSTANTS

// Precomputed tables for multiplications that are necessary in our implementation.
// 2, 3 correspond to MC
// 13, 9, 11,14 correspond to the inverse of MC
// the others correspond to linear relations

static const uint8_t table2[256] = {
0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94,
96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126,
128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158,
160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190,
192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222,
224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254,
27, 25, 31, 29, 19, 17, 23, 21, 11, 9, 15, 13, 3, 1, 7, 5,
59, 57, 63, 61, 51, 49, 55, 53, 43, 41, 47, 45, 35, 33, 39, 37,
91, 89, 95, 93, 83, 81, 87, 85, 75, 73, 79, 77, 67, 65, 71, 69,
123, 121, 127, 125, 115, 113, 119, 117, 107, 105, 111, 109, 99, 97, 103, 101,
155, 153, 159, 157, 147, 145, 151, 149, 139, 137, 143, 141, 131, 129, 135, 133,
187, 185, 191, 189, 179, 177, 183, 181, 171, 169, 175, 173, 163, 161, 167, 165,
219, 217, 223, 221, 211, 209, 215, 213, 203, 201, 207, 205, 195, 193, 199, 197,
251, 249, 255, 253, 243, 241, 247, 245, 235, 233, 239, 237, 227, 225, 231, 229
};

static const uint8_t table3[256] = {
0, 3, 6, 5, 12, 15, 10, 9, 24, 27, 30, 29, 20, 23, 18, 17,
48, 51, 54, 53, 60, 63, 58, 57, 40, 43, 46, 45, 36, 39, 34, 33,
96, 99, 102, 101, 108, 111, 106, 105, 120, 123, 126, 125, 116, 119, 114, 113,
80, 83, 86, 85, 92, 95, 90, 89, 72, 75, 78, 77, 68, 71, 66, 65,
192, 195, 198, 197, 204, 207, 202, 201, 216, 219, 222, 221, 212, 215, 210, 209,
240, 243, 246, 245, 252, 255, 250, 249, 232, 235, 238, 237, 228, 231, 226, 225,
160, 163, 166, 165, 172, 175, 170, 169, 184, 187, 190, 189, 180, 183, 178, 177,
144, 147, 150, 149, 156, 159, 154, 153, 136, 139, 142, 141, 132, 135, 130, 129,
155, 152, 157, 158, 151, 148, 145, 146, 131, 128, 133, 134, 143, 140, 137, 138,
171, 168, 173, 174, 167, 164, 161, 162, 179, 176, 181, 182, 191, 188, 185, 186,
251, 248, 253, 254, 247, 244, 241, 242, 227, 224, 229, 230, 239, 236, 233, 234,
203, 200, 205, 206, 199, 196, 193, 194, 211, 208, 213, 214, 223, 220, 217, 218,
91, 88, 93, 94, 87, 84, 81, 82, 67, 64, 69, 70, 79, 76, 73, 74,
107, 104, 109, 110, 103, 100, 97, 98, 115, 112, 117, 118, 127, 124, 121, 122,
59, 56, 61, 62, 55, 52, 49, 50, 35, 32, 37, 38, 47, 44, 41, 42,
11, 8, 13, 14, 7, 4, 1, 2, 19, 16, 21, 22, 31, 28, 25, 26
};

static const uint8_t table7[256] = {
0, 7, 14, 9, 28, 27, 18, 21, 56, 63, 54, 49, 36, 35, 42, 45,
112, 119, 126, 121, 108, 107, 98, 101, 72, 79, 70, 65, 84, 83, 90, 93,
224, 231, 238, 233, 252, 251, 242, 245, 216, 223, 214, 209, 196, 195, 202, 205,
144, 151, 158, 153, 140, 139, 130, 133, 168, 175, 166, 161, 180, 179, 186, 189,
219, 220, 213, 210, 199, 192, 201, 206, 227, 228, 237, 234, 255, 248, 241, 246,
171, 172, 165, 162, 183, 176, 185, 190, 147, 148, 157, 154, 143, 136, 129, 134,
59, 60, 53, 50, 39, 32, 41, 46, 3, 4, 13, 10, 31, 24, 17, 22,
75, 76, 69, 66, 87, 80, 89, 94, 115, 116, 125, 122, 111, 104, 97, 102,
173, 170, 163, 164, 177, 182, 191, 184, 149, 146, 155, 156, 137, 142, 135, 128,
221, 218, 211, 212, 193, 198, 207, 200, 229, 226, 235, 236, 249, 254, 247, 240,
77, 74, 67, 68, 81, 86, 95, 88, 117, 114, 123, 124, 105, 110, 103, 96,
61, 58, 51, 52, 33, 38, 47, 40, 5, 2, 11, 12, 25, 30, 23, 16,
118, 113, 120, 127, 106, 109, 100, 99, 78, 73, 64, 71, 82, 85, 92, 91,
6, 1, 8, 15, 26, 29, 20, 19, 62, 57, 48, 55, 34, 37, 44, 43,
150, 145, 152, 159, 138, 141, 132, 131, 174, 169, 160, 167, 178, 181, 188, 187,
230, 225, 232, 239, 250, 253, 244, 243, 222, 217, 208, 215, 194, 197, 204, 203
};

static const uint8_t table13[256] = {
0, 13, 26, 23, 52, 57, 46, 35, 104, 101, 114, 127, 92, 81, 70, 75,
208, 221, 202, 199, 228, 233, 254, 243, 184, 181, 162, 175, 140, 129, 150, 155,
187, 182, 161, 172, 143, 130, 149, 152, 211, 222, 201, 196, 231, 234, 253, 240,
107, 102, 113, 124, 95, 82, 69, 72, 3, 14, 25, 20, 55, 58, 45, 32,
109, 96, 119, 122, 89, 84, 67, 78, 5, 8, 31, 18, 49, 60, 43, 38,
189, 176, 167, 170, 137, 132, 147, 158, 213, 216, 207, 194, 225, 236, 251, 246,
214, 219, 204, 193, 226, 239, 248, 245, 190, 179, 164, 169, 138, 135, 144, 157,
6, 11, 28, 17, 50, 63, 40, 37, 110, 99, 116, 121, 90, 87, 64, 77,
218, 215, 192, 205, 238, 227, 244, 249, 178, 191, 168, 165, 134, 139, 156, 145,
10, 7, 16, 29, 62, 51, 36, 41, 98, 111, 120, 117, 86, 91, 76, 65,
97, 108, 123, 118, 85, 88, 79, 66, 9, 4, 19, 30, 61, 48, 39, 42,
177, 188, 171, 166, 133, 136, 159, 146, 217, 212, 195, 206, 237, 224, 247, 250,
183, 186, 173, 160, 131, 142, 153, 148, 223, 210, 197, 200, 235, 230, 241, 252,
103, 106, 125, 112, 83, 94, 73, 68, 15, 2, 21, 24, 59, 54, 33, 44,
12, 1, 22, 27, 56, 53, 34, 47, 100, 105, 126, 115, 80, 93, 74, 71,
220, 209, 198, 203, 232, 229, 242, 255, 180, 185, 174, 163, 128, 141, 154, 151
};

static const uint8_t table9[256] = {
0, 9, 18, 27, 36, 45, 54, 63, 72, 65, 90, 83, 108, 101, 126, 119,
144, 153, 130, 139, 180, 189, 166, 175, 216, 209, 202, 195, 252, 245, 238, 231,
59, 50, 41, 32, 31, 22, 13, 4, 115, 122, 97, 104, 87, 94, 69, 76,
171, 162, 185, 176, 143, 134, 157, 148, 227, 234, 241, 248, 199, 206, 213, 220,
118, 127, 100, 109, 82, 91, 64, 73, 62, 55, 44, 37, 26, 19, 8, 1,
230, 239, 244, 253, 194, 203, 208, 217, 174, 167, 188, 181, 138, 131, 152, 145,
77, 68, 95, 86, 105, 96, 123, 114, 5, 12, 23, 30, 33, 40, 51, 58,
221, 212, 207, 198, 249, 240, 235, 226, 149, 156, 135, 142, 177, 184, 163, 170,
236, 229, 254, 247, 200, 193, 218, 211, 164, 173, 182, 191, 128, 137, 146, 155,
124, 117, 110, 103, 88, 81, 74, 67, 52, 61, 38, 47, 16, 25, 2, 11,
215, 222, 197, 204, 243, 250, 225, 232, 159, 150, 141, 132, 187, 178, 169, 160,
71, 78, 85, 92, 99, 106, 113, 120, 15, 6, 29, 20, 43, 34, 57, 48,
154, 147, 136, 129, 190, 183, 172, 165, 210, 219, 192, 201, 246, 255, 228, 237,
10, 3, 24, 17, 46, 39, 60, 53, 66, 75, 80, 89, 102, 111, 116, 125,
161, 168, 179, 186, 133, 140, 151, 158, 233, 224, 251, 242, 205, 196, 223, 214,
49, 56, 35, 42, 21, 28, 7, 14, 121, 112, 107, 98, 93, 84, 79, 70
};

static const uint8_t table11[256] = {
0, 11, 22, 29, 44, 39, 58, 49, 88, 83, 78, 69, 116, 127, 98, 105,
176, 187, 166, 173, 156, 151, 138, 129, 232, 227, 254, 245, 196, 207, 210, 217,
123, 112, 109, 102, 87, 92, 65, 74, 35, 40, 53, 62, 15, 4, 25, 18,
203, 192, 221, 214, 231, 236, 241, 250, 147, 152, 133, 142, 191, 180, 169, 162,
246, 253, 224, 235, 218, 209, 204, 199, 174, 165, 184, 179, 130, 137, 148, 159,
70, 77, 80, 91, 106, 97, 124, 119, 30, 21, 8, 3, 50, 57, 36, 47,
141, 134, 155, 144, 161, 170, 183, 188, 213, 222, 195, 200, 249, 242, 239, 228,
61, 54, 43, 32, 17, 26, 7, 12, 101, 110, 115, 120, 73, 66, 95, 84,
247, 252, 225, 234, 219, 208, 205, 198, 175, 164, 185, 178, 131, 136, 149, 158,
71, 76, 81, 90, 107, 96, 125, 118, 31, 20, 9, 2, 51, 56, 37, 46,
140, 135, 154, 145, 160, 171, 182, 189, 212, 223, 194, 201, 248, 243, 238, 229,
60, 55, 42, 33, 16, 27, 6, 13, 100, 111, 114, 121, 72, 67, 94, 85,
1, 10, 23, 28, 45, 38, 59, 48, 89, 82, 79, 68, 117, 126, 99, 104,
177, 186, 167, 172, 157, 150, 139, 128, 233, 226, 255, 244, 197, 206, 211, 216,
122, 113, 108, 103, 86, 93, 64, 75, 34, 41, 52, 63, 14, 5, 24, 19,
202, 193, 220, 215, 230, 237, 240, 251, 146, 153, 132, 143, 190, 181, 168, 163
};

static const uint8_t table14[256] = {
0, 14, 28, 18, 56, 54, 36, 42, 112, 126, 108, 98, 72, 70, 84, 90,
224, 238, 252, 242, 216, 214, 196, 202, 144, 158, 140, 130, 168, 166, 180, 186,
219, 213, 199, 201, 227, 237, 255, 241, 171, 165, 183, 185, 147, 157, 143, 129,
59, 53, 39, 41, 3, 13, 31, 17, 75, 69, 87, 89, 115, 125, 111, 97,
173, 163, 177, 191, 149, 155, 137, 135, 221, 211, 193, 207, 229, 235, 249, 247,
77, 67, 81, 95, 117, 123, 105, 103, 61, 51, 33, 47, 5, 11, 25, 23,
118, 120, 106, 100, 78, 64, 82, 92, 6, 8, 26, 20, 62, 48, 34, 44,
150, 152, 138, 132, 174, 160, 178, 188, 230, 232, 250, 244, 222, 208, 194, 204,
65, 79, 93, 83, 121, 119, 101, 107, 49, 63, 45, 35, 9, 7, 21, 27,
161, 175, 189, 179, 153, 151, 133, 139, 209, 223, 205, 195, 233, 231, 245, 251,
154, 148, 134, 136, 162, 172, 190, 176, 234, 228, 246, 248, 210, 220, 206, 192,
122, 116, 102, 104, 66, 76, 94, 80, 10, 4, 22, 24, 50, 60, 46, 32,
236, 226, 240, 254, 212, 218, 200, 198, 156, 146, 128, 142, 164, 170, 184, 182,
12, 2, 16, 30, 52, 58, 40, 38, 124, 114, 96, 110, 68, 74, 88, 86,
55, 57, 43, 37, 15, 1, 19, 29, 71, 73, 91, 85, 127, 113, 99, 109,
215, 217, 203, 197, 239, 225, 243, 253, 167, 169, 187, 181, 159, 145, 131, 141
};

static const uint8_t table68[256] = {
0, 68, 136, 204, 11, 79, 131, 199, 22, 82, 158, 218, 29, 89, 149, 209,
44, 104, 164, 224, 39, 99, 175, 235, 58, 126, 178, 246, 49, 117, 185, 253,
88, 28, 208, 148, 83, 23, 219, 159, 78, 10, 198, 130, 69, 1, 205, 137,
116, 48, 252, 184, 127, 59, 247, 179, 98, 38, 234, 174, 105, 45, 225, 165,
176, 244, 56, 124, 187, 255, 51, 119, 166, 226, 46, 106, 173, 233, 37, 97,
156, 216, 20, 80, 151, 211, 31, 91, 138, 206, 2, 70, 129, 197, 9, 77,
232, 172, 96, 36, 227, 167, 107, 47, 254, 186, 118, 50, 245, 177, 125, 57,
196, 128, 76, 8, 207, 139, 71, 3, 210, 150, 90, 30, 217, 157, 81, 21,
123, 63, 243, 183, 112, 52, 248, 188, 109, 41, 229, 161, 102, 34, 238, 170,
87, 19, 223, 155, 92, 24, 212, 144, 65, 5, 201, 141, 74, 14, 194, 134,
35, 103, 171, 239, 40, 108, 160, 228, 53, 113, 189, 249, 62, 122, 182, 242,
15, 75, 135, 195, 4, 64, 140, 200, 25, 93, 145, 213, 18, 86, 154, 222,
203, 143, 67, 7, 192, 132, 72, 12, 221, 153, 85, 17, 214, 146, 94, 26,
231, 163, 111, 43, 236, 168, 100, 32, 241, 181, 121, 61, 250, 190, 114, 54,
147, 215, 27, 95, 152, 220, 16, 84, 133, 193, 13, 73, 142, 202, 6, 66,
191, 251, 55, 115, 180, 240, 60, 120, 169, 237, 33, 101, 162, 230, 42, 110
};

static const uint8_t table71[256] = {
0, 71, 142, 201, 7, 64, 137, 206, 14, 73, 128, 199, 9, 78, 135, 192,
28, 91, 146, 213, 27, 92, 149, 210, 18, 85, 156, 219, 21, 82, 155, 220,
56, 127, 182, 241, 63, 120, 177, 246, 54, 113, 184, 255, 49, 118, 191, 248,
36, 99, 170, 237, 35, 100, 173, 234, 42, 109, 164, 227, 45, 106, 163, 228,
112, 55, 254, 185, 119, 48, 249, 190, 126, 57, 240, 183, 121, 62, 247, 176,
108, 43, 226, 165, 107, 44, 229, 162, 98, 37, 236, 171, 101, 34, 235, 172,
72, 15, 198, 129, 79, 8, 193, 134, 70, 1, 200, 143, 65, 6, 207, 136,
84, 19, 218, 157, 83, 20, 221, 154, 90, 29, 212, 147, 93, 26, 211, 148,
224, 167, 110, 41, 231, 160, 105, 46, 238, 169, 96, 39, 233, 174, 103, 32,
252, 187, 114, 53, 251, 188, 117, 50, 242, 181, 124, 59, 245, 178, 123, 60,
216, 159, 86, 17, 223, 152, 81, 22, 214, 145, 88, 31, 209, 150, 95, 24,
196, 131, 74, 13, 195, 132, 77, 10, 202, 141, 68, 3, 205, 138, 67, 4,
144, 215, 30, 89, 151, 208, 25, 94, 158, 217, 16, 87, 153, 222, 23, 80,
140, 203, 2, 69, 139, 204, 5, 66, 130, 197, 12, 75, 133, 194, 11, 76,
168, 239, 38, 97, 175, 232, 33, 102, 166, 225, 40, 111, 161, 230, 47, 104,
180, 243, 58, 125, 179, 244, 61, 122, 186, 253, 52, 115, 189, 250, 51, 116
};

static const uint8_t table201[256] = {
0, 201, 137, 64, 9, 192, 128, 73, 18, 219, 155, 82, 27, 210, 146, 91,
36, 237, 173, 100, 45, 228, 164, 109, 54, 255, 191, 118, 63, 246, 182, 127,
72, 129, 193, 8, 65, 136, 200, 1, 90, 147, 211, 26, 83, 154, 218, 19,
108, 165, 229, 44, 101, 172, 236, 37, 126, 183, 247, 62, 119, 190, 254, 55,
144, 89, 25, 208, 153, 80, 16, 217, 130, 75, 11, 194, 139, 66, 2, 203,
180, 125, 61, 244, 189, 116, 52, 253, 166, 111, 47, 230, 175, 102, 38, 239,
216, 17, 81, 152, 209, 24, 88, 145, 202, 3, 67, 138, 195, 10, 74, 131,
252, 53, 117, 188, 245, 60, 124, 181, 238, 39, 103, 174, 231, 46, 110, 167,
59, 242, 178, 123, 50, 251, 187, 114, 41, 224, 160, 105, 32, 233, 169, 96,
31, 214, 150, 95, 22, 223, 159, 86, 13, 196, 132, 77, 4, 205, 141, 68,
115, 186, 250, 51, 122, 179, 243, 58, 97, 168, 232, 33, 104, 161, 225, 40,
87, 158, 222, 23, 94, 151, 215, 30, 69, 140, 204, 5, 76, 133, 197, 12,
171, 98, 34, 235, 162, 107, 43, 226, 185, 112, 48, 249, 176, 121, 57, 240,
143, 70, 6, 207, 134, 79, 15, 198, 157, 84, 20, 221, 148, 93, 29, 212,
227, 42, 106, 163, 234, 35, 99, 170, 241, 56, 120, 177, 248, 49, 113, 184,
199, 14, 78, 135, 206, 7, 71, 142, 213, 28, 92, 149, 220, 21, 85, 156
};

static const uint8_t table203[256] = {
0, 203, 141, 70, 1, 202, 140, 71, 2, 201, 143, 68, 3, 200, 142, 69,
4, 207, 137, 66, 5, 206, 136, 67, 6, 205, 139, 64, 7, 204, 138, 65,
8, 195, 133, 78, 9, 194, 132, 79, 10, 193, 135, 76, 11, 192, 134, 77,
12, 199, 129, 74, 13, 198, 128, 75, 14, 197, 131, 72, 15, 196, 130, 73,
16, 219, 157, 86, 17, 218, 156, 87, 18, 217, 159, 84, 19, 216, 158, 85,
20, 223, 153, 82, 21, 222, 152, 83, 22, 221, 155, 80, 23, 220, 154, 81,
24, 211, 149, 94, 25, 210, 148, 95, 26, 209, 151, 92, 27, 208, 150, 93,
28, 215, 145, 90, 29, 214, 144, 91, 30, 213, 147, 88, 31, 212, 146, 89,
32, 235, 173, 102, 33, 234, 172, 103, 34, 233, 175, 100, 35, 232, 174, 101,
36, 239, 169, 98, 37, 238, 168, 99, 38, 237, 171, 96, 39, 236, 170, 97,
40, 227, 165, 110, 41, 226, 164, 111, 42, 225, 167, 108, 43, 224, 166, 109,
44, 231, 161, 106, 45, 230, 160, 107, 46, 229, 163, 104, 47, 228, 162, 105,
48, 251, 189, 118, 49, 250, 188, 119, 50, 249, 191, 116, 51, 248, 190, 117,
52, 255, 185, 114, 53, 254, 184, 115, 54, 253, 187, 112, 55, 252, 186, 113,
56, 243, 181, 126, 57, 242, 180, 127, 58, 241, 183, 124, 59, 240, 182, 125,
60, 247, 177, 122, 61, 246, 176, 123, 62, 245, 179, 120, 63, 244, 178, 121
};

// tables for the S-Box and its inverse
static const uint8_t sbox_table[256] = {
99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118,
202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156, 164, 114, 192,
183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216, 49, 21,
4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117,
9, 131, 44, 26, 27, 110, 90, 160, 82, 59, 214, 179, 41, 227, 47, 132,
83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207,
208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168,
81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 33, 16, 255, 243, 210,
205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115,
96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219,
224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98, 145, 149, 228, 121,
231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8,
186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138,
112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193, 29, 158,
225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223,
140, 161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22
};

static const uint8_t invsbox_table[256] = {
82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243, 215, 251,
124, 227, 57, 130, 155, 47, 255, 135, 52, 142, 67, 68, 196, 222, 233, 203,
84, 123, 148, 50, 166, 194, 35, 61, 238, 76, 149, 11, 66, 250, 195, 78,
8, 46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37,
114, 248, 246, 100, 134, 104, 152, 22, 212, 164, 92, 204, 93, 101, 182, 146,
108, 112, 72, 80, 253, 237, 185, 218, 94, 21, 70, 87, 167, 141, 157, 132,
144, 216, 171, 0, 140, 188, 211, 10, 247, 228, 88, 5, 184, 179, 69, 6,
208, 44, 30, 143, 202, 63, 15, 2, 193, 175, 189, 3, 1, 19, 138, 107,
58, 145, 17, 65, 79, 103, 220, 234, 151, 242, 207, 206, 240, 180, 230, 115,
150, 172, 116, 34, 231, 173, 53, 133, 226, 249, 55, 232, 28, 117, 223, 110,
71, 241, 26, 113, 29, 41, 197, 137, 111, 183, 98, 14, 170, 24, 190, 27,
252, 86, 62, 75, 198, 210, 121, 32, 154, 219, 192, 254, 120, 205, 90, 244,
31, 221, 168, 51, 136, 7, 199, 49, 177, 18, 16, 89, 39, 128, 236, 95,
96, 81, 127, 169, 25, 181, 74, 13, 45, 229, 122, 159, 147, 201, 156, 239,
160, 224, 59, 77, 174, 42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97,
23, 43, 4, 126, 186, 119, 214, 38, 225, 105, 20, 99, 85, 33, 12, 125
};


inline uint8_t times2(uint8_t b) {
    return table2[b];
}

inline uint8_t times3(uint8_t b) {
    return table3[b];
}

inline uint8_t times7(uint8_t b) {
    return table7[b];
}

inline uint8_t times9(uint8_t b) {
    return table9[b];
}

inline uint8_t times11(uint8_t b) {
    return table11[b];
}

inline uint8_t times13(uint8_t b) {
    return table13[b];
}

inline uint8_t times14(uint8_t b) {
    return table14[b];
}

inline uint8_t times68(uint8_t b) {
    return table68[b];
}

inline uint8_t times71(uint8_t b) {
    return table71[b];
}

inline uint8_t times201(uint8_t b) {
    return table201[b];
}

inline uint8_t times203(uint8_t b) {
    return table203[b];
}

inline uint8_t sbox(uint8_t b) {
    return sbox_table[b];
}

inline uint8_t invsbox(uint8_t b) {
    return invsbox_table[b];
}


// round constants of Haraka, as arrays of bytes which allow to fetch easily
// a single byte of constant at a given round
static const uint8_t haraka_rcons_table[40][16] = {
{157,123,129,117,240,254,197,178,10,192,32,230,76,112,132,6}, 
{23,247,8,47,164,107,15,100,107,160,243,136,225,180,102,139}, 
{20,145,2,159,96,157,2,207,152,132,242,83,45,222,2,52}, 
{121,79,91,253,175,188,243,187,8,79,123,46,230,234,214,14}, 
{68,112,57,190,28,205,238,121,139,68,114,72,203,176,207,203}, 
{123,5,138,43,237,53,83,141,183,50,144,110,238,205,234,126}, 
{27,239,79,218,97,39,65,226,208,124,46,94,67,143,194,103}, 
{59,11,199,31,226,253,95,103,7,204,202,175,176,217,36,41}, 
{238,101,212,185,202,143,219,236,233,127,134,230,241,99,77,171}, 
{51,126,3,173,79,64,42,91,100,205,183,212,132,191,48,28}, 
{0,152,246,141,46,139,2,105,191,35,23,148,185,11,204,178}, 
{138,45,157,92,200,158,170,74,114,85,111,222,166,120,4,250}, 
{212,159,18,41,46,79,250,14,18,42,119,107,43,159,180,223}, 
{238,18,106,187,174,17,214,50,54,162,73,244,68,3,161,30}, 
{166,236,168,156,201,0,150,95,132,0,5,75,136,73,4,175}, 
{236,147,229,39,227,199,162,120,79,156,25,157,216,94,2,33}, 
{115,1,212,130,205,46,40,185,183,201,89,167,248,170,58,191}, 
{107,125,48,16,217,239,242,55,23,176,134,97,13,112,96,98}, 
{198,154,252,246,83,145,194,129,67,4,48,33,194,69,202,90}, 
{58,148,209,54,232,146,175,44,187,104,107,34,60,151,35,146}, 
{180,113,16,229,88,185,186,108,235,134,88,34,56,146,191,211}, 
{141,18,225,36,221,253,61,147,119,198,240,174,229,60,134,219}, 
{177,18,34,203,227,141,228,131,156,160,235,255,104,98,96,187}, 
{125,247,43,199,78,26,185,45,156,209,228,226,220,211,75,115}, 
{78,146,179,44,196,21,20,75,67,27,48,97,195,71,187,67}, 
{153,104,235,22,221,49,178,3,246,239,7,231,168,117,167,219}, 
{44,71,202,126,2,35,94,142,119,89,117,60,75,97,243,109}, 
{249,23,134,184,185,229,27,109,119,125,222,214,23,90,167,205}, 
{93,238,70,169,157,6,108,157,170,233,168,107,240,67,107,236}, 
{193,39,243,59,89,17,83,162,43,51,87,249,80,105,30,203}, 
{217,208,14,96,83,3,237,228,156,97,218,0,117,12,238,44}, 
{80,163,164,99,188,186,187,128,171,12,233,150,161,165,177,240}, 
{57,202,141,147,48,222,13,171,136,41,150,94,2,177,61,174}, 
{66,180,117,46,168,243,20,136,11,164,84,213,56,143,187,23}, 
{246,22,10,54,121,183,182,174,215,127,66,95,91,138,187,52}, 
{222,175,186,255,24,89,206,67,56,84,229,203,65,82,246,38}, 
{120,201,158,131,247,156,202,162,106,2,243,185,84,154,233,76}, 
{53,18,144,34,40,110,192,64,190,247,223,27,26,165,81,174}, 
{207,89,166,72,15,188,115,193,43,210,126,186,60,97,193,160}, 
{161,157,197,233,253,189,214,74,136,130,40,2,3,204,106,117}
};

// Returns the byte number b (numbering is from 0 to 15, as in the standard AES
// numbering) of round constant in substate i at round r.
inline uint8_t rccon(int r, int i, int b) {
    return haraka_rcons_table[4*r + i][b];
}



//====================================================
//============= OPERATIONS

// an AES state is represented as an array of 16 bytes
typedef array<uint8_t, 16> state_t;
// a Haraka state is represented as an array of 4 AES states. 
// This is inspired by (and follows from) the reference implementation of
// Haraka in python (see https://github.com/kste/haraka)
typedef array<state_t, 4> hkstate_t;

// XORs the round constant at a given round, for a given substate
inline void xor_rcons(state_t& state, int r, int j) {
    for (uint8_t i = 0; i < 16; i++) {
        state[i] ^= haraka_rcons_table[4*r+j][i];
    }
}

// Applies SB
inline void subbytes(state_t& state) {
    for (uint8_t i = 0; i < 16; i++) {
        state[i] = sbox(state[i]);
    }
}

// Applies SB inverse
inline void invsubbytes(state_t& state) {
    for (uint8_t i = 0; i < 16; i++) {
        state[i] = invsbox(state[i]);
    }
}

// Applies SR
inline void shiftrows(state_t& state) {
    uint8_t tmp;
    tmp           = (state)[1];
    (state)[1] = (state)[5];
    (state)[5] = (state)[9];
    (state)[9] = (state)[13];
    (state)[13] = tmp;
    tmp = (state)[10];
    (state)[10] = (state)[2];
    (state)[2] = tmp;
    tmp = (state)[14];
    (state)[14] = (state)[6];
    (state)[6] = tmp;
    tmp = (state)[15];
    (state)[15] = (state)[11];
    (state)[11] = (state)[7];
    (state)[7] = (state)[3];
    (state)[3] = tmp;
}

// Applies SB and SR (that's a little faster)
inline void sbsr(state_t& state) {
    uint8_t tmp;
    tmp           = sbox(state[1]);
    (state)[1] = sbox((state)[5]);
    (state)[5] = sbox((state)[9]);
    (state)[9] = sbox((state)[13]);
    (state)[13] = tmp;
    tmp = sbox((state)[10]);
    (state)[10] = sbox((state)[2]);
    (state)[2] = tmp;
    tmp = sbox((state)[14]);
    (state)[14] = sbox((state)[6]);
    (state)[6] = tmp;
    tmp = sbox((state)[15]);
    (state)[15] = sbox((state)[11]);
    (state)[11] = sbox((state)[7]);
    (state)[7] = sbox((state)[3]);
    (state)[3] = tmp;
    state[0] = sbox(state[0]);
    state[4] = sbox(state[4]);
    state[8] = sbox(state[8]);
    state[12] = sbox(state[12]);
}

// Applies SR inverse
inline void invshiftrows(state_t& state) {
    uint8_t tmp;
    tmp           = (state)[1];
    (state)[1] = (state)[13];
    (state)[13] = (state)[9];
    (state)[9] = (state)[5];
    (state)[5] = tmp;
    tmp = (state)[10];
    (state)[10] = (state)[2];
    (state)[2] = tmp;
    tmp = (state)[14];
    (state)[14] = (state)[6];
    (state)[6] = tmp;
    tmp = (state)[3];
    (state)[3] = (state)[7];
    (state)[7] = (state)[11];
    (state)[11] = (state)[15];
    (state)[15] = tmp;
}

// Applies SR and SB inverse (that's a little faster)
inline void invsbsr(state_t& state) {
    uint8_t tmp;
    tmp           = invsbox((state)[1]);
    (state)[1] = invsbox((state)[13]);
    (state)[13] = invsbox((state)[9]);
    (state)[9] = invsbox((state)[5]);
    (state)[5] = tmp;
    tmp = invsbox((state)[10]);
    (state)[10] = invsbox((state)[2]);
    (state)[2] = tmp;
    tmp = invsbox((state)[14]);
    (state)[14] = invsbox((state)[6]);
    (state)[6] = tmp;
    tmp = invsbox((state)[3]);
    (state)[3] = invsbox((state)[7]);
    (state)[7] = invsbox((state)[11]);
    (state)[11] = invsbox((state)[15]);
    (state)[15] = tmp;
    state[0] = invsbox(state[0]);
    state[4] = invsbox(state[4]);
    state[8] = invsbox(state[8]);
    state[12] = invsbox(state[12]);
}


// Prints a Haraka state as 4 AES states side by side. The bytes are placed in
// the same way as in the figure given in our Appendix C.
static void print_hkstate(hkstate_t& s) {
    for (uint8_t i = 0; i < 4; i++) {
        for (uint8_t j = 0; j < 4; j++) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << int((s)[j][i]) << " ";
            std::cout << std::hex << std::setw(2) << std::setfill('0') << int((s)[j][i+4]) << " "; 
            std::cout << std::hex << std::setw(2) << std::setfill('0') << int((s)[j][i+8]) << " ";
            std::cout << std::hex << std::setw(2) << std::setfill('0') << int((s)[j][i+12]) << " ";
            std::cout << " ";
        }
        std::cout << std::endl;
    }
}

// Prints a Haraka state as a list of bytes, e.g. and input state. This is intended to
// be directly copied in the python reference implementation, to check the results
// that we obtained.
static void print_hkstate_aspython(hkstate_t& s) {
    for (uint8_t i = 0; i < 4; i++) {
        for (uint8_t j = 0; j < 16; j++) {
            std::cout << "0x"<< std::hex << std::setw(2) << std::setfill('0') << int((s)[i][j]) << ", ";
        }
    }
    std::cout << std::endl;
}

// Applies MC
inline void mixcolumns(state_t& state) {
    uint8_t a,b,c,d;
    for (int i =0; i < 4; i++) {
        a = (state)[0+4*i];
        b = (state)[1+4*i];
        c = (state)[2+4*i];
        d = (state)[3+4*i];
        (state)[0+4*i] = times2(a) ^ times3(b) ^ c ^ d;
        (state)[1+4*i] = a ^ times2(b) ^ times3(c) ^ d;
        (state)[2+4*i] = a ^ b ^ times2(c) ^ times3(d);
        (state)[3+4*i] = times3(a) ^ b ^ c ^ times2(d);
    }
}

// Applies MC and the round constant addition (a little faster)
inline void mcrc(state_t& state, int r, int j) {
    uint8_t a,b,c,d;
    for (int i =0; i < 4; i++) {
        a = (state)[0+4*i];
        b = (state)[1+4*i];
        c = (state)[2+4*i];
        d = (state)[3+4*i];
        (state)[0+4*i] = times2(a) ^ times3(b) ^ c ^ d ^ haraka_rcons_table[4*r+j][0+4*i];
        (state)[1+4*i] = a ^ times2(b) ^ times3(c) ^ d ^ haraka_rcons_table[4*r+j][1 + 4*i];
        (state)[2+4*i] = a ^ b ^ times2(c) ^ times3(d) ^ haraka_rcons_table[4*r+j][2 + 4*i];
        (state)[3+4*i] = times3(a) ^ b ^ c ^ times2(d) ^ haraka_rcons_table[4*r+j][3 + 4*i];
    }
}

// Applies MC inverse
inline void invmixcolumns(state_t& state) {
    uint8_t a,b,c,d;
    for (int i =0; i < 4; i++) {
        a = (state)[0+4*i];
        b = (state)[1+4*i];
        c = (state)[2+4*i];
        d = (state)[3+4*i];
        (state)[0+4*i] = times14(a) ^ times11(b) ^ times13(c) ^ times9(d);
        (state)[1+4*i] = times9(a) ^ times14(b) ^ times11(c) ^times13(d);
        (state)[2+4*i] = times13(a) ^ times9(b) ^ times14(c) ^ times11(d);
        (state)[3+4*i] = times11(a) ^ times13(b) ^ times9(c) ^ times14(d);
    }
}

// Applies MC and round constant addition inverse
inline void invmcrc(state_t& state, int r, int j) {
    uint8_t a,b,c,d;
    for (int i =0; i < 4; i++) {
        a = (state)[0+4*i]^ haraka_rcons_table[4*r+j][0+4*i];
        b = (state)[1+4*i]^ haraka_rcons_table[4*r+j][1+4*i];
        c = (state)[2+4*i]^ haraka_rcons_table[4*r+j][2+4*i];
        d = (state)[3+4*i]^ haraka_rcons_table[4*r+j][3+4*i];
        (state)[0+4*i] = times14(a) ^ times11(b) ^ times13(c) ^ times9(d);
        (state)[1+4*i] = times9(a) ^ times14(b) ^ times11(c) ^times13(d);
        (state)[2+4*i] = times13(a) ^ times9(b) ^ times14(c) ^ times11(d);
        (state)[3+4*i] = times11(a) ^ times13(b) ^ times9(c) ^ times14(d);
    }
}


// Encrypts one round of AES at round r and position j in the Haraka state
inline void aesenc(state_t& s, int r, int j) {
    sbsr(s);
    mcrc(s,r,j);
}

// Decrypts one round of AES at round r and position j in the Haraka state
inline void invaesenc(state_t& s, int r, int j) {
    invmcrc(s,r,j);
    invsbsr(s);
}


// Performs MIX
inline void mix512(hkstate_t& s) {
    uint8_t a;
    for (int i = 0; i < 4; i++) {
        a = (s)[0][4 + i];
        (s)[0][4 + i]  = (s)[2][12 + i]; // 1 = 11
        (s)[2][12 + i] = (s)[1][4 + i]; // 11 = 5
        (s)[1][4 + i]  = (s)[0][0 + i]; // 5 = 0
        (s)[0][0 + i]  = (s)[0][12 + i]; // 0 = 3
        (s)[0][12 + i] = (s)[3][12 + i]; // 3 = 15
        (s)[3][12 + i] = (s)[3][8 + i]; // 15 = 14
        (s)[3][8 + i]  = (s)[1][8 + i]; // 14 = 6
        (s)[1][8 + i]  = (s)[3][0 + i]; // 6 = 12
        (s)[3][0 + i]  = (s)[0][8 + i]; // 12 = 2
        (s)[0][8 + i]  = (s)[1][12 + i]; // 2 = 7
        (s)[1][12+i]   = (s)[1][0+i]; // 7 = 4
        (s)[1][0+i]    = (s)[2][0+i]; // 4 = 8
        (s)[2][0+i]    = (s)[2][4+i]; // 8 = 9
        (s)[2][4+i] = a; // 9 = 1
        a = (s)[2][8 + i]; // 10
        (s)[2][8 + i] = (s)[3][4 + i]; // 10 = 13
        (s)[3][4 + i] = a;
    }
}

// Performs MIX inverse
inline void invmix512(hkstate_t& s) {
    uint8_t a;
    for (int i = 0; i < 4; i++) {
        a = (s)[2][8 + i]; // 10
        (s)[2][8 + i] =  (s)[3][4 + i]; // 10 = 13
        (s)[3][4 + i] = a;
        a = (s)[0][4 + i];
        (s)[0][4 + i] =  (s)[2][4+i]; // 1 = 9
        (s)[2][4+i] =    (s)[2][0+i]; // 9 = 8
        (s)[2][0+i] =    (s)[1][0+i]; // 8 = 4
        (s)[1][0+i] =    (s)[1][12+i]; // 4 = 7
        (s)[1][12 + i] = (s)[0][8 + i]; // 7 = 2
        (s)[0][8 + i] =  (s)[3][0 + i]; // 2 = 12
        (s)[3][0 + i] =  (s)[1][8 + i]; // 12 = 6
        (s)[1][8 + i] =  (s)[3][8 + i]; // 6 = 14
        (s)[3][8 + i] =  (s)[3][12 + i]; // 14 = 15
        (s)[3][12 + i] = (s)[0][12 + i]; // 15 = 3
        (s)[0][12 + i] = (s)[0][0 + i]; // 3 = 0
        (s)[0][0 + i] =  (s)[1][4 + i]; // 0 = 5
        (s)[1][4 + i] =  (s)[2][12 + i]; // 5 = 11
        (s)[2][12 + i] = a; // 11 = 1
    }
}

// Applies Haraka.
inline void haraka512pi(hkstate_t& hkstate) {
    for (int r = 0; r < 10; r++) {
        for (int j = 0; j < 4; j++) {
            aesenc( hkstate[j], r, j);
        }
        if (r % 2 == 1) {
            mix512(hkstate);
        }
    }
}


//====================================================================
//========== FASTER OPERATIONS USING THE AES-NI INSTRUCTIONS


// converts 128i value to a 16-byte array (actually writes in the array passed by reference)
void value_to_state(array<uint8_t, 16>& s, __m128i value) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(s.data()), value);
}

// converts 16-byte array to 128ivalue
__m128i state_to_value(array<uint8_t, 16>& s) {
    return _mm_loadu_si128(reinterpret_cast<__m128i*>(s.data()));
}

// Applies the Haraka-512 permutation. This code is taken from
// the reference implementation of Haraka-512 (except the conversions from
// and back to our representation as byte arrays).
void haraka512pi_opt(array<__m128i,40>& rc, hkstate_t& hkstate) {
    array<__m128i, 4> s;
    __m128i tmp;
    for (int i = 0; i < 4; i++) { s[i] = state_to_value(hkstate[i]); }
    for (int r = 0; r < 10; r++) {
        s[0] = _mm_aesenc_si128(s[0], rc[4*r]);
        s[1] = _mm_aesenc_si128(s[1], rc[4*r + 1]);
        s[2] = _mm_aesenc_si128(s[2], rc[4*r + 2]);
        s[3] = _mm_aesenc_si128(s[3], rc[4*r + 3]);


        if (r % 2 == 1) {
            tmp  = _mm_unpacklo_epi32(s[0], s[1]);
            s[0] = _mm_unpackhi_epi32(s[0], s[1]);
            s[1] = _mm_unpacklo_epi32(s[2], s[3]);
            s[2] = _mm_unpackhi_epi32(s[2], s[3]);
            s[3] = _mm_unpacklo_epi32(s[0], s[2]);
            s[0] = _mm_unpackhi_epi32(s[0], s[2]);
            s[2] = _mm_unpackhi_epi32(s[1],  tmp);
            s[1] = _mm_unpacklo_epi32(s[1],  tmp);
        }
    }
   for (int i = 0; i < 4; i++) { value_to_state(hkstate[i], s[i]); }
}




//===============================================================
//============= IMPLEMENTATION OF OUR ATTACK: PART 1

// Forward computation path, but stopping at x5. Writes the value of x5
// in the state s passed by reference. Only the 'dof' input truly matters here,
// the rest should be constant.
inline void computefwd_x5(array< array<uint8_t, 16>, 4>& s,
        array<uint8_t, 8>& round2_match_through_mc,
        array<uint8_t, 20>& x3_guesses,
        array<uint8_t, 16>& x4_guesses, array<uint8_t, 12>& x5_guesses,
        array<uint8_t, 4>& dof) {

    uint8_t w2a_1 = dof[0]; uint8_t w2a_6 = dof[1]; 
    uint8_t w2a_11 = dof[2] ; uint8_t w2a_12 = dof[3];
    s[0][1] = w2a_1 ^ rccon(2, 0, 1);
    s[0][6] = w2a_6 ^ rccon(2, 0, 6);
    s[0][11] = w2a_11 ^ rccon(2, 0, 11);
    s[0][12] = w2a_12 ^ rccon(2, 0, 12);
    s[0][0] = (times201(round2_match_through_mc[0])
                    ^ times68(round2_match_through_mc[1])
                    ^ times203(w2a_1) ^ rccon(2,0,0) );
    s[0][2] = (times68(round2_match_through_mc[0])
                    ^ times201(round2_match_through_mc[1])
                    ^ times71(w2a_1) ^ rccon(2,0,2) );
    s[0][5] = ( times68(round2_match_through_mc[2] ) 
                   ^ times201(round2_match_through_mc[3] ) ^times203(w2a_6)  ^ rccon(2, 0, 5) );

    s[0][7] = ( times201(round2_match_through_mc[2] ) 
                   ^ times68(round2_match_through_mc[3] ) ^ times71(w2a_6) ^ rccon(2, 0, 7) );
    s[0][8] = ( times201( round2_match_through_mc[4] ) 
                   ^ times68(round2_match_through_mc[5]) ^times71(w2a_11) ^ rccon(2, 0, 8) );
    s[0][10] = ( times68(round2_match_through_mc[4] ) 
                   ^ times201(round2_match_through_mc[5] ) ^times203(w2a_11) ^ rccon(2, 0, 10) );
    s[0][13] = ( times68(round2_match_through_mc[6] ) 
                   ^ times201(round2_match_through_mc[7] ) ^times71(w2a_12) ^ rccon(2, 0, 13) );
    s[0][15] = ( times201(round2_match_through_mc[6] ) 
                   ^ times68(round2_match_through_mc[7] ) ^times203(w2a_12) ^ rccon(2, 0, 15) );
    s[1][1] = x3_guesses[0]; s[1][2] = x3_guesses[1];
    s[1][6] = x3_guesses[2]; s[1][7] = x3_guesses[3];
    s[1][8] = x3_guesses[4]; s[1][11] = x3_guesses[5];
    s[1][12] = x3_guesses[6]; s[1][13] = x3_guesses[7];
    s[2][1] = x3_guesses[8]; s[2][6] = x3_guesses[9];
    s[2][11] = x3_guesses[10]; s[2][12] = x3_guesses[11];
    s[3][0] = x3_guesses[12]; s[3][1] = x3_guesses[13];
    s[3][5] = x3_guesses[14]; s[3][6] = x3_guesses[15];
    s[3][10] = x3_guesses[16]; s[3][11] = x3_guesses[17];
    s[3][12] = x3_guesses[18]; s[3][15] = x3_guesses[19];
    // now contains x3. Apply round 3.
    
    for (int j = 0; j < 4; j++) {  aesenc( s[j], 3, j); }
    mix512(s);
    s[1][0] = x4_guesses[0];  s[1][1] = x4_guesses[1];
    s[1][2] = x4_guesses[2]; s[1][3] = x4_guesses[3];
    s[1][12] = x4_guesses[4]; s[1][13] = x4_guesses[5];
    s[1][14] = x4_guesses[6]; s[1][15] = x4_guesses[7];
    s[3][4] = x4_guesses[8]; s[3][5] = x4_guesses[9];
    s[3][6] =  x4_guesses[10]; s[3][7] =  x4_guesses[11];
    s[3][12] =  x4_guesses[12]; s[3][13] =  x4_guesses[13];
    s[3][14] =  x4_guesses[14]; s[3][15] =  x4_guesses[15];
    // now contains x4. Apply round 4
    
    aesenc( s[0], 4, 0); aesenc( s[1], 4, 1); aesenc( s[3], 4, 3); 
    // only partial computations are necessary
    s[2][1] = x5_guesses[0]; s[2][2] = x5_guesses[1];
    s[2][3] = x5_guesses[2]; s[2][4] = x5_guesses[3];
    s[2][6] = x5_guesses[4]; s[2][7] = x5_guesses[5];
    s[2][8] = x5_guesses[6]; s[2][9] = x5_guesses[7];
    s[2][11] = x5_guesses[8]; s[2][12] = x5_guesses[9];
    s[2][13] = x5_guesses[10]; s[2][14] = x5_guesses[11];
    // now contains x5.
}


// Forward computation path. The result is written in "output_tuple". The state
// s is only used during the computation.
inline void computefwd(array< array<uint8_t, 16>, 4>& s,
    array<uint8_t, 8>& round2_match_through_mc,
         array<uint8_t, 20>& x3_guesses,
        array<uint8_t, 16>& x4_guesses, array<uint8_t, 12>& x5_guesses,
        array<uint8_t, 4>& dof, array<uint8_t, 4>& output_tuple) {
    computefwd_x5(s, round2_match_through_mc,
            x3_guesses, x4_guesses, x5_guesses, dof);
    for (int j = 0; j < 4; j++) {  aesenc( s[j], 5, j); }
    mix512(s); // now x6
    aesenc( s[0], 6, 0); aesenc( s[2], 6, 2); aesenc( s[3], 6, 3); // x7 (partial)
    aesenc( s[0], 7, 0); aesenc( s[2], 7, 2); aesenc( s[3], 7, 3); // x8 (partial)
    mix512(s);
    // apply subbytes and shiftrows in place on s[2] to get z8
    subbytes(s[2]); shiftrows(s[2]);
    output_tuple[0] = times7(s[2][0]) ^ s[2][1] ^ times7(s[2][2]);
    output_tuple[1] = s[2][4] ^ times2(s[2][5]) ^ times3(s[2][7]);
    output_tuple[2] = times7(s[2][8]) ^ times7(s[2][10]) ^ s[2][11];
    output_tuple[3] = times3(s[2][13]) ^ s[2][14] ^ times2(s[2][15]);
}        


// Backward computation path. The result is written in "output_tuple".
inline void computebwd(array< array<uint8_t, 16>, 4>& s,
        array<uint8_t, 8>& round2_match_through_mc,
        array<uint8_t, 20>& x3_guesses,
        array<uint8_t, 16>& x4_guesses, array<uint8_t, 12>& x5_guesses,
        array<uint8_t, 4>& dof, array<uint8_t, 4>& output_tuple) {
    s[2][0] = dof[0]; s[2][5] = dof[1]; s[2][10] = dof[2]; s[2][15] = dof[3];
    s[2][1] = x5_guesses[0]; s[2][2] = x5_guesses[1];
    s[2][3] = x5_guesses[2]; s[2][4] = x5_guesses[3];
    s[2][6] = x5_guesses[4]; s[2][7] = x5_guesses[5];
    s[2][8] = x5_guesses[6]; s[2][9] = x5_guesses[7];
    s[2][11] = x5_guesses[8]; s[2][12] = x5_guesses[9];
    s[2][13] = x5_guesses[10]; s[2][14] = x5_guesses[11]; // now contains x5
    invaesenc( s[2], 4, 2); // only a partial computation is necessary
    s[1][0] = x4_guesses[0];  s[1][1] = x4_guesses[1];
    s[1][2] = x4_guesses[2]; s[1][3] = x4_guesses[3];
    s[1][12] = x4_guesses[4]; s[1][13] = x4_guesses[5];
    s[1][14] = x4_guesses[6]; s[1][15] = x4_guesses[7];
    s[3][4] = x4_guesses[8]; s[3][5] = x4_guesses[9];
    s[3][6] =  x4_guesses[10]; s[3][7] =  x4_guesses[11];
    s[3][12] =  x4_guesses[12]; s[3][13] =  x4_guesses[13];
    s[3][14] =  x4_guesses[14]; s[3][15] =  x4_guesses[15]; // now contains x4
    invmix512(s);
    for (int j = 0; j < 4; j++) {  invaesenc( s[j], 3, j); }
    s[1][1] = x3_guesses[0]; s[1][2] = x3_guesses[1];
    s[1][6] = x3_guesses[2]; s[1][7] = x3_guesses[3];
    s[1][8] = x3_guesses[4]; s[1][11] = x3_guesses[5];
    s[1][12] = x3_guesses[6]; s[1][13] = x3_guesses[7];
    s[2][1] = x3_guesses[8]; s[2][6] = x3_guesses[9];
    s[2][11] = x3_guesses[10]; s[2][12] = x3_guesses[11];
    s[3][0] = x3_guesses[12]; s[3][1] = x3_guesses[13];
    s[3][5] = x3_guesses[14]; s[3][6] = x3_guesses[15];
    s[3][10] = x3_guesses[16]; s[3][11] = x3_guesses[17];
    s[3][12] = x3_guesses[18]; s[3][15] = x3_guesses[19]; // now x3
    uint8_t w2a_3 = s[0][3] ^ rccon(2, 0, 3);
    uint8_t w2a_4 = s[0][4] ^ rccon(2, 0, 4);
    uint8_t w2a_9 = s[0][9] ^ rccon(2, 0, 9);
    uint8_t w2a_14 = s[0][14] ^ rccon(2, 0, 14);
    // create z2a
    s[0][1] = round2_match_through_mc[0]^times13(w2a_3);
    s[0][3] = round2_match_through_mc[1]^times14(w2a_3);
    s[0][4] = round2_match_through_mc[2]^times14(w2a_4);
    s[0][6] = round2_match_through_mc[3]^times13(w2a_4);
    s[0][9] = round2_match_through_mc[4]^times14(w2a_9);
    s[0][11] = round2_match_through_mc[5]^times13(w2a_9);
    s[0][12] =  round2_match_through_mc[6]^times13(w2a_14);
    s[0][14] = round2_match_through_mc[7]^times14(w2a_14);
    invsbsr(s[0]);
    for (int j = 1; j < 4; j++) {  invaesenc( s[j], 2, j); } // now x2
    invmix512(s);
    invaesenc( s[2], 1, 2); invaesenc( s[3], 1, 3);
    invaesenc( s[2], 0, 2); invaesenc( s[3], 0, 3);// now x0 (partial)
    invmix512(s);
    invaesenc( s[2], 9, 2); // now x9 (partial)
    xor_rcons(s[2], 8, 2); // here w8c
    output_tuple[0] = times2(s[2][2]) ^ times3(s[2][3]);
    output_tuple[1] = s[2][4] ^ s[2][7];
    output_tuple[2] = times2(s[2][8]) ^ times3(s[2][9]);
    output_tuple[3] = s[2][13] ^ s[2][14];
}


// From a pair of forward and backward choices that matched, recompute
// x0 and its image. Write x0 in the reference 's' and pi(x0) in the reference 'pix0'.
inline void compute_x0_pix0_from_choices(array<__m128i,40>& rc, 
        array< array<uint8_t, 16>, 4>& s, array< array<uint8_t, 16>, 4>& pix0,
        array<uint8_t, 8>& round2_match_through_mc,
        array<uint8_t, 20>& x3_guesses,
        array<uint8_t, 16>& x4_guesses, array<uint8_t, 12>& x5_guesses,
        array<uint8_t, 4>& fwddof, array<uint8_t, 4>& bwddof) {
    computefwd_x5(s, round2_match_through_mc,
            x3_guesses, x4_guesses, x5_guesses, fwddof);
    s[2][0] = bwddof[0]; s[2][5] = bwddof[1]; s[2][10] = bwddof[2]; s[2][15] = bwddof[3]; // full x5
    __m128i s0 = state_to_value(s[0]);
    __m128i s1 = state_to_value(s[1]);
    __m128i s2 = state_to_value(s[2]);
    __m128i s3 = state_to_value(s[3]);
    __m128i tmp;

    // copy this to pix0
//    for (int i = 0; i < 4; i++) {
//        for (int j = 0; j < 16; j++) {pix0[i][j] = s[i][j];}
//    }
    // now apply inverse rounds to get to x0
    // go to x4
    
    for (int j = 0; j < 4; j++) {  invaesenc( s[j], 4, j); } // x4
    invmix512(s);
    for (int j = 0; j < 4; j++) {  invaesenc( s[j], 3, j); } // x3
    for (int j = 0; j < 4; j++) {  invaesenc( s[j], 2, j); } // x2
    invmix512(s);
    for (int j = 0; j < 4; j++) {  invaesenc( s[j], 1, j); } // x1
    for (int j = 0; j < 4; j++) {  invaesenc( s[j], 0, j); } // x0
    
    
    // apply fwd rounds on the tuple s0, s1, s2, s3 which will give pi(x0)
    // for these lines an optimized implementation of Haraka is used, which
    // comes from the reference. Before that the commented lines were used instead,
    // where we use only our own implementation of Haraka. You can check that this
    // gives the same results. The optimized implementation seems a bit faster in practice.
    s0 = _mm_aesenc_si128(s0, rc[4*5]); s1 = _mm_aesenc_si128(s1, rc[4*5 + 1]);
    s2 = _mm_aesenc_si128(s2, rc[4*5+2]); s3 = _mm_aesenc_si128(s3, rc[4*5 + 3]); 
    // mix
    tmp  = _mm_unpacklo_epi32(s0, s1); s0 = _mm_unpackhi_epi32(s0, s1);
    s1 = _mm_unpacklo_epi32(s2, s3); s2 = _mm_unpackhi_epi32(s2, s3);
    s3 = _mm_unpacklo_epi32(s0, s2); s0 = _mm_unpackhi_epi32(s0, s2);
    s2 = _mm_unpackhi_epi32(s1,  tmp); s1 = _mm_unpacklo_epi32(s1,  tmp);
    s0 = _mm_aesenc_si128(s0, rc[4*6]); s1 = _mm_aesenc_si128(s1, rc[4*6 + 1]);
    s2 = _mm_aesenc_si128(s2, rc[4*6+2]); s3 = _mm_aesenc_si128(s3, rc[4*6 + 3]); 
    s0 = _mm_aesenc_si128(s0, rc[4*7]); s1 = _mm_aesenc_si128(s1, rc[4*7 + 1]);
    s2 = _mm_aesenc_si128(s2, rc[4*7+2]); s3 = _mm_aesenc_si128(s3, rc[4*7 + 3]); 
    tmp  = _mm_unpacklo_epi32(s0, s1); s0 = _mm_unpackhi_epi32(s0, s1);
    s1 = _mm_unpacklo_epi32(s2, s3); s2 = _mm_unpackhi_epi32(s2, s3);
    s3 = _mm_unpacklo_epi32(s0, s2); s0 = _mm_unpackhi_epi32(s0, s2);
    s2 = _mm_unpackhi_epi32(s1,  tmp); s1 = _mm_unpacklo_epi32(s1,  tmp);
    s0 = _mm_aesenc_si128(s0, rc[4*8]); s1 = _mm_aesenc_si128(s1, rc[4*8 + 1]);
    s2 = _mm_aesenc_si128(s2, rc[4*8+2]); s3 = _mm_aesenc_si128(s3, rc[4*8 + 3]); 
    s0 = _mm_aesenc_si128(s0, rc[4*9]); s1 = _mm_aesenc_si128(s1, rc[4*9 + 1]);
    s2 = _mm_aesenc_si128(s2, rc[4*9+2]); s3 = _mm_aesenc_si128(s3, rc[4*9 + 3]); 
    tmp  = _mm_unpacklo_epi32(s0, s1); s0 = _mm_unpackhi_epi32(s0, s1);
    s1 = _mm_unpacklo_epi32(s2, s3); s2 = _mm_unpackhi_epi32(s2, s3);
    s3 = _mm_unpacklo_epi32(s0, s2); s0 = _mm_unpackhi_epi32(s0, s2);
    s2 = _mm_unpackhi_epi32(s1,  tmp); s1 = _mm_unpacklo_epi32(s1,  tmp);
    value_to_state(pix0[0], s0);
    value_to_state(pix0[1], s1);
    value_to_state(pix0[2], s2);
    value_to_state(pix0[3], s3);

    // apply fwd rounds to get to pix0
//    for (int j = 0; j < 4; j++) {  aesenc( pix0[j], 5, j); }
//    mix512(pix0);
//    for (int j = 0; j < 4; j++) {  aesenc( pix0[j], 6, j); }
//    for (int j = 0; j < 4; j++) {  aesenc( pix0[j], 7, j); }
//    mix512(pix0);
//    for (int j = 0; j < 4; j++) {  aesenc( pix0[j], 8, j); }
//    for (int j = 0; j < 4; j++) {  aesenc( pix0[j], 9, j); }
//    mix512(pix0);
}


//===============================================================
//============= IMPLEMENTATION OF OUR ATTACK: PART 2


// We use some multithreading. This mutex objects locks the table when threads
// access it.
std::mutex mtx;


// Populate the table of forward results. The table is passed by reference to be
// accessed by the different threads.
// The argument 'ind' specifies which thread is running (there are 4 of them in total).
void populatetable(array<array<array< vector<array<uint8_t, 5>>, 256>,256>,256>& fwdvecs,
        int ind, uint8_t seed, int size1) {
    // values for the guesses (should match those we give in the other function)
    array<uint8_t, 8> round2_match_through_mc = {0,0,0,0,0,0,0, seed};
    array<uint8_t, 20> x3_guesses = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    array<uint8_t, 16> x4_guesses = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    array<uint8_t, 12> x5_guesses = {0,0,0,0, 0,0,0,0, 0,0,0,0};
    
    // temporary variables
    array<uint8_t, 4> fwddof = {0, 0, 0, 0};
    array< array<uint8_t, 16>, 4> fwds;
    array<uint8_t, 4> res;
    // buffer of results that are stored before we go put them in the shared table
    vector<array<uint8_t, 8>> buffer;
    int bufsize = 0;
    
    for (int i = 0; i < size1; i++) {
        if (i % (1 << 24) == 0) {
            mtx.lock();
            // display some advancement (in log2)
            cout << dec << "thread " << int(ind) << " : " << log(double(i))/log(2);
            cout << " / " << log(double(size1))/log(2) << " fwd done" << endl;
            mtx.unlock();
        }
        
        if (i % 4 == ind){
            // create forward degrees of freedom
            fwddof[0] = uint8_t( (i & 0xFF) );
            fwddof[1] = uint8_t((i >> 8) & 0xFF); 
            fwddof[2] = uint8_t((i >> 16) & 0xFF); 
            fwddof[3] = uint8_t((i >> 24) & 0xFF);
            
            computefwd(fwds, round2_match_through_mc,
             x3_guesses, x4_guesses, x5_guesses, fwddof, res);
            // Using the notation of the paper: we put in 'buffer' the array
            // {m0, m1, m2, m3, f0, f1, f2, f3}
            buffer.push_back( {res[0], res[1], res[2], res[3], 
                                        fwddof[0], fwddof[1], fwddof[2], fwddof[3]} );
            bufsize ++;
        
            // empty the buffer
            if (bufsize > 10000) {
                mtx.lock();
                // using the notation of the paper: we put in the table
                // at index [m0, m1, m2] the array {m3, f0, f1, f2, f3}
                // (A standard PC does not have enough RAM for a table of 2^32 entries)
                for (auto &v: buffer) {
                    fwdvecs[v[0]][v[1]][v[2]].push_back({v[3], 
                                        v[4], v[5], v[6], v[7]} );
                }
                mtx.unlock();
                bufsize = 0;
                buffer.clear();
            }
        }
    }
}


// condition for printing a result (printing some partial results allows
// the user to keep faith)
bool printcond(array< array<uint8_t, 16>, 4>& pix0) {
    return (pix0[2][4] == 0 and pix0[2][7] == 0  
            and  pix0[2][8] == 0 and pix0[2][9] == 0
            and pix0[2][13] == 0 and pix0[2][14] == 0);
}


// condition for having a full result, i.e. two columns to zero
bool fullcond(array< array<uint8_t, 16>, 4>& pix0) {
    return (pix0[2][2] == 0 and pix0[2][3] == 0 and pix0[2][13] == 0 and pix0[2][14] == 0
          and pix0[2][4] == 0 and pix0[2][7] == 0  and  pix0[2][8] == 0 and pix0[2][9] == 0);
}


// Backwards search procedure. Accesses the shared table.
void bwdsearch(array<__m128i,40>& rc, array<array<array< vector<array<uint8_t, 5>>, 256>,256>,256>& fwdvecs,
        int ind, uint8_t seed, uint64_t size2) {

    array<uint8_t, 8> round2_match_through_mc = {0,0,0,0,0,0,0, seed};
    array<uint8_t, 20> x3_guesses = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    array<uint8_t, 16> x4_guesses = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    array<uint8_t, 12> x5_guesses = {0,0,0,0, 0,0,0,0, 0,0,0,0};
    bool verb = true; // if true, will print partial solutions 
                      // (with only 48 bits to zero instead of 64)
                      
    // teomporary variables
    array<uint8_t, 4> bwddof = {0, 0, 0, 0};
    array<uint8_t, 4> fwddof = {0, 0, 0, 0};
    array< array<uint8_t, 16>, 4> bwds;

    array<uint8_t, 4> res;
    array< array<uint8_t, 16>, 4> x0;
    array< array<uint8_t, 16>, 4> pix0;
    int count_pairs = 0;
    uint64_t i;
    int ii, jj;
    
    int bufsize = 0;
    vector<array<uint8_t, 8>> buffer;
    vector<array<uint8_t, 8>> matches;

    for (i = 0; i < size2; i++) {
        if (i % 4 == uint64_t(ind)) {
            bwddof[0] = uint8_t(i & 0xFF); 
            bwddof[1] = uint8_t((i >> 8) & 0xFF); 
            bwddof[2] = uint8_t((i >> 16) & 0xFF); 
            bwddof[3] = uint8_t((i >> 24) & 0xFF);
            
            computebwd(bwds, round2_match_through_mc, x3_guesses, x4_guesses, x5_guesses, bwddof, res);
            // Using the notations of the paper: insert in the buffer the array
            // {m0, m1, m2, m3, b0, b1, b2, b3}
            buffer.push_back( {res[0], res[1], res[2], res[3], 
                                    bwddof[0], bwddof[1], bwddof[2], bwddof[3]} );
            bufsize++;
        }
        if (bufsize > 10000) {
            matches.clear();
            mtx.lock();
            // Access the table now.
            for (auto &v: buffer) {
                for (auto &vv : fwdvecs[v[0]][v[1]][v[2]]) {
                    // Fetch an array at the position [m0][m1][m2]
                    if (vv[0] == v[3]) {
                        // if m3 matches, that is a full match.
                        // create array with dofs of fwd and bwd:
                        matches.push_back({vv[1], vv[2], vv[3], vv[4], v[4], v[5], v[6], v[7]});
                    }
                }
            }
            mtx.unlock();
            buffer.clear();
            bufsize = 0;
            // After accessing the table, we release it, and now we compute the
            // matches.
            for(auto &v: matches) {
                count_pairs++;
                
                if (count_pairs % (1 << 22) == 0) {
                    mtx.lock();
                    cout << dec << "thread " << int(ind) << " : ";
                    cout << log(double(count_pairs))/log(2) <<  " pairs done" << endl;
                    mtx.unlock();
                }
                // read dofs of fwd and bwd
                for (ii = 0; ii < 4; ii++) {fwddof[ii] = v[ii];}
                for (ii = 0; ii < 4; ii++) {bwddof[ii] = v[ii+4];}
                // Compute x0 and its image
                compute_x0_pix0_from_choices(rc, x0, pix0, round2_match_through_mc,
                     x3_guesses, x4_guesses, x5_guesses, fwddof, bwddof);                
                
                for (ii = 0; ii < 4; ii++) {
                    for (jj=0; jj < 16; jj++) {pix0[ii][jj] ^= x0[ii][jj];}
                }
                
                invmix512(pix0);
                for (ii = 0; ii < 4; ii++) {
                    invmixcolumns(pix0[ii]); invshiftrows(pix0[ii]);
                }
                // check!
                if  ( (printcond(pix0) and verb) or fullcond(pix0) ) {
                    mtx.lock();
                    if (fullcond(pix0)) {
                        cout << "!!!!!!!!!!!!!! ======= FULL SOLUTION: " << endl;
                        cout << "Seed: " << dec << int(seed) << endl;
                    }
                    
                    cout << "============ Solution: " << endl;
                    cout << "x0: " << endl;
                    print_hkstate(x0);
                    print_hkstate_aspython(x0);
                    cout << "MC-1(sum):" << endl;
                    // re-apply the mix and shiftrows that we un-applied earlier
                    for (ii = 0; ii < 4; ii++) { shiftrows(pix0[ii]); }
                    mix512(pix0);
                    print_hkstate(pix0);
                    mtx.unlock();
                }
            }
        }
    }
}

// The full search procedure. Uses less than 8GB of RAM. A single run requires
// less than one hour on a laptop; 2^3 = 8 runs will be necessary on average
// to obtain a full solution.
static void search(array<__m128i,40>& rc, uint8_t seed) {
    
    int size1 = (1 << 29); // size of the table
    uint64_t size2 = 4294967295; // search space for the second step (maximal size = 2^32 - 1)
    
    static array<array<array< vector<array<uint8_t, 5>>, 256>,256>,256> fwdvecs;
    using clock = std::chrono::system_clock;
    using sec = std::chrono::duration<double>;
    thread th1, th2, th3, th4;


    auto t1 = clock::now(); sec duration;
    
    th1 = thread(populatetable, std::ref(fwdvecs), 0, seed, size1);
    th2 = thread(populatetable, std::ref(fwdvecs), 1, seed, size1);
    th3 = thread(populatetable, std::ref(fwdvecs), 2, seed, size1);
    th4 = thread(populatetable, std::ref(fwdvecs), 3, seed, size1);
    th1.join(); th2.join(); th3.join();th4.join();
    
    duration = clock::now() - t1;
    std::cout << "Time of first step (seconds): " << duration.count() << std::endl;
    t1 = clock::now();

    th1 = thread(bwdsearch, std::ref(rc), std::ref(fwdvecs), 0, seed, size2);
    th2 = thread(bwdsearch, std::ref(rc), std::ref(fwdvecs), 1, seed, size2);
    th3 = thread(bwdsearch, std::ref(rc), std::ref(fwdvecs), 2, seed, size2);
    th4 = thread(bwdsearch, std::ref(rc), std::ref(fwdvecs), 3, seed, size2);
    th1.join(); th2.join(); th3.join();th4.join();


    duration = clock::now() - t1;
    std::cout << "Time of second step (seconds): " << duration.count() << std::endl;
}



int main(int argc, char *argv[]) {
    
    // from haraka ref implementation
    array<__m128i,40> rc;
    // s[4], tmp, rc[40];

    // round constants used in Haraka. This is required for the optimized
    // implementation. Our own byte-array implementation does not require that.
    rc[0] = _mm_set_epi32(0x0684704c,0xe620c00a,0xb2c5fef0,0x75817b9d);
    rc[1] = _mm_set_epi32(0x8b66b4e1,0x88f3a06b,0x640f6ba4,0x2f08f717);
    rc[2] = _mm_set_epi32(0x3402de2d,0x53f28498,0xcf029d60,0x9f029114);
    rc[3] = _mm_set_epi32(0x0ed6eae6,0x2e7b4f08,0xbbf3bcaf,0xfd5b4f79);
    rc[4] = _mm_set_epi32(0xcbcfb0cb,0x4872448b,0x79eecd1c,0xbe397044);
    rc[5] = _mm_set_epi32(0x7eeacdee,0x6e9032b7,0x8d5335ed,0x2b8a057b);
    rc[6] = _mm_set_epi32(0x67c28f43,0x5e2e7cd0,0xe2412761,0xda4fef1b);
    rc[7] = _mm_set_epi32(0x2924d9b0,0xafcacc07,0x675ffde2,0x1fc70b3b);
    rc[8] = _mm_set_epi32(0xab4d63f1,0xe6867fe9,0xecdb8fca,0xb9d465ee);
    rc[9] = _mm_set_epi32(0x1c30bf84,0xd4b7cd64,0x5b2a404f,0xad037e33);
    rc[10] = _mm_set_epi32(0xb2cc0bb9,0x941723bf,0x69028b2e,0x8df69800);
    rc[11] = _mm_set_epi32(0xfa0478a6,0xde6f5572,0x4aaa9ec8,0x5c9d2d8a);
    rc[12] = _mm_set_epi32(0xdfb49f2b,0x6b772a12,0x0efa4f2e,0x29129fd4);
    rc[13] = _mm_set_epi32(0x1ea10344,0xf449a236,0x32d611ae,0xbb6a12ee);
    rc[14] = _mm_set_epi32(0xaf044988,0x4b050084,0x5f9600c9,0x9ca8eca6);
    rc[15] = _mm_set_epi32(0x21025ed8,0x9d199c4f,0x78a2c7e3,0x27e593ec);
    rc[16] = _mm_set_epi32(0xbf3aaaf8,0xa759c9b7,0xb9282ecd,0x82d40173);
    rc[17] = _mm_set_epi32(0x6260700d,0x6186b017,0x37f2efd9,0x10307d6b);
    rc[18] = _mm_set_epi32(0x5aca45c2,0x21300443,0x81c29153,0xf6fc9ac6);
    rc[19] = _mm_set_epi32(0x9223973c,0x226b68bb,0x2caf92e8,0x36d1943a);
    rc[20] = _mm_set_epi32(0xd3bf9238,0x225886eb,0x6cbab958,0xe51071b4);
    rc[21] = _mm_set_epi32(0xdb863ce5,0xaef0c677,0x933dfddd,0x24e1128d);
    rc[22] = _mm_set_epi32(0xbb606268,0xffeba09c,0x83e48de3,0xcb2212b1);
    rc[23] = _mm_set_epi32(0x734bd3dc,0xe2e4d19c,0x2db91a4e,0xc72bf77d);
    rc[24] = _mm_set_epi32(0x43bb47c3,0x61301b43,0x4b1415c4,0x2cb3924e);
    rc[25] = _mm_set_epi32(0xdba775a8,0xe707eff6,0x03b231dd,0x16eb6899);
    rc[26] = _mm_set_epi32(0x6df3614b,0x3c755977,0x8e5e2302,0x7eca472c);
    rc[27] = _mm_set_epi32(0xcda75a17,0xd6de7d77,0x6d1be5b9,0xb88617f9);
    rc[28] = _mm_set_epi32(0xec6b43f0,0x6ba8e9aa,0x9d6c069d,0xa946ee5d);
    rc[29] = _mm_set_epi32(0xcb1e6950,0xf957332b,0xa2531159,0x3bf327c1);
    rc[30] = _mm_set_epi32(0x2cee0c75,0x00da619c,0xe4ed0353,0x600ed0d9);
    rc[31] = _mm_set_epi32(0xf0b1a5a1,0x96e90cab,0x80bbbabc,0x63a4a350);
    rc[32] = _mm_set_epi32(0xae3db102,0x5e962988,0xab0dde30,0x938dca39);
    rc[33] = _mm_set_epi32(0x17bb8f38,0xd554a40b,0x8814f3a8,0x2e75b442);
    rc[34] = _mm_set_epi32(0x34bb8a5b,0x5f427fd7,0xaeb6b779,0x360a16f6);
    rc[35] = _mm_set_epi32(0x26f65241,0xcbe55438,0x43ce5918,0xffbaafde);
    rc[36] = _mm_set_epi32(0x4ce99a54,0xb9f3026a,0xa2ca9cf7,0x839ec978);
    rc[37] = _mm_set_epi32(0xae51a51a,0x1bdff7be,0x40c06e28,0x22901235);
    rc[38] = _mm_set_epi32(0xa0c1613c,0xba7ed22b,0xc173bc0f,0x48a659cf);
    rc[39] = _mm_set_epi32(0x756acc03,0x02288288,0x4ad6bdfd,0xe9c59da1);
    
    
    search(rc, 53); // turned out to be a good "seed". After some time, will 
    // output the result given in the paper (you can then terminate the process).


    return 0;
}
