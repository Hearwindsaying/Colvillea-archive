#pragma once
 // Copyright (c) 2012 Leonhard Gruenschloss (leonhard@gruenschloss.org)
 //
 // Permission is hereby granted, free of charge, to any person obtaining a copy
 // of this software and associated documentation files (the "Software"), to deal
 // in the Software without restriction, including without limitation the rights
 // to
 // use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 // of the Software, and to permit persons to whom the Software is furnished to
 // do
 // so, subject to the following conditions:
 //
 // The above copyright notice and this permission notice shall be included in
 // all copies or substantial portions of the Software.
 //
 // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 // SOFTWARE.

 // These matrices are based on the following publication:
 //
 // S. Joe and F. Y. Kuo: "Constructing Sobol sequences with better
 // two-dimensional projections", SIAM J. Sci. Comput. 30, 2635-2654 (2008).
 //
 // The tabulated direction numbers are available here:
 // http://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201

#include <stdint.h>

struct SobolMatrix
{
	static constexpr int NumSobolDimensions = 1024;
	static constexpr int SobolMatrixSize = 52;
	static const uint32_t SobolMatrices32[NumSobolDimensions * SobolMatrixSize];
	static const uint64_t VdCSobolMatrices[][SobolMatrixSize];
	static const uint64_t VdCSobolMatricesInv[][SobolMatrixSize];
};
