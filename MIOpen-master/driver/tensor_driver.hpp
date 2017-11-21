/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_MIOPEN_TENSOR_DRIVER_HPP
#define GUARD_MIOPEN_TENSOR_DRIVER_HPP

#include <algorithm>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_extra.hpp>
#include <numeric>
#include <vector>

std::vector<int> GetTensorLengths(miopenTensorDescriptor_t& tensor)
{
    int n;
    int c;
    int h;
    int w;

    miopenGet4dTensorDescriptorLengths(tensor, &n, &c, &h, &w);

    return std::vector<int>({n, c, h, w});
}

std::vector<int> GetTensorStrides(miopenTensorDescriptor_t& tensor)
{
    int nstride;
    int cstride;
    int hstride;
    int wstride;

    miopenGet4dTensorDescriptorStrides(tensor, &nstride, &cstride, &hstride, &wstride);

    return std::vector<int>({nstride, cstride, hstride, wstride});
}

int SetTensor4d(miopenTensorDescriptor_t t, std::vector<int>& len)
{
    return miopenSet4dTensorDescriptor(t, miopenFloat, UNPACK_VEC4(len));
}

size_t GetTensorSize(miopenTensorDescriptor_t& tensor)
{
    std::vector<int> len = GetTensorLengths(tensor);
    size_t sz            = std::accumulate(len.begin(), len.end(), 1, std::multiplies<int>());

    return sz;
}
#endif // GUARD_MIOPEN_TENSOR_DRIVER_HPP
