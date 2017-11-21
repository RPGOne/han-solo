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
#ifndef GUARD_MIOPEN_HIPOC_PROGRAM_HPP
#define GUARD_MIOPEN_HIPOC_PROGRAM_HPP

#include <hip/hip_runtime_api.h>
#include <miopen/manage_ptr.hpp>
#include <string>

namespace miopen {

using hipModulePtr = MIOPEN_MANAGE_PTR(hipModule_t, hipModuleUnload);
struct HIPOCProgram
{
    using SharedModulePtr = std::shared_ptr<typename std::remove_pointer<hipModule_t>::type>;
    using FilePtr         = MIOPEN_MANAGE_PTR(FILE*, std::fclose);
    HIPOCProgram();
    HIPOCProgram(const std::string& program_name, std::string params, bool is_kernel_str);
    SharedModulePtr module;
    std::string name;
};
} // namespace miopen

#endif
