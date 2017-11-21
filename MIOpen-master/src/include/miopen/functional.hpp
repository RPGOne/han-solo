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
#ifndef GUARD_MLOPEN_FUNCTIONAL_HPP
#define GUARD_MLOPEN_FUNCTIONAL_HPP

#include <miopen/each_args.hpp>
#include <miopen/returns.hpp>
#include <utility>

namespace miopen {
namespace detail {

template <class F, std::size_t... Ns>
auto each_i_impl(F f, seq<Ns...>) MIOPEN_RETURNS(f(std::integral_constant<std::size_t, Ns>{}...));
}

template <class F, class P>
struct by_t
{
    F f;
    P p;
    template <class... Ts>
    auto operator()(Ts&&... xs) const MIOPEN_RETURNS(f(p(std::forward<Ts>(xs))...))
};

template <class F, class P>
by_t<F, P> by(F f, P p)
{
    return {std::move(f), std::move(p)};
}

template <class F>
struct sequence_t
{
    F f;
    template <class IntegralConstant>
    auto operator()(IntegralConstant) const
        MIOPEN_RETURNS(detail::each_i_impl(f,
                                           typename detail::gens<IntegralConstant::value>::type{}));
};

template <class F>
sequence_t<F> sequence(F f)
{
    return {std::move(f)};
}

} // namespace miopen

#endif
