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
#include "test.hpp"
#include <array>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/activ.hpp>
#include <miopen/miopen.h>
#include <miopen/stringutils.hpp>
#include <miopen/tensor.hpp>
#include <utility>

#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

std::string to_name(miopenActivationMode_t m)
{
#define STRING_CASE(x) \
    case x: return #x; break;
    switch(m)
    {
        STRING_CASE(miopenActivationPATHTRU)
        STRING_CASE(miopenActivationLOGISTIC)
        STRING_CASE(miopenActivationTANH)
        STRING_CASE(miopenActivationRELU)
        STRING_CASE(miopenActivationSOFTRELU)
        STRING_CASE(miopenActivationABS)
        STRING_CASE(miopenActivationPOWER)
    }
    return "";
}

template <class T>
struct verify_forward_activation
{
    tensor<T> input;
    miopen::ActivationDescriptor desc;

    template <class A>
    tensor<T> cpu(A a)
    {
        auto out = input;

        input.par_for_each(
            [&](int o, int w, int i, int j) { out(o, w, i, j) = a(input(o, w, i, j)); });

        return out;
    }

    template <class A>
    tensor<T> gpu(A)
    {
        auto&& handle = get_handle();
        auto out      = input;
        auto in_dev   = handle.Write(input.data);
        auto out_dev  = handle.Write(out.data);

        int alpha = 1, beta = 1;

        desc.Forward(handle, &alpha, input.desc, in_dev.get(), &beta, out.desc, out_dev.get());

        out.data = handle.Read<T>(out_dev, out.data.size());
        return out;
    }

    template <class A>
    void fail(float, A)
    {
        std::cout << "Forward Activation: " << to_name(desc.GetMode()) << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

template <class T>
struct verify_backwards_activation
{
    tensor<T> input;
    tensor<T> dout;
    tensor<T> out;
    miopen::ActivationDescriptor desc;

    template <class A>
    tensor<T> cpu(A a)
    {
        auto dinput = input;

        input.par_for_each([&](int o, int w, int i, int j) {
            dinput(o, w, i, j) = a(dout(o, w, i, j), input(o, w, i, j), out(o, w, i, j));
        });

        return dinput;
    }

    template <class A>
    tensor<T> gpu(A)
    {
        auto&& handle = get_handle();
        auto dinput   = input;

        auto in_dev   = handle.Write(input.data);
        auto dout_dev = handle.Write(dout.data);
        auto out_dev  = handle.Write(out.data);
        auto din_dev  = handle.Write(dinput.data);

        int alpha = 1, beta = 1;

        desc.Forward(handle, &alpha, input.desc, in_dev.get(), &beta, out.desc, out_dev.get());
        desc.Backward(handle,
                      &alpha,
                      // y
                      out.desc,
                      out_dev.get(),
                      // dy
                      dout.desc,
                      dout_dev.get(),
                      // x
                      input.desc,
                      in_dev.get(),
                      &beta,
                      // dx
                      dinput.desc,
                      din_dev.get());

        dinput.data = handle.Read<T>(din_dev, dinput.data.size());
        return dinput;
    }

    template <class A>
    void fail(float, A)
    {
        std::cout << "Backwards Activation: " << to_name(desc.GetMode()) << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

struct select_first
{
    template <class T>
    auto operator()(const T& x) MIOPEN_RETURNS(x.first);
};

template <class T>
struct activation_driver : test_driver
{
    tensor<T> input;
    double alpha     = 1;
    double beta      = 1;
    double power     = 1;
    std::string mode = "PATHTRU";
    std::unordered_map<std::string, std::function<void()>> lookup;

    template <class A>
    struct callback
    {
        void operator()(activation_driver* self) const { self->template run<A>(); }
    };

    template <class Forward, class Backward>
    void add_mode(miopenActivationMode_t m, Forward f, Backward b)
    {
        lookup.emplace(transform_mode(to_name(m)), [=] { this->run(m, f, b); });
    }

    activation_driver()
    {
        add_mode(miopenActivationPATHTRU, [=](T x) { return x; }, [=](T, T x, T) { return x; });
        add_mode(miopenActivationLOGISTIC,
                 [=](T x) { return 1 / (1 + std::exp(-x)); },
                 [=](T dy, T, T y) { return dy * y * (1 - y); });
        add_mode(miopenActivationTANH,
                 [=](T x) { return alpha * std::tanh(beta * x); },
                 [=](T dy, T, T y) { return dy * (1 - y * y); });
        add_mode(miopenActivationRELU,
                 [=](T x) { return (x > 0) ? x : x * beta; },
                 [=](T dy, T, T) { return std::max<T>(0, dy); });
        add_mode(miopenActivationSOFTRELU,
                 [=](T x) { return std::log(1 + std::exp(x)); },
                 [=](T dy, T x, T) {
                     static const float threshold = 50.;
                     T expval = std::exp(std::min(x, static_cast<T>(threshold)));
                     return dy * expval / (expval + 1.0);
                 });
        add_mode(miopenActivationABS,
                 [=](T x) { return std::abs(x); },
                 [=](T dy, T x, T) { return dy * ((x >= 0) ? 1 : -1); });
        add_mode(miopenActivationPOWER,
                 [=](T x) { return std::pow(alpha + beta * x, power); },
                 [=](T, T x, T y) {
                     auto divisor = alpha + beta * x;
                     return (miopen::float_equal(divisor, 0)) ? 0 : alpha * beta * y / divisor;
                 });
        add(input, "input", get_input_tensor());
        add(alpha, "alpha");
        add(beta, "beta");
        add(power, "power");
        add(mode, "mode", generate_data(modes()));
    }

    std::vector<std::string> modes()
    {
        std::vector<std::string> result(lookup.size());
        std::transform(lookup.begin(), lookup.end(), result.begin(), select_first{});
        return result;
    }

    miopen::ActivationDescriptor make_descriptor(miopenActivationMode_t m) const
    {
        return {m, alpha, beta, power};
    }

    static std::string transform_mode(std::string s)
    {
        return miopen::RemovePrefix(miopen::ToUpper(s), "MIOPENACTIVATION");
    }

    void run() { lookup[transform_mode(mode)](); }

    template <class Forward, class Backward>
    void run(miopenActivationMode_t m, Forward f, Backward b)
    {
        auto desc = make_descriptor(m);
        auto out  = verify(verify_forward_activation<T>{input, desc}, f);
        auto dout = out.first;
        dout.generate([&](int n, int c, int h, int w) {
            T x      = out.first(n, c, h, w);
            double y = (877 * n + 547 * c + 701 * h + 1049 * w + static_cast<int>(769 * x)) % 2503;
            return ((x * y) / 1301.0);
        });
        verify(verify_backwards_activation<T>{input, dout, out.first, desc}, b);
    }
};

int main(int argc, const char* argv[]) { test_drive<activation_driver<float>>(argc, argv); }
