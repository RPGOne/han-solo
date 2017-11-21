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
#include <miopen/batch_norm.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <utility>

#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <cmath>
#include <ctime>
#include <iomanip>

// Run CPU emulations in hierarchical reduction mode.
//#define MIO_HEIRARCH_SEL 0
#define MIO_BN_TEST_EXPAVGFACTOR 0.1
#define MIO_BN_TEST_EPSILON 1e-6

//****************************************************
// FORWARD TRAIN
//****************************************************
template <class T>
struct verify_forward_train_bn_per_activation
{

    const tensor<T> input;
    const tensor<T> scale;
    const tensor<T> shift;

    std::tuple<tensor<T>, tensor<T>, tensor<T>, tensor<T>, tensor<T>> cpu()
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        double epsilon      = MIO_BN_TEST_EPSILON;
        double expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;

        int n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tie4(input.desc.GetLengths());

        auto out = tensor<T>{n_batch, channels, height, width};
        std::fill(out.begin(), out.end(), 0);

        auto runMean    = tensor<T>{1, channels, height, width}.generate(rand_gen{});
        auto runVar     = tensor<T>{1, channels, height, width}.generate(rand_gen{});
        auto saveMean   = tensor<T>{1, channels, height, width};
        auto saveInvVar = tensor<T>{1, channels, height, width};
        const auto n    = double(n_batch);

        par_for(channels, 1, [&](int cidx) {

            double mean_accum     = 0.;
            double variance_accum = 0.;
            double elemStd        = 0.;
            double elemInvVar     = 0.;
            double inhat          = 0.;
            double newRunMean     = 0.;
            double adjust         = 0.;

            // process the batch per channel
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns

                    mean_accum = 0.;
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // #1 calculate the mean :: iterating through the stack of images in the
                        // mini_batch
                        mean_accum += input(bidx, cidx, row, column);
                    }
                    mean_accum /= n;

                    elemStd = variance_accum = 0.;
                    // #2 calculate the variances :: sigma^2 = (1/batch_mean) * sum( (x_i -
                    // batch_mean)^2 )
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        elemStd = (input(bidx, cidx, row, column) -
                                   mean_accum); // (x_i - mean) //this is reused but needs recalc
                        variance_accum += elemStd * elemStd; // sum{ (x_i - mean)^2 }
                    }                                        // end for(n)
                    variance_accum /= n;                     // (1/N)*sum{ (x_i - mean)^2 }

                    // #3 add epsilon for numeric stability, sqr_root, and invert
                    elemInvVar = 1.0 / double(sqrt(variance_accum + epsilon));

                    // #4 apply the normalization :: x_hat = (x_i - mean) / sqrt(variance_accum -
                    // epsilon)
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    {                                                            // via mini_batch
                        elemStd = (input(bidx, cidx, row, column) - mean_accum); // (x_i - mean)
                        inhat   = elemStd * elemInvVar;
                        // #5 Gamma and Beta adjust :: y_i = gamma*x_hat + beta
                        out(bidx, cidx, row, column) =
                            scale(0, cidx, row, column) * inhat + shift(0, cidx, row, column);
                    } // end for(n_batch)

                    newRunMean = runMean(0, cidx, row, column) * (1.0 - expAvgFactor);
                    runMean(0, cidx, row, column) =
                        mean_accum * expAvgFactor + newRunMean; // newMean*factor + tmp

                    // var(n+1) = p * var(n-1) + (1 - p)*(b/b-1)*var(n)
                    adjust = (n_batch == 1) ? variance_accum : (n / (n - 1.0)) * variance_accum;
                    runVar(0, cidx, row, column) =
                        expAvgFactor * runVar(0, cidx, row, column) + (1 - expAvgFactor) * adjust;

                    saveMean(0, cidx, row, column)   = mean_accum;
                    saveInvVar(0, cidx, row, column) = elemInvVar;

                } // for (column)
            }     // for (row)
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_train_bn_per_activation pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

        return std::make_tuple(out, runMean, runVar, saveMean, saveInvVar);
    }

    std::tuple<tensor<T>, tensor<T>, tensor<T>, tensor<T>, tensor<T>> gpu()
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        int n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tie4(input.desc.GetLengths());

        auto out = input;
        std::fill(out.begin(), out.end(), 0);
        auto runMean    = tensor<T>{1, channels, height, width}.generate(rand_gen{});
        auto runVar     = tensor<T>{1, channels, height, width}.generate(rand_gen{});
        auto saveMean   = tensor<T>{1, channels, height, width};
        auto saveInvVar = tensor<T>{1, channels, height, width};

        // in buffers
        auto in_dev    = handle.Write(input.data);
        auto scale_dev = handle.Write(scale.data);
        auto shift_dev = handle.Write(shift.data);

        // out buffers
        auto runMean_dev    = handle.Write(runMean.data);
        auto runVar_dev     = handle.Write(runVar.data);
        auto saveMean_dev   = handle.Create<T>(channels * height * width);
        auto saveInvVar_dev = handle.Create<T>(channels * height * width);
        auto out_dev        = handle.Create<T>(n_batch * channels * height * width);

        double epsilon      = MIO_BN_TEST_EPSILON;
        double expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;

        int alpha = 1, beta = 1;

        miopen::BatchNormForwardTraining(handle,
                                         miopenBNPerActivation,
                                         &alpha,
                                         &beta,
                                         input.desc,
                                         in_dev.get(),
                                         out.desc,
                                         out_dev.get(),
                                         scale.desc,
                                         scale_dev.get(),
                                         shift_dev.get(),
                                         expAvgFactor,
                                         runMean_dev.get(),
                                         runVar_dev.get(),
                                         epsilon,
                                         saveMean_dev.get(),
                                         saveInvVar_dev.get());

        saveMean.data   = handle.Read<T>(saveMean_dev, saveMean.data.size());
        saveInvVar.data = handle.Read<T>(saveInvVar_dev, saveInvVar.data.size());
        runMean.data    = handle.Read<T>(runMean_dev, runMean.data.size());
        runVar.data     = handle.Read<T>(runVar_dev, runVar.data.size());
        out.data        = handle.Read<T>(out_dev, out.data.size());

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_train_bn_per_activation pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

        return std::make_tuple(out, runMean, runVar, saveMean, saveInvVar);
    }

    void fail(int badtensor)
    {
        std::cout << "Forward Train Per Activation Batch Normalization: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
        switch(badtensor)
        {
        case(0): std::cout << "Output tensor output failed verification." << std::endl; break;
        case(1): std::cout << "Running Mean output tensor failed verification." << std::endl; break;
        case(2):
            std::cout << "Running Variance output tensor failed verification." << std::endl;
            break;
        case(3): std::cout << "Saved Mean tensor failed verification." << std::endl; break;
        case(4): std::cout << "Saved Variance tensor failed verification." << std::endl; break;
        }
    }
};

//****************************************************
// FORWARD INFERENCE
//****************************************************
template <class T>
struct verify_forward_infer_bn_per_activation_recalc
{

    const tensor<T> input;
    const tensor<T> scale;
    const tensor<T> shift;

    tensor<T> cpu()
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        double epsilon = MIO_BN_TEST_EPSILON;

        int n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tie4(input.desc.GetLengths());

        auto out = tensor<T>{n_batch, channels, height, width};
        std::fill(out.begin(), out.end(), 0);

        const auto n = double(n_batch);

        par_for(channels, 1, [&](int cidx) {

            double elemStd        = 0.;
            double elemInvVar     = 0.;
            double mean_accum     = 0.;
            double variance_accum = 0.;
            double inhat          = 0.;

            // process the batch per channel
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    mean_accum = 0.;

                    // #1 calculate the mean
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // iterating through the stack of images in the mini_batch
                        mean_accum += input(bidx, cidx, row, column);
                    }
                    mean_accum /= n;

                    elemStd        = 0.;
                    variance_accum = 0.;
                    // #2 calculate the variances
                    // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    {                                                          // via mini_batch
                        elemStd = input(bidx, cidx, row, column) - mean_accum; // (x_i - mean)
                        variance_accum += elemStd * elemStd; // sum{ (x_i - mean)^2 }
                    }                                        // end for(n)
                    variance_accum /= n;                     // (1/N)*sum{ (x_i - mean)^2 }

                    // #3 add epsilon for numeric stability, sqr_root, and invert
                    elemInvVar = 1.0 / double(sqrt(variance_accum + epsilon));

                    // #4 apply the normalization
                    // x_hat = (x_i - mean) / sqrt(variance_accum - epsilon)
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // per (x-dims) channel load a block of data into LDS
                        elemStd = input(bidx, cidx, row, column) - mean_accum; // (x_i - mean)
                        inhat   = elemStd * elemInvVar;
                        // #5 Gamma and Beta adjust // y_i = gamma*x_hat + beta
                        out(bidx, cidx, row, column) =
                            scale(0, cidx, row, column) * inhat + shift(0, cidx, row, column);
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_infer_bn_per_activation_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    tensor<T> gpu()
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();
        auto out      = input;
        std::fill(out.begin(), out.end(), 0);

        auto in_dev    = handle.Write(input.data);
        auto scale_dev = handle.Write(scale.data);
        auto shift_dev = handle.Write(shift.data);
        auto out_dev   = handle.Write(out.data);

        double epsilon = MIO_BN_TEST_EPSILON;

        int alpha = 1, beta = 1;

        miopen::BatchNormForwardInference(handle,
                                          miopenBNPerActivation,
                                          &alpha,
                                          &beta,
                                          input.desc,
                                          in_dev.get(),
                                          out.desc,
                                          out_dev.get(),
                                          scale.desc,
                                          scale_dev.get(),
                                          shift_dev.get(),
                                          nullptr,
                                          nullptr,
                                          epsilon);
        out.data = handle.Read<T>(out_dev, out.data.size());

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer_bn_per_activation_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    void fail(int)
    {
        std::cout << "Forward Inference Per Activation Batch Normalization Recalc: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

template <class T>
struct verify_forward_infer_bn_per_activation_use_est
{

    const tensor<T> input;
    const tensor<T> scale;
    const tensor<T> shift;
    const tensor<T> estMean;
    const tensor<T> estVar;

    tensor<T> cpu()
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        double epsilon = MIO_BN_TEST_EPSILON;

        int n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tie4(input.desc.GetLengths());

        auto out = tensor<T>{n_batch, channels, height, width};
        std::fill(out.begin(), out.end(), 0);

        par_for(channels, 1, [&](int cidx) {
            double elemStd    = 0.;
            double mean       = 0.;
            double variance   = 0.;
            double inhat      = 0.;
            double elemInvVar = 0.;

            // process the batch per channel
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    mean       = estMean(0, cidx, row, column);
                    variance   = estVar(0, cidx, row, column);
                    elemInvVar = 1.0 / double(sqrt(variance + epsilon));
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    {                                                    // via mini_batch
                        elemStd = input(bidx, cidx, row, column) - mean; // (x_i - mean)
                        inhat   = elemStd * elemInvVar;
                        // #5 Gamma and Beta adjust :: y_i = gamma*x_hat + beta
                        out(bidx, cidx, row, column) =
                            scale(0, cidx, row, column) * inhat + shift(0, cidx, row, column);
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_infer_bn_per_activation_use_est pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    tensor<T> gpu()
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();
        auto out      = input;
        std::fill(out.begin(), out.end(), 0);

        auto in_dev      = handle.Write(input.data);
        auto scale_dev   = handle.Write(scale.data);
        auto shift_dev   = handle.Write(shift.data);
        auto estMean_dev = handle.Write(estMean.data);
        auto estVar_dev  = handle.Write(estVar.data);
        auto out_dev     = handle.Write(out.data);

        double epsilon = MIO_BN_TEST_EPSILON;

        int alpha = 1, beta = 1;

        miopen::BatchNormForwardInference(handle,
                                          miopenBNPerActivation,
                                          &alpha,
                                          &beta,
                                          input.desc,
                                          in_dev.get(),
                                          out.desc,
                                          out_dev.get(),
                                          scale.desc,
                                          scale_dev.get(),
                                          shift_dev.get(),
                                          estMean_dev.get(),
                                          estVar_dev.get(),
                                          epsilon); // TODO: add multi-in
        out.data = handle.Read<T>(out_dev, out.data.size());

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer_bn_per_activation_use_est pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    void fail(int)
    {
        std::cout << "Forward Inference Per Activation Batch Normalization Use Estimated: "
                  << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

//****************************************************
// BACKWARDS PROPAGATION
//****************************************************
template <class T>
struct verify_backward_bn_per_activation_use_saved
{

    const tensor<T> x_input;
    const tensor<T> dy_input;
    const tensor<T> scale;
    const tensor<T> savedMean;
    const tensor<T> savedInvVar;

    std::tuple<tensor<T>, tensor<T>, tensor<T>> cpu()
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        int n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tie4(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        auto dscale = tensor<T>{1, channels, height, width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<T>{1, channels, height, width};
        std::fill(dshift.begin(), dshift.end(), 0);

        const unsigned int in_cstride = height * width;
        const auto n                  = double(n_batch);

        par_for(channels, 1, [&](int cidx) {

            double elemStd = 0.;
            unsigned int xhat_index;
            double mean       = 0.;
            double elemInvVar = 0.;
            double dyelem     = 0.;
            double dxhat      = 0.;
            double dxhathat   = 0.;
            double tmp1       = 0.;
            std::vector<double> xhat(n_batch * in_cstride);

            // process the batch per channel
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    dxhat    = 0.;
                    dxhathat = 0.;

                    mean       = savedMean(0, cidx, row, column);   // HxW elements
                    elemInvVar = savedInvVar(0, cidx, row, column); // HxW elements

                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index = in_cstride * bidx + (width * row + column);
                        // per (x-dims) channel load a block of data into LDS
                        elemStd          = x_input(bidx, cidx, row, column) - mean; // (x_i - mean)
                        xhat[xhat_index] = elemStd * elemInvVar;
                        dyelem           = dy_input(bidx, cidx, row, column);
                        dshift(0, cidx, row, column) += dyelem;
                        dscale(0, cidx, row, column) += xhat[xhat_index] * dyelem;
                        tmp1 = scale(0, cidx, row, column) * dyelem;
                        dxhat += tmp1;
                        dxhathat += tmp1 * xhat[xhat_index];

                    } // end for(n_batchs)
                    dscale(0, cidx, row, column) /= n;

                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index  = in_cstride * bidx + (width * row + column);
                        tmp1        = xhat[xhat_index] * dxhathat + dxhat;
                        double tmp2 = n_batch * dxhat - tmp1;
                        double tmp3 = elemInvVar / (double(n));
                        dx_out(bidx, cidx, row, column) = tmp3 * tmp2;
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_bn_per_activation_use_saved pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return std::make_tuple(dx_out, dscale, dshift);
    }

    std::tuple<tensor<T>, tensor<T>, tensor<T>> gpu()
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();

        double epsilon = MIO_BN_TEST_EPSILON;

        int n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tie4(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        auto dscale = tensor<T>{1, channels, height, width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<T>{1, channels, height, width};
        std::fill(dshift.begin(), dshift.end(), 0);

        int alpha = 1, beta = 1;

        auto xin_dev         = handle.Write(x_input.data);
        auto dyin_dev        = handle.Write(dy_input.data);
        auto scale_dev       = handle.Write(scale.data);
        auto dscale_dev      = handle.Write(dscale.data);
        auto dshift_dev      = handle.Write(dshift.data);
        auto dx_out_dev      = handle.Write(dx_out.data);
        auto savedMean_dev   = handle.Write(savedMean.data);
        auto savedInvVar_dev = handle.Write(savedInvVar.data);

        miopen::BatchNormBackward(handle,
                                  miopenBNPerActivation,
                                  &alpha,
                                  &beta,
                                  &alpha,
                                  &beta,
                                  x_input.desc,
                                  xin_dev.get(),
                                  dy_input.desc,
                                  dyin_dev.get(),
                                  dx_out.desc,
                                  dx_out_dev.get(),
                                  scale.desc,
                                  scale_dev.get(),
                                  dscale_dev.get(),
                                  dshift_dev.get(),
                                  epsilon,
                                  savedMean_dev.get(),
                                  savedInvVar_dev.get());
        dx_out.data = handle.Read<T>(dx_out_dev, dx_out.data.size());
        dscale.data = handle.Read<T>(dscale_dev, dscale.data.size());
        dshift.data = handle.Read<T>(dshift_dev, dshift.data.size());

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU backward_bn_per_activation_use_saved pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return std::make_tuple(dx_out, dscale, dshift);
    }

    void fail(int badtensor)
    {
        std::cout << "Backward Batch Per Activation Normalization Using Saved Mean and Variance: "
                  << std::endl;
        std::cout << "X Input tensor: " << x_input.desc.ToString() << std::endl;
        std::cout << "Delta Y Input tensor: " << dy_input.desc.ToString() << std::endl;
        switch(badtensor)
        {
        case(0):
            std::cout << "Delta X output tensor output failed verification." << std::endl;
            break;
        case(1): std::cout << "Delta scale output tensor failed verification." << std::endl; break;
        case(2): std::cout << "Delta shift output tensor failed verification." << std::endl; break;
        }
    }
};

template <class T>
struct verify_backward_bn_per_activation_recalc
{

    const tensor<T> x_input;
    const tensor<T> dy_input;
    const tensor<T> scale;

    std::tuple<tensor<T>, tensor<T>, tensor<T>> cpu()
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        double epsilon = MIO_BN_TEST_EPSILON;

        int n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tie4(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        auto dscale = tensor<T>{1, channels, height, width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<T>{1, channels, height, width};
        std::fill(dshift.begin(), dshift.end(), 0);

        const unsigned int in_cstride = height * width;
        const auto n                  = double(n_batch);

        par_for(channels, 1, [&](int cidx) {

            double elemStd = 0.;
            unsigned int xhat_index;
            double mean       = 0.;
            double elemInvVar = 0.;
            double dyelem     = 0.;
            double variance   = 0.;
            double dxhat      = 0.;
            double dxhathat   = 0.;
            double tmp1       = 0.;
            std::vector<double> xhat(n_batch * in_cstride);

            // process the batch per channel
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    mean = 0.;
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // #1 calculate the mean
                        mean += x_input(bidx, cidx, row, column);
                    }
                    mean /= n;

                    elemStd  = 0.;
                    variance = 0.;
                    // #2 calculate the variances
                    // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // per (x-dims) channel load a block of data into LDS
                        elemStd = x_input(bidx, cidx, row, column) - mean; // (x_i - mean)
                        variance += elemStd * elemStd;                     // sum{ (x_i - mean)^2 }
                    }                                                      // end for(n)
                    variance /= n; // (1/N)*sum{ (x_i - mean)^2 }

                    // #3 add epsilon for numeric stability, sqr_root, and invert
                    elemInvVar = 1.0 / double(sqrt(variance + epsilon));

                    dxhat    = 0.;
                    dxhathat = 0.;

                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index = in_cstride * bidx + (width * row + column);
                        // per (x-dims) channel load a block of data into LDS
                        elemStd          = x_input(bidx, cidx, row, column) - mean; // (x_i - mean)
                        xhat[xhat_index] = elemStd * elemInvVar;
                        dyelem           = dy_input(bidx, cidx, row, column);
                        dshift(0, cidx, row, column) += dyelem;
                        dscale(0, cidx, row, column) += xhat[xhat_index] * dyelem;
                        tmp1 = scale(0, cidx, row, column) * dyelem;
                        dxhat += tmp1;
                        dxhathat += tmp1 * xhat[xhat_index];

                    } // end for(n_batchs)
                    dscale(0, cidx, row, column) /= n;

                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index  = in_cstride * bidx + (width * row + column);
                        tmp1        = xhat[xhat_index] * dxhathat + dxhat;
                        double tmp2 = n_batch * dxhat - tmp1;
                        double tmp3 = elemInvVar / double(n);
                        dx_out(bidx, cidx, row, column) = tmp3 * tmp2;
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
        });
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_bn_per_activation_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return std::make_tuple(dx_out, dscale, dshift);
    }

    std::tuple<tensor<T>, tensor<T>, tensor<T>> gpu()
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();

        int n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tie4(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, height, width};
        // std::fill(dx_out.begin(), dx_out.end(), 0);

        auto dscale = tensor<T>{1, channels, height, width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<T>{1, channels, height, width};
        std::fill(dshift.begin(), dshift.end(), 0);

        int alpha = 1, beta = 1;

        auto xin_dev    = handle.Write(x_input.data);
        auto dyin_dev   = handle.Write(dy_input.data);
        auto scale_dev  = handle.Write(scale.data);
        auto dscale_dev = handle.Write(dscale.data);
        auto dshift_dev = handle.Write(dshift.data);
        auto dx_out_dev = handle.Write(dx_out.data);

        double epsilon = MIO_BN_TEST_EPSILON;

        miopen::BatchNormBackward(handle,
                                  miopenBNPerActivation,
                                  &alpha,
                                  &beta,
                                  &alpha,
                                  &beta,
                                  x_input.desc,
                                  xin_dev.get(),
                                  dy_input.desc,
                                  dyin_dev.get(),
                                  dx_out.desc,
                                  dx_out_dev.get(),
                                  scale.desc,
                                  scale_dev.get(),
                                  dscale_dev.get(),
                                  dshift_dev.get(),
                                  epsilon,
                                  nullptr,
                                  nullptr);
        dx_out.data = handle.Read<T>(dx_out_dev, dx_out.data.size());
        dscale.data = handle.Read<T>(dscale_dev, dscale.data.size());
        dshift.data = handle.Read<T>(dshift_dev, dshift.data.size());

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU backward_bn_per_activation_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return std::make_tuple(dx_out, dscale, dshift);
    }

    void fail(int badtensor)
    {
        std::cout << "Backward Batch Per Activation Normalization Recalc Mean and Variance: "
                  << std::endl;
        std::cout << "X Input tensor: " << x_input.desc.ToString() << std::endl;
        std::cout << "Delta Y Input tensor: " << dy_input.desc.ToString() << std::endl;
        switch(badtensor)
        {
        case(0):
            std::cout << "Delta X output tensor output failed verification." << std::endl;
            break;
        case(1): std::cout << "Delta scale output tensor failed verification." << std::endl; break;
        case(2): std::cout << "Delta shift output tensor failed verification." << std::endl; break;
        }
    }
};

//====== DRIVERS ===========================================

template <class T>
struct batch_norm_per_activation_driver : test_driver
{
    tensor<T> input;
    tensor<T> scale;
    tensor<T> shift;

    batch_norm_per_activation_driver()
    {
        this->batch_factor = 4;
        add(input, "input", get_bn_peract_input_tensor());
    }

    void run()
    {
        int n, c, h, w;
        std::tie(n, c, h, w) = miopen::tie4(input.desc.GetLengths());

        if(n == 1)
        { // Invalid batch size for batch norm tests.
            return;
        }
        scale = tensor<T>{1, c, h, w}.generate(rand_gen{});
        shift = tensor<T>{1, c, h, w}.generate(rand_gen{});

        // train
        auto outpair = verify(verify_forward_train_bn_per_activation<T>{input, scale, shift});
        // returns:  std::make_tuple(out,runMean,runVar,saveMean,saveInvVar);

        // inference recalc
        verify(verify_forward_infer_bn_per_activation_recalc<T>{input, scale, shift});

        // inference use estimated running values
        auto estMean = std::get<1>(outpair.second);
        auto estVar  = std::get<2>(outpair.second);
        verify(verify_forward_infer_bn_per_activation_use_est<T>{
            input, scale, shift, estMean, estVar});

        // backprop recalc
        auto dy_input =
            tensor<T>{n, c, h, w}.generate(rand_gen{}); //= std::get<0>(outpair.first);//
        verify(verify_backward_bn_per_activation_recalc<T>{input, dy_input, scale});

        // backprop use saved values
        auto savedMean   = std::get<3>(outpair.second);
        auto savedInvVar = std::get<4>(outpair.second);
        verify(verify_backward_bn_per_activation_use_saved<T>{
            input, dy_input, scale, savedMean, savedInvVar});
    }
};

int main(int argc, const char* argv[])
{
#if(MIO_BN_TIME_EVERYTHING == 1)
    auto t_start = std::chrono::high_resolution_clock::now();
#endif
    test_drive<batch_norm_per_activation_driver<float>>(argc, argv);

#if(MIO_BN_TIME_EVERYTHING == 1)
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Wall clock: full PER_ACTIVATION test pass time: "
              << std::chrono::duration<double>(t_end - t_start).count() << " seconds." << std::endl;
#endif
}
