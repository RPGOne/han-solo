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

#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "test.hpp"
#include "verify.hpp"
#include <array>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/batch_norm.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <utility>

// Run CPU emulations in hierarchical reduction mode.
#define MIO_HEIRARCH_SEL 1
#define MIO_BN_TEST_EXPAVGFACTOR 0.1
#define MIO_BN_TEST_EPSILON 0.000001
#define MIO_BN_SP_TEST_DEBUG 0

//****************************************************
// FORWARD TRAIN
//****************************************************
template <class T>
struct verify_forward_train_bn_spatial
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

        int rs_n_batch, rs_channels, rs_height, rs_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNSpatial);

        std::tie(rs_n_batch, rs_channels, rs_height, rs_width) =
            miopen::tie4(derivedBnDesc.GetLengths());

        auto runMean = tensor<T>{rs_n_batch, rs_channels, rs_height, rs_width}.generate(rand_gen{});
        auto runVar  = tensor<T>{rs_n_batch, rs_channels, rs_height, rs_width}.generate(rand_gen{});
        auto saveMean   = tensor<T>{rs_n_batch, rs_channels, rs_height, rs_width};
        auto saveInvVar = tensor<T>{rs_n_batch, rs_channels, rs_height, rs_width};
        auto out        = input;
        std::fill(out.begin(), out.end(), 0);

        const unsigned int in_cstride = height * width;
        const auto nhw                = double(in_cstride * n_batch);

        par_for(channels, 1, [&](int cidx) {

            double elemStd        = 0.;
            double variance_accum = 0.;
            double mean_accum     = 0.;
            double invVar         = 0.;
            double newRunMean     = 0.;
            double adjust         = 0.;

#if(MIO_HEIRARCH_SEL == 1)
            std::vector<double> variance_accum_arr(height, 0.0);
            std::vector<double> mean_accum_arr(height, 0.0);
            std::vector<double> dshift_accum_arr(height, 0.0);
            std::vector<double> dscale_accum_arr(height, 0.0);
#endif

            mean_accum = 0.;

#if(MIO_HEIRARCH_SEL == 0)
            // process the batch per channel
            for(int bidx = 0; bidx < n_batch; bidx++)
            { // via mini_batch
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns
                        // #1 calculate the mean
                        // iterating through the stack of images in the mini_batch
                        mean_accum += input(bidx, cidx, row, column);
                    } // end for (column)
                }     // end for (row)
            }         // end for (n)
#else
            for (int row = 0; row < height; row++){ //via rows
                for(int column = 0; column < width; column++){// via columns
                    for (int bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                        mean_accum_arr[row] += input(bidx,cidx,row,column);
                    }	
                }// for (column)
            }// for (row)  
            for(int i = 0; i<height; i++) mean_accum += mean_accum_arr[i];
#endif
            mean_accum /= nhw;

            elemStd        = 0.;
            variance_accum = 0.;

#if(MIO_HEIRARCH_SEL == 0)
            // #2 calculate the variances
            // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
            for(int bidx = 0; bidx < n_batch; bidx++)
            { // via mini_batch
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns
                        // using out buffer as scratchpad
                        out(bidx, cidx, row, column) = elemStd =
                            (input(bidx, cidx, row, column) - mean_accum); // (x_i - mean)
                        variance_accum += (elemStd * elemStd);             // sum{ (x_i - mean)^2 }
                    }                                                      // end for (column)
                }                                                          // end for (row)
            }                                                              // end for(n)

#else
            for (int row = 0; row < height; row++){ //via rows
                for(int column = 0; column < width; column++){// via columns
                    for (int bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                        out(bidx,cidx,row,column) = elemStd = input(bidx,cidx,row,column) - mean_accum;
                        variance_accum_arr[row] += elemStd*elemStd;
                    }	
                }// for (column)
            }// for (row)  
            for(int i = 0; i<height; i++) variance_accum += variance_accum_arr[i];
#endif
            variance_accum /= nhw; // (1/N)*sum{ (x_i - mean)^2 }
            // #3 add epsilon for numeric stability, sqr_root, and invert
            invVar = 1.0 / sqrt(variance_accum + epsilon);

            // #4 apply the normalization
            // x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
            for(int bidx = 0; bidx < n_batch; bidx++)
            { // via mini_batch
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns
                        // #5 Gamma and Beta adjust
                        // y_i = gamma*x_hat + beta
                        out(bidx, cidx, row, column) =
                            scale(0, cidx, 0, 0) * (invVar * out(bidx, cidx, row, column)) +
                            shift(0, cidx, 0, 0);
                    } // for (column)
                }     // for (row)
            }         // end for(n_batchs)

            saveMean(0, cidx, 0, 0)   = mean_accum;
            saveInvVar(0, cidx, 0, 0) = invVar;

            newRunMean = runMean(0, cidx, 0, 0) * (1 - expAvgFactor);
            runMean(0, cidx, 0, 0) = mean_accum * expAvgFactor + newRunMean; // newMean*factor + tmp
            // var(n+1) = p * var(n-1) + (1 - p)*(b/b-1)*var(n)
            adjust = (n_batch * height * width == 1) ? variance_accum
                                                     : (nhw / (nhw - 1)) * variance_accum;
            runVar(0, cidx, 0, 0) =
                expAvgFactor * runVar(0, cidx, 0, 0) + (1 - expAvgFactor) * adjust;
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_train_bn_spatial pass time: "
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

        int rs_n_batch, rs_channels, rs_height, rs_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};

        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNSpatial);

        std::tie(rs_n_batch, rs_channels, rs_height, rs_width) =
            miopen::tie4(derivedBnDesc.GetLengths());

        auto runMean = tensor<T>{rs_n_batch, rs_channels, rs_height, rs_width}.generate(rand_gen{});
        auto runVar  = tensor<T>{rs_n_batch, rs_channels, rs_height, rs_width}.generate(rand_gen{});
        auto saveMean   = tensor<T>{rs_n_batch, rs_channels, rs_height, rs_width};
        auto saveInvVar = tensor<T>{rs_n_batch, rs_channels, rs_height, rs_width};

        // in buffers
        auto in_dev    = handle.Write(input.data);
        auto scale_dev = handle.Write(scale.data);
        auto shift_dev = handle.Write(shift.data);

        // out buffers
        auto runMean_dev    = handle.Write(runMean.data);
        auto runVar_dev     = handle.Write(runVar.data);
        auto saveMean_dev   = handle.Create<T>(channels);
        auto saveInvVar_dev = handle.Create<T>(channels);
        auto out_dev        = handle.Create<T>(n_batch * channels * height * width);

        double epsilon      = MIO_BN_TEST_EPSILON;
        double expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;

        int alpha = 1, beta = 1;
        miopen::BatchNormForwardTraining(handle,
                                         miopenBNSpatial,
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

        std::cout << "Wall clock: GPU forward_train_bn_spatial pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

        return std::make_tuple(out, runMean, runVar, saveMean, saveInvVar);
    }

    void fail(int badtensor)
    {

        std::cout << "Forward Train Spatial Batch Normalization: " << std::endl;
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
struct verify_forward_infer_bn_spatial_recalc
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

        auto out = input;
        std::fill(out.begin(), out.end(), 0);

        const unsigned int in_cstride = height * width;
        const auto nhw                = double(in_cstride * n_batch);

        par_for(channels, 1, [&](int cidx) {

            double elemStd        = 0.;
            double variance_accum = 0.;
            double mean_accum     = 0.;
            double inhat          = 0.;
            double invVar         = 0.;

            mean_accum = 0.;
            // process the batch per channel
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    // #1 calculate the mean
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // iterating through the stack of images in the mini_batch
                        mean_accum += input(bidx, cidx, row, column);
                    } // end for (n)
                }     // end for (column)
            }         // end for (row)
            mean_accum /= nhw;

            elemStd        = 0.;
            variance_accum = 0.;
            // #2 calculate the variances
            // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // using out buffer as scratchpad
                        out(bidx, cidx, row, column) = elemStd =
                            (input(bidx, cidx, row, column) - mean_accum); // (x_i - mean)
                        variance_accum += (elemStd * elemStd);             // sum{ (x_i - mean)^2 }
                    }                                                      // end for(n)
                }                                                          // end for (column)
            }                                                              // end for (row)
            variance_accum /= nhw; // (1/N)*sum{ (x_i - mean)^2 }

            // #3 add epsilon for numeric stability, sqr_root, and invert
            invVar = 1.0 / sqrt(variance_accum + epsilon);

            // #4 apply the normalization
            // x_hat = (x_i - mean) / sqrt(variance_accum - epsilon)
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        elemStd =
                            out(bidx, cidx, row, column); // using saved values from output tensor
                        inhat = elemStd * invVar;
                        // #5 Gamma and Beta adjust // y_i = gamma*x_hat + beta
                        out(bidx, cidx, row, column) =
                            scale(0, cidx, 0, 0) * inhat + shift(0, cidx, 0, 0);
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_infer_bn_spatial_recalc pass time: "
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

        int alpha = 1, beta = 1;

        double epsilon = MIO_BN_TEST_EPSILON;

        miopen::BatchNormForwardInference(handle,
                                          miopenBNSpatial,
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

        std::cout << "Wall clock: GPU forward_infer_bn_spatial_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    void fail(int)
    {
        std::cout << "Forward Inference Spatial Batch Normalization Recalc: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

template <class T>
struct verify_forward_infer_bn_spatial_use_est
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

        auto out = input;
        std::fill(out.begin(), out.end(), 0);

        par_for(channels, 1, [&](int cidx) {
            double elemStd  = 0.;
            double variance = 0.;
            double mean     = 0.;
            double inhat    = 0.;
            double invVar   = 0.;

            mean     = estMean(0, cidx, 0, 0);
            variance = estVar(0, cidx, 0, 0);
            invVar   = 1.0 / sqrt(variance + epsilon);
            // process the batch per channel
            for(int bidx = 0; bidx < n_batch; bidx++)
            { // via mini_batch
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns

                        elemStd = input(bidx, cidx, row, column) - mean;
                        inhat   = elemStd * invVar;
                        out(bidx, cidx, row, column) =
                            scale(0, cidx, 0, 0) * inhat + shift(0, cidx, 0, 0);
                    }
                }
            }
        });
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_infer_bn_spatial_use_est pass time: "
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
        auto estMean_dev = handle.Write(estMean.data);
        auto estVar_dev  = handle.Write(estVar.data);
        auto scale_dev   = handle.Write(scale.data);
        auto shift_dev   = handle.Write(shift.data);
        auto out_dev     = handle.Write(out.data);

        int alpha = 1, beta = 1;

        double epsilon = MIO_BN_TEST_EPSILON;

        miopen::BatchNormForwardInference(handle,
                                          miopenBNSpatial,
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
                                          epsilon);
        out.data = handle.Read<T>(out_dev, out.data.size());
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer_bn_spatial_use_est pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    void fail(int)
    {
        std::cout << "Forward Inference Spatial Batch Normalization Use Estimated: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

//****************************************************
// BACKWARDS PROPAGATION
//****************************************************
template <class T>
struct verify_backward_bn_spatial_recalc
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

        int ss_n_batch, ss_channels, ss_height, ss_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x_input.desc, miopenBNSpatial);
        std::tie(ss_n_batch, ss_channels, ss_height, ss_width) =
            miopen::tie4(derivedBnDesc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        auto dscale = tensor<T>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<T>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dshift.begin(), dshift.end(), 0);

        const unsigned int in_cstride = height * width;
        const auto nhw                = double(in_cstride * n_batch);

        par_for(channels, 1, [&](int cidx) {

            double elemStd = 0.;
            unsigned int xhat_index;
            double mean     = 0.;
            double invVar   = 0.;
            double dyelem   = 0.;
            double variance = 0.;

            std::vector<double> xhat(n_batch * in_cstride, 0.0);

#if(MIO_HEIRARCH_SEL == 1)
            std::vector<double> variance_accum_arr(height, 0.0);
            std::vector<double> mean_accum_arr(height, 0.0);
            std::vector<double> dshift_accum_arr(height, 0.0);
            std::vector<double> dscale_accum_arr(height, 0.0);
#endif

            // process the batch per channel
            mean = 0.;
#if(MIO_HEIRARCH_SEL == 0)
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // #1 calculate the mean
                        mean += x_input(bidx, cidx, row, column);
                    }
                } // for (column)
            }     // for (row)
#else
            for (int row = 0; row < height; row++){ //via rows
                for(int column = 0; column < width; column++){// via columns
                    for (int bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                        mean_accum_arr[row] += x_input(bidx,cidx,row,column);
                    }	
                }// for (column)
            }// for (row)  
            for(int i = 0; i<height; i++) mean += mean_accum_arr[i];
#endif
            mean /= nhw;

            elemStd  = 0.;
            variance = 0.;
#if(MIO_HEIRARCH_SEL == 0)
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    // #2 calculate the variances
                    // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // per (x-dims) channel load a block of data into LDS
                        elemStd = x_input(bidx, cidx, row, column) - mean; // (x_i - mean)
                        variance += elemStd * elemStd;                     // sum{ (x_i - mean)^2 }
                    }                                                      // end for(n)
                }                                                          // for (column)
            }                                                              // for (row)
#else
            for (int row = 0; row < height; row++){ //via rows
                for(int column = 0; column < width; column++){// via columns
                    for (int bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                        elemStd = x_input(bidx,cidx,row,column) - mean;
                        variance_accum_arr[row] += elemStd*elemStd;
                    }	
                }// for (column)
            }// for (row)  
            for(int i = 0; i<height; i++) variance += variance_accum_arr[i];
#endif
            variance /= nhw; // (1/(N*H*W))*sum{ (x_i - mean)^2 }
            invVar = 1. / double(sqrt(variance + epsilon));

            dscale(0, cidx, 0, 0) = 0.;

#if(MIO_HEIRARCH_SEL == 0)
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index = in_cstride * bidx + (width * row + column);
                        // per (x-dims) channel load a block of data into LDS
                        elemStd          = x_input(bidx, cidx, row, column) - mean; // (x_i - mean)
                        xhat[xhat_index] = elemStd * invVar;
                        dyelem           = dy_input(bidx, cidx, row, column);
                        dshift(0, cidx, 0, 0) += dyelem;
                        dscale(0, cidx, 0, 0) += xhat[xhat_index] * dyelem;
                    } // end for(n_batch)
                }     // for (column)
            }         // for (row)
#else   
            
            for (int row = 0; row < height; row++){ //via rows
                for(int column = 0; column < width; column++){// via columns
                    for (int bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                        xhat_index = in_cstride*bidx + (width*row + column);
                        //per (x-dims) channel load a block of data into LDS
                        elemStd             = x_input(bidx,cidx,row,column) - mean;// (x_i - mean)
                        xhat[xhat_index]    = elemStd*invVar;
                        dyelem              = dy_input(bidx,cidx,row,column);
                        dshift_accum_arr[row] += dyelem;
                        dscale_accum_arr[row] += xhat[xhat_index]*dyelem;
                    }	
                }// for (column)
            }// for (row)  
            for(int i = 0; i<height; i++) {
                dshift(0,cidx,0,0) += dshift_accum_arr[i];    
                dscale(0,cidx,0,0) += dscale_accum_arr[i];    
            }
#endif
            dscale(0, cidx, 0, 0) /= nhw;

            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index = in_cstride * bidx + (width * row + column);

                        double tmp1 =
                            nhw * dy_input(bidx, cidx, row, column) - dshift(0, cidx, 0, 0);
                        double tmp2 = -xhat[xhat_index] * dscale(0, cidx, 0, 0);
                        double tmp3 = (scale(0, cidx, 0, 0) * invVar) / nhw;
                        dx_out(bidx, cidx, row, column) = tmp3 * (tmp2 + tmp1);
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
        });           // for (channel)

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_bn_spatial_recalc pass time: "
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
        std::fill(dx_out.begin(), dx_out.end(), 0);

        int ss_n_batch, ss_channels, ss_height, ss_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x_input.desc, miopenBNSpatial);
        std::tie(ss_n_batch, ss_channels, ss_height, ss_width) =
            miopen::tie4(derivedBnDesc.GetLengths());

        auto dscale = tensor<T>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<T>{ss_n_batch, ss_channels, ss_height, ss_width};
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
                                  miopenBNSpatial,
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

        std::cout << "Wall clock: GPU backward_bn_spatial_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return std::make_tuple(dx_out, dscale, dshift);
    }

    void fail(int badtensor)
    {
        std::cout << "Backward Batch Spatial Normalization Recalc Mean and Variance: " << std::endl;
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
struct verify_backward_bn_spatial_use_saved
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

        int ss_n_batch, ss_channels, ss_height, ss_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x_input.desc, miopenBNSpatial);
        std::tie(ss_n_batch, ss_channels, ss_height, ss_width) =
            miopen::tie4(derivedBnDesc.GetLengths());

        auto dscale = tensor<T>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<T>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dshift.begin(), dshift.end(), 0);

        const unsigned int in_cstride = height * width;
        const auto nhw                = double(in_cstride * n_batch);

        par_for(channels, 1, [&](int cidx) {

            double elemStd = 0.;
            unsigned int xhat_index;
            double mean   = 0.;
            double invVar = 0.;
            double dyelem = 0.;

            std::vector<double> xhat(n_batch * in_cstride, 0.0);

#if(MIO_HEIRARCH_SEL == 1)
            std::vector<double> dshift_accum_arr(height, 0.0);
            std::vector<double> dscale_accum_arr(height, 0.0);
#endif

            // process the batch per channel
            mean   = savedMean(0, cidx, 0, 0);   // HxW elements
            invVar = savedInvVar(0, cidx, 0, 0); // HxW elements
            dscale(0, cidx, 0, 0) = 0.;

#if(MIO_HEIRARCH_SEL == 0)
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index = in_cstride * bidx + (width * row + column);
                        // per (x-dims) channel load a block of data into LDS
                        elemStd          = x_input(bidx, cidx, row, column) - mean; // (x_i - mean)
                        xhat[xhat_index] = elemStd * invVar;
                        dyelem           = dy_input(bidx, cidx, row, column);
                        dshift(0, cidx, 0, 0) += dyelem;
                        dscale(0, cidx, 0, 0) += xhat[xhat_index] * dyelem;
                    } // end for(n_batch)
                }     // for (column)
            }         // for (row)
#else   
            
            for (int row = 0; row < height; row++){ //via rows
                for(int column = 0; column < width; column++){// via columns
                    for (int bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                        xhat_index = in_cstride*bidx + (width*row + column);
                        //per (x-dims) channel load a block of data into LDS
                        elemStd             = x_input(bidx,cidx,row,column) - mean;// (x_i - mean)
                        xhat[xhat_index]    = elemStd*invVar;
                        dyelem              = dy_input(bidx,cidx,row,column);
                        dshift_accum_arr[row] += dyelem;
                        dscale_accum_arr[row] += xhat[xhat_index]*dyelem;
                    }	
                }// for (column)
            }// for (row)  
            for(int i = 0; i<height; i++) {
                dshift(0,cidx,0,0) += dshift_accum_arr[i];    
                dscale(0,cidx,0,0) += dscale_accum_arr[i];    
            }
#endif
            dscale(0, cidx, 0, 0) /= nhw;

            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index = in_cstride * bidx + (width * row + column);

                        double tmp1 =
                            nhw * dy_input(bidx, cidx, row, column) - dshift(0, cidx, 0, 0);
                        double tmp2 = -xhat[xhat_index] * dscale(0, cidx, 0, 0);
                        double tmp3 = (scale(0, cidx, 0, 0) * invVar) / nhw;
                        dx_out(bidx, cidx, row, column) = tmp3 * (tmp2 + tmp1);
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
        });           // for (channel)
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_bn spatial_use_saved pass time: "
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
        std::fill(dx_out.begin(), dx_out.end(), 0);

        int ss_n_batch, ss_channels, ss_height, ss_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x_input.desc, miopenBNSpatial);
        std::tie(ss_n_batch, ss_channels, ss_height, ss_width) =
            miopen::tie4(derivedBnDesc.GetLengths());

        auto dscale = tensor<T>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<T>{ss_n_batch, ss_channels, ss_height, ss_width};
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

        double epsilon = MIO_BN_TEST_EPSILON;

        miopen::BatchNormBackward(handle,
                                  miopenBNSpatial,
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

        std::cout << "Wall clock: GPU backward_bn_spatial_use_saved pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

        return std::make_tuple(dx_out, dscale, dshift);
    }

    void fail(int badtensor)
    {
        std::cout << "Backward Batch Spatial Normalization Use Saved Mean and Variance: "
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
struct batch_norm_spatial_driver : test_driver
{
    tensor<T> input;
    tensor<T> scale;
    tensor<T> shift;
    batch_norm_spatial_driver()
    {
        this->batch_factor = 8;
        // this->verbose=true;
        add(input, "input", get_bn_spatial_input_tensor());
    }

    void run()
    {
        return;
        int n, c, h, w;

        std::tie(n, c, h, w) = miopen::tie4(input.desc.GetLengths());

        if(n == 1)
        { // Invalid batch size for batch normalization
            return;
        }

        int ssn, ssc, ssh, ssw;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNSpatial);
        std::tie(ssn, ssc, ssh, ssw) = miopen::tie4(derivedBnDesc.GetLengths());

        scale = tensor<T>{ssn, ssc, ssh, ssw}.generate(rand_gen{});
        shift = tensor<T>{ssn, ssc, ssh, ssw}.generate(rand_gen{});

// train
#if(MIO_BN_SP_TEST_DEBUG == 1)
        std::cout << "Running forward train spatial with R and S set." << std::endl;
#endif
        auto outpair = verify(verify_forward_train_bn_spatial<T>{input, scale, shift});
// returns:  std::make_tuple(out,runMean,runVar,saveMean,saveInvVar);

// inference recalc
#if(MIO_BN_SP_TEST_DEBUG == 1)
        std::cout << "Running forward inference spatial recalc." << std::endl;
#endif
        verify(verify_forward_infer_bn_spatial_recalc<T>{input, scale, shift});

        // inference use estimated running values
        auto estMean = std::get<1>(outpair.second);
        auto estVar  = std::get<2>(outpair.second);
#if(MIO_BN_SP_TEST_DEBUG == 1)
        std::cout << "Running forward inference spatial with R set." << std::endl;
#endif
        verify(verify_forward_infer_bn_spatial_use_est<T>{input, scale, shift, estMean, estVar});

        // backprop recalc
        auto dy_input = std::get<0>(outpair.second);

#if(MIO_BN_SP_TEST_DEBUG == 2)
        auto debugvals = verify(verify_backward_bn_spatial_recalc<T>{input, dy_input, scale});
        auto gpuout    = std::get<0>(debugvals.second);
        auto cpuout    = std::get<0>(debugvals.first);

        double maxdiff = 0.;
        int mn         = 0;
        int mc         = 0;
        int mh         = 0;
        int mw         = 0;

        for(int bidx = 0; bidx < n; bidx++)
        { // via mini_batch
            for(int cidx = 0; cidx < c; cidx++)
            { // via mini_batch
                for(int row = 0; row < h; row++)
                { // via rows
                    for(int column = 0; column < w; column++)
                    { // via columns
                        double diff =
                            fabs(gpuout(bidx, cidx, row, column) - cpuout(bidx, cidx, row, column));
                        if(diff > maxdiff)
                        {
                            maxdiff = diff;
                            mn      = bidx;
                            mc      = cidx;
                            mh      = row;
                            mw      = column;
                        }
                        if(diff > 1.)
                        {
                            std::cout << "gpu[" << bidx << ", " << cidx << ", " << row << ", "
                                      << column << "]: " << gpuout(bidx, cidx, row, column)
                                      << " :: ";
                            std::cout << "cpu[" << bidx << ", " << cidx << ", " << row << ", "
                                      << column << "]: " << cpuout(bidx, cidx, row, column)
                                      << " :: ";
                            std::cout << "diff: " << diff << std::endl;
                        }
                    }
                }
            }
        }
        if(maxdiff > 0)
        {
            std::cout << "Max diff: " << maxdiff << std::endl;
            std::cout << "gpu[" << mn << ", " << mc << ", " << mh << ", " << mw
                      << "]: " << gpuout(mn, mc, mh, mw) << " :: ";
            std::cout << "cpu[" << mn << ", " << mc << ", " << mh << ", " << mw
                      << "]: " << cpuout(mn, mc, mh, mw) << std::endl;
        }
#else
#if(MIO_BN_SP_TEST_DEBUG == 1)
        std::cout << "Running back propagation spatial recalc." << std::endl;
#endif
        verify(verify_backward_bn_spatial_recalc<T>{input, dy_input, scale});
#endif

        // backprop use saved values
        auto savedMean   = std::get<3>(outpair.second);
        auto savedInvVar = std::get<4>(outpair.second);
#if(MIO_BN_SP_TEST_DEBUG == 1)
        std::cout << "Running back propagation spatial with S set." << std::endl;
#endif
        verify(verify_backward_bn_spatial_use_saved<T>{
            input, dy_input, scale, savedMean, savedInvVar});
    }
};

int main(int argc, const char* argv[])
{
#if(MIO_BN_TIME_EVERYTHING == 1)
    auto t_start = std::chrono::high_resolution_clock::now();
#endif
    test_drive<batch_norm_spatial_driver<float>>(argc, argv);

#if(MIO_BN_TIME_EVERYTHING == 1)
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "Wall clock: full SPATIAL test pass time: "
              << std::chrono::duration<double>(t_end - t_start).count() << " seconds." << std::endl;
#endif
    exit(0);
}
