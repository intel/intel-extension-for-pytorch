/*******************************************************************************
 * Copyright 2016-2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <memory>
#include "StopWatch.h"
#include "mkl_wrappers.h"
#include <cmath>

template <typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result) {
        std::cout << (std::string("SYCL runtime error: ") + +file + ":" + std::to_string(line) +
                      " \n");
    }
}

#define check_sycl_error(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
class GemmTest {
public:
    GemmTest(int m,
             int n,
             int k,
             oneapi::mkl::transpose ta,
             oneapi::mkl::transpose tb,
             dpct::queue_ptr h)
        : M(m), N(n), K(k), transa(ta), transb(tb), handle(h)
    {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.in_order_queue();
        check_sycl_error(DPCT_CHECK_ERROR(A = (T*)sycl::malloc_device(sizeof(T) * M * K, q_ct1)));
        check_sycl_error(DPCT_CHECK_ERROR(B = (T*)sycl::malloc_device(sizeof(T) * K * N, q_ct1)));
        check_sycl_error(DPCT_CHECK_ERROR(C = (T*)sycl::malloc_device(sizeof(T) * M * N, q_ct1)));
    }

    ~GemmTest()
    {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.in_order_queue();
        check_sycl_error(DPCT_CHECK_ERROR(sycl::free(A, q_ct1)));
        check_sycl_error(DPCT_CHECK_ERROR(sycl::free(B, q_ct1)));
        check_sycl_error(DPCT_CHECK_ERROR(sycl::free(C, q_ct1)));
    }

    std::array<int, 3> TestAlgo(int loops)
    {
        float alpha = (T)1.0f;
        float beta = (T)0.0f;

        int algo_fw = Run(loops, [=](int algo) {
            mkl_gemm_ex(handle,
                           oneapi::mkl::transpose::trans,
                           oneapi::mkl::transpose::nontrans,
                           N,
                           M,
                           K,
                           &alpha,
                           &beta,
                           B,
                           A,
                           C,
                           static_cast<int>(algo));
        });

        int algo_bw1 = Run(loops, [=](int algo) {
            mkl_gemm_ex(handle,
                           oneapi::mkl::transpose::nontrans,
                           oneapi::mkl::transpose::trans,
                           K,
                           N,
                           M,
                           &alpha,
                           &beta,
                           A,
                           C,
                           B,
                           static_cast<int>(algo));
        });

        int algo_bw2 = Run(loops, [=](int algo) {
            mkl_gemm_ex(handle,
                           oneapi::mkl::transpose::nontrans,
                           oneapi::mkl::transpose::nontrans,
                           K,
                           M,
                           N,
                           &alpha,
                           &beta,
                           B,
                           C,
                           A,
                           static_cast<int>(algo));
        });

        return std::array<int, 3>({algo_fw, algo_bw1, algo_bw2});
    }

    template <typename Func>
    int Run(int loops, Func f)
    {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
        float fast_latency = (std::numeric_limits<float>::max)();
        int fast_algo = 0;

        for (int algo = (int)99; algo <= (int)115;
             algo++) {
            int warm_up = 5;
            for (int i = 0; i < warm_up; ++i) f(algo);

            dev_ct1.queues_wait_and_throw();
            Stopwatch timer;
            timer.Restart();

            for (int i = 0; i < loops; ++i) f(algo);

            dev_ct1.queues_wait_and_throw();
            timer.Stop();

            float avg_latency = (float)timer.GetTimeInSeconds() * 1000 / loops;

            printf("algo-%d: %.3fms\n", algo, avg_latency);

            if (avg_latency < fast_latency) {
                fast_latency = avg_latency;
                fast_algo = algo;
            }
        }

        printf("fast_algo %d: %.3f ms\n", fast_algo, fast_latency);

        return fast_algo;
    }

private:
    int M, N, K;
    dpct::queue_ptr handle;
    oneapi::mkl::transpose transa, transb;
    T *A, *B, *C;
};

template <typename T>
class StridedGemmTest {
public:
    StridedGemmTest(int b,
                    int m,
                    int n,
                    int k,
                    oneapi::mkl::transpose ta,
                    oneapi::mkl::transpose tb,
                    dpct::queue_ptr h)
        : bsz(b), M(m), N(n), K(k), transa(ta), transb(tb), handle(h)
    {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.in_order_queue();
        check_sycl_error(
            DPCT_CHECK_ERROR(A = (T*)sycl::malloc_device(sizeof(T) * M * K * bsz, q_ct1)));
        check_sycl_error(
            DPCT_CHECK_ERROR(B = (T*)sycl::malloc_device(sizeof(T) * K * N * bsz, q_ct1)));
        check_sycl_error(
            DPCT_CHECK_ERROR(C = (T*)sycl::malloc_device(sizeof(T) * M * N * bsz, q_ct1)));
    }

    ~StridedGemmTest()
    {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.in_order_queue();
        check_sycl_error(DPCT_CHECK_ERROR(sycl::free(A, q_ct1)));
        check_sycl_error(DPCT_CHECK_ERROR(sycl::free(B, q_ct1)));
        check_sycl_error(DPCT_CHECK_ERROR(sycl::free(C, q_ct1)));
    }

    std::array<int, 3> TestAlgo(int loops)
    {
        float alpha = (T)1.0f;
        float beta = (T)0.0f;

        int algo_fw = Run(loops, [=](int algo) {
            int stride_a = M * K;
            int stride_b = N * K;
            int stride_c = M * N;

            mkl_strided_batched_gemm(handle,
                                        M,
                                        N,
                                        K,
                                        &alpha,
                                        &beta,
                                        A,
                                        B,
                                        C,
                                        transa,
                                        transb,
                                        stride_a,
                                        stride_b,
                                        stride_c,
                                        bsz,
                                        static_cast<int>(algo));
        });

        int algo_bw1 = Run(loops, [=](int algo) {
            int mb = (transa == oneapi::mkl::transpose::trans ? K : M);
            int kb = (transa == oneapi::mkl::transpose::trans ? M : K);

            int stride_a = mb * N;
            int stride_b = N * kb;
            int stride_c = M * K;

            // B need to transpose.
            oneapi::mkl::transpose op_b =
                (transb == oneapi::mkl::transpose::trans ? oneapi::mkl::transpose::nontrans
                                                         : oneapi::mkl::transpose::trans);

            // Calculate d_A.
            mkl_strided_batched_gemm(handle,
                                        mb,
                                        kb,
                                        N,
                                        &alpha,
                                        &beta,
                                        (transa == oneapi::mkl::transpose::trans ? B : C),
                                        (transa == oneapi::mkl::transpose::trans ? C : B),
                                        A,
                                        oneapi::mkl::transpose::nontrans,
                                        op_b,
                                        stride_a,
                                        stride_b,
                                        stride_c,
                                        bsz,
                                        static_cast<int>(algo));
        });

        int algo_bw2 = Run(loops, [=](int algo) {
            // A need to transpose.
            oneapi::mkl::transpose op_a =
                (transa == oneapi::mkl::transpose::trans ? oneapi::mkl::transpose::nontrans
                                                         : oneapi::mkl::transpose::trans);

            int stride_a = M * K;
            int stride_b = M * N;
            int stride_c = N * K;

            // Calculate d_B.
            mkl_strided_batched_gemm(handle,
                                        K,
                                        N,
                                        M,
                                        &alpha,
                                        &beta,
                                        A,
                                        C,
                                        B,
                                        op_a,
                                        oneapi::mkl::transpose::nontrans,
                                        stride_a,
                                        stride_b,
                                        stride_c,
                                        bsz,
                                        static_cast<int>(algo));
        });

        return std::array<int, 3>({algo_fw, algo_bw1, algo_bw2});
    }

    template <typename Func>
    int Run(int loops, Func f)
    {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
        float fast_latency = (std::numeric_limits<float>::max)();
        int fast_algo = 0;

        for (int algo = (int)99; algo <= (int)115;
             algo++) {
            int warm_up = 5;
            for (int i = 0; i < warm_up; ++i) f(algo);

            dev_ct1.queues_wait_and_throw();
            Stopwatch timer;
            timer.Restart();

            for (int i = 0; i < loops; ++i) f(algo);

            dev_ct1.queues_wait_and_throw();
            timer.Stop();

            float avg_latency = (float)timer.GetTimeInSeconds() * 1000 / loops;

            printf("algo-%d: %.3fms\n", algo, avg_latency);

            if (avg_latency < fast_latency) {
                fast_latency = avg_latency;
                fast_algo = algo;
            }
        }

        printf("fast_algo %d: %.3f ms\n", fast_algo, fast_latency);

        return fast_algo;
    }

private:
    int bsz, M, N, K;
    dpct::queue_ptr handle;
    oneapi::mkl::transpose transa, transb;
    T *A, *B, *C;
};
