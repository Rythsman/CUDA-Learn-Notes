/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
#include <curand.h>
#include <cstdio>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
template <typename G, typename T>
__device__ __forceinline__ T reduce(G const &g, T const *s)
{
    T ad;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        int const of = 73 * i;
        T ac{0};
        ac += s[of + g.thread_rank()];      // Add [0, 32)
        ac += s[of + 32 + g.thread_rank()]; // Add [32, 64)
        if (g.thread_rank() < 9)
            ac += s[of + 64 + g.thread_rank()]; // Add [64, 73)
        ac = cooperative_groups::reduce(g, ac, cooperative_groups::plus<T>());
        if (g.thread_rank() == i)
            ad = ac;
    }
    return ad;
}
template <typename T>
__global__ void row_reduce(T *in, T *ou)
{
    __shared__ T sb[4 * 32];
    __shared__ T sa[4 * 2336];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    auto bl = cooperative_groups::this_thread_block();
    auto ti = cooperative_groups::tiled_partition<32>(bl);
    int const br = 128 * bl.group_index().x;
    if (bl.thread_rank() == 0)
    {
        init(&bar, 128);
        cde::fence_proxy_async_shared_cta();
    }
    bl.sync();
    barrier::arrival_token token;
    if (bl.thread_rank() == 0)
    {
        cde::cp_async_bulk_global_to_shared(sa, in + 73 * br, sizeof(T) * 4 * 2336, bar);
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(T) * 4 * 2336);
    }
    else
    {
        token = bar.arrive();
    }
    bar.wait(std::move(token));
    sb[bl.thread_rank()] = reduce(ti, sa + 2336 * ti.meta_group_rank());
    cde::fence_proxy_async_shared_cta();
    bl.sync();
    if (bl.thread_rank() == 0)
    {
        cde::cp_async_bulk_shared_to_global(ou + 128 * bl.group_index().x, sb, sizeof(T) * 128);
        cde::cp_async_bulk_commit_group();
        cde::cp_async_bulk_wait_group_read<0>();
    }
}
int main()
{
    constexpr int ROW = 4380032;
    constexpr int COL = 73;
    std::unique_ptr<float[]> s(new float[ROW]);
    std::unique_ptr<float[]> s1(new float[ROW]);
    std::unique_ptr<float[]> h(new float[ROW * COL]);
    curandGenerator_t generator;
    curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(generator, h.get(), ROW * COL);
    for (int i = 0; i < ROW; i++)
    {
        s[i] = s1[i] = 0.0f;
        for (int j = 0; j < COL; j++)
            s[i] += h[i * COL + j];
    }
    float *din, *dout;
    cudaMalloc(&din, sizeof(float) * ROW * COL);
    cudaMalloc(&dout, sizeof(float) * ROW);
    cudaMemcpy(din, h.get(), sizeof(float) * ROW * COL, cudaMemcpyHostToDevice);
    row_reduce<<<ROW / (32 * (128 / 32)), 128>>>(din, dout);
    cudaDeviceSynchronize();
    cudaMemcpy(s1.get(), dout, sizeof(float) * ROW, cudaMemcpyDeviceToHost);
    for (int i = 0; i < ROW; i++)
        if (s[i] - s1[i] > 1e-4)
            printf("%d: %f %f\n", i, s[i], s1[i]);
    cudaFree(din);
    cudaFree(dout);
    curandDestroyGenerator(generator);
    return 0;
}