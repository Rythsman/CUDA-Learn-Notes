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

template<typename INTYPE=float, typename OUTTYPE=float, int ROWS=4380000, int COLS=73> 
__global__ void row_reduce(INTYPE* data_in, OUTTYPE* data_out) {
    namespace cg = cooperative_groups;

    constexpr int ROWS_EACH_WARP = 878; // for ROWS=4380000
    constexpr int WARP_SIZE = 32;
    
    auto block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<WARP_SIZE>(block);
    const int inner_warp_id = tile32.thread_rank();
    const unsigned int warp_id = cg::this_grid().thread_rank() / WARP_SIZE;
    const int start_row = ROWS_EACH_WARP * warp_id;
    const int end_row = min(ROWS_EACH_WARP * (warp_id + 1), ROWS);

    for(int i = start_row; i < end_row; ++i) {
        float sum = 0.0;
        sum += data_in[i * COLS  + inner_warp_id];
        sum += data_in[i * COLS +  WARP_SIZE + inner_warp_id];
        if(inner_warp_id < 9) {
            sum += data_in[i * COLS + 2 * WARP_SIZE + inner_warp_id];
        }

        sum = cg::reduce(tile32, sum, cg::plus<float>());
        if(inner_warp_id == 0) {
            data_out[i] = sum;
        }
    }
}


int main()
{
    constexpr int ROW = 4380000;
    constexpr int COL = 73;

    std::unique_ptr<float[]> s(new float[ROW]);
    std::unique_ptr<float[]> s1(new float[ROW]);
    std::unique_ptr<float[]> h(new float[ROW * COL]);

    curandGenerator_t generator;
    curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(generator, h.get(), ROW * COL);

    for(int i = 0; i < ROW; i++) {
        s[i] = s1[i] = 0.0f;
        for(int j = 0; j < COL; j++)
            s[i] += h[i * COL + j];
    }

    float* din, *dout;
    cudaMalloc(&din, sizeof(float) * ROW * COL);
    cudaMalloc(&dout, sizeof(float) * ROW);
    cudaMemcpy(din, h.get(), sizeof(float) * ROW * COL, cudaMemcpyHostToDevice);

    row_reduce<<<156, 1024>>>(din, dout);
    cudaDeviceSynchronize();

    cudaMemcpy(s1.get(), dout, sizeof(float) * ROW, cudaMemcpyDeviceToHost);

    for(int i = 0; i < ROW; i++)
        if(s[i] - s1[i] > 1e-3) 
            printf("%d: %f %f\n", i, s[i], s1[i]);

    cudaFree(din);
    cudaFree(dout);

    curandDestroyGenerator(generator);
    return 0;
}