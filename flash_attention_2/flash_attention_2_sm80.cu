#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "flash_attention_2_sm80.hpp"


template <typename T>
void gen_rand_data(T* data, int n) {
    for (int i = 0; i < n; ++i) {
        float v = (rand() % 200 - 100) * 0.01;
        data[i] = v;
    }
}

void test_attention(cudaStream_t stream = 0) {
    using namespace cute;
    using elem_type = cute::half_t;

    srand(10086);

    constexpr int kHeadDim = 32;
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 32;
    constexpr int kNWarps = 4;

    int bs = 1;
    int head = 2;
    int seqlen_q = 256;
    int seqlen_k = 128;
    int dim = kHeadDim;

    thrust::host_vector<elem_type> h_Q(bs * head * seqlen_q * dim);
    thrust::host_vector<elem_type> h_K(bs * head * seqlen_k * dim);
    thrust::host_vector<elem_type> h_V(bs * head * seqlen_k * dim);


    for (int j = 0; j < bs * head * seqlen_q * dim; ++j) h_Q[j] = (j / 100.0f);
    for (int j = 0; j < bs * head * seqlen_k * dim; ++j) h_K[j] = ((j + 1) / 100.0f);
    for (int j = 0; j < bs * head * seqlen_k * dim; ++j) h_V[j] = ((j + 2) / 100.0f);
    //gen_rand_data(h_Q.data(), h_Q.size());
    //gen_rand_data(h_K.data(), h_K.size());
    //gen_rand_data(h_V.data(), h_V.size());

    thrust::device_vector<elem_type> d_Q = h_Q;
    thrust::device_vector<elem_type> d_K = h_K;
    thrust::device_vector<elem_type> d_V = h_V;

    Flash_fwd_params params;
    params.bs = bs;
    params.head = head;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.dim = dim;

    params.q_batch_stride = head * seqlen_q * dim;
    params.k_batch_stride = head * seqlen_k * dim;
    params.v_batch_stride = head * seqlen_k * dim;
    params.q_head_stride = seqlen_q * dim;
    params.k_head_stride = seqlen_k * dim;
    params.v_head_stride = seqlen_k * dim;

    params.q_ptr = d_Q.data().get();
    params.k_ptr = d_K.data().get();
    params.v_ptr = d_V.data().get();

    using Kernel_traits = Flash_fwd_kernel_traits<kHeadDim, kBlockM, kBlockN, kNWarps, elem_type>;

    auto kernel = &flash::flash_attention_v2_cute_kernel<Kernel_traits, false, Flash_fwd_params>;

    // Q smem size + KV smem size
    constexpr int smem_size = Kernel_traits::kSmemSize;
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    const int num_m_block = ceil_div(params.seqlen_q, Kernel_traits::kBlockM);

    dim3 grid(num_m_block, params.bs * params.head, 1);
    dim3 block(size(Kernel_traits::TiledMma{}));

    kernel<<<grid, block, smem_size, stream>>>(params);

    CUTE_CHECK_LAST();
}


int main(int argc, char** argv) {
    test_attention();
}

