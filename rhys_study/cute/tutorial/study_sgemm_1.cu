#include <cute/tensor.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cutlass/util/helper_cuda.hpp"

template <class ProblemShape, class CtaTiler, class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout, class TC, class CStride, class CSmemLayout, class CThreadLayout>
__global__ static __launch_bounds__(decltype(size(CThreadLayout{}))::value) void gemm_kernel(
    ProblemShape shape_MNK, CtaTiler cta_tiler, TA const *A, AStride dA, ASmemLayout sA, AThreadLayout tA,
    TB const *B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
    TC *C, CStride dC, CSmemLayout sC, CThreadLayout tC)
{
  // check
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

  static_assert(is_static<AThreadLayout>::value);
  static_assert(is_static<BThreadLayout>::value);
  static_assert(is_static<CThreadLayout>::value);

  // 256
  CUTE_STATIC_ASSERT_V(size(tA) == size(tB));
  CUTE_STATIC_ASSERT_V(size(tC) == size(tA));

  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) == Int<0>{});

  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) == Int<0>{});

  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == Int<0>{});

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler)); // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler)); // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler)); // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler)); // BLK_K
  CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), dA));       // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dB));       // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC));       // dC strides for shape MN

  // make tensor
}

template <class TA, class TB, class TC>
void gemm(int m, int n, int k, TA const *A, TB const *B, TC *C)
{
  using namespace cute;

  auto prob_shape = make_shape(m, n, k);
  // calculate cta tiler

  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};

  auto cta_tiler = make_shape(bM, bN, bK);

  // calculate stride
  auto dA = make_stride(1, m);
  auto dB = make_stride(1, k);
  auto dC = make_stride(1, m);

  // calculate smem layout
  auto sA = make_layout(make_shape(bM, bK));
  auto sB = make_layout(make_shape(bN, bK));
  auto sC = make_layout(make_shape(bM, bN));

  // calculate thread layout
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(m, bM)), size(ceil_div(n, bN)));
  cudaStream_t stream = 0;
  gemm_kernel<<<dimGrid, dimBlock, 0, stream>>>(
      prob_shape, cta_tiler, A, dA, sA, tA,
      B, dB, sB, tB,
      C, dC, sC, tC);
}

int main()
{
  int m, n, k;
  m = n = 5120;
  k = 4096;
  cute::device_init(0);
  // create the host tensors

  using TA = float;
  using TB = float;
  using TC = float;
  thrust::host_vector<TA> h_A(m * k);
  thrust::host_vector<TB> h_B(n * k);
  thrust::host_vector<TC> h_C(m * n);

  // init the host tensors
  for (int i = 0; i < m * k; ++i)
  {
    h_A[i] = static_cast<TA>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < n * k; ++i)
  {
    h_B[i] = static_cast<TB>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < m * n; ++i)
  {
    h_C[i] = 0.0;
  }
  // create device tensors
  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  // calculate the gemm
  gemm(m, n, k, d_A.data().get(), d_B.data().get(), d_C.data().get());
}
