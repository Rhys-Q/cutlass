
#include "cute/int_tuple.hpp"
// #include "cute/util/print_tensor.hpp"
#include "cute/util/type_traits.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"
#include <cstdint>
#include <cute/tensor.hpp>
#include <cutlass/trace.h>
#include <iostream>
#include <ostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <type_traits>
/*
copy
copy 函数将source Tensor的元素 copy到 destination Tensor中。

**/
/*
Interface and specialization opportunities

copy有两个实现：

template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);
这个函数是根据src和dst的类型自动推断copy的实现方式的通用版本。
以及
template <class... CopyArgs,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<CopyArgs...>       const& copy_atom,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);
这个函数允许用户通过Copy_Atom参数指定copy的实现方式。
*/

/*
Parallelism and synchronization depend on parameter types

copy在一个thread内是串行的，但是在一个block或cluster内是并行的。
如果copy是并行的，就需考虑同步问题。
copy算法可能会使用cp.async同步指令。
*/

/*
下面是一个通用的copy实现
这个实现还有一些优化点，比如：
1. 如果这两个tensor，有已知更优的访问指令，如cp.async，可以使用这些指令。
2. 如果两个tensor是静态的， 能保证元素矢量化可用，例如4个ld.global.b32可以被矢量化为一个ld.global.b128，可以使用矢量化的访问。
3. 尽可能判断一下当前采用的copy指令是否是合适的。
目前cute的copy实现将上述优化都考虑到了。
*/

template <class TA, class ALayout, class TB, class BLayout>
CUTE_HOST_DEVICE void copy(Tenosr<TA, ALAyout> const &src,
                           Tensor<TB, BLayout> &dst)
{
    for (int i = 0; i < src.size(); ++i)
    {
        dst(i) = src(i);
    }
};

/*
copy_if
相比较copy，copy_if多了一个predicate参数。predicate 是一个shape和src相同rank的tensor，表示src中哪些元素需要被copy到dst中。非0的元素表示需要copy，0表示不需要copy。
*/

/*
gemm
参数有三个：A、B、C
- V 表示 一个"vector"，独立元素的一个mode
- M 和 N表示 表示rows的数量，结果矩阵C的columns
- K表示gemm的reduce mode

1. (V) x (V) => (V) elements-wise product：Cv += Av * Bv。会适配到FMA或者MMA。
2. (M) x (N) => (M,N) ，向量的外积，Cmn += Am * Bn
3. (M,K) x (N, K) => (M,N) 矩阵乘法，Cmn += Amk * Bnk
4. (V, M) x (V, N) => (V, M, N) batch 向量外积 Cvmn += Avm * Bvn
5. (v, M, K) x (V, N, K) => (V, M, N) batch matmul Cvmn += Avmk * Bvnk
*/

/*
axpby
y = a*x + b*y
a和b是标量，x和y是tensor
*/

/*
fill
填充tensor
*/

/*
clear
用0填充
*/
int main() {
};