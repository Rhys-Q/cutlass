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
using namespace cute;

/*

Tensor 由Engine和layout组成。
Layout用于计算给定coord的offset，Engine则用于随机访问。
Tensor的实际数据可以是Global Memory，Shared Memory，Register
Memory，或者是一个变换或者生成的数据 on the fly。
*/

/*
Fundamental operations

tensor<I...>(Tensor). The subtensor corresponding to the the I...th mode of the
Tensor.
*/

/*
Tensor Engine
Engine 是对iterator或者 array of data的封装，它使用std::array类似的接口
一般来说，用户不需要直接构造Engine，当Tensor创建的时候，会自动使用合适的Engine，例如ArrayEngine，ViewEngine，ConstViewEngine等。

Tagged Iterators
任何random-access
iterator都可以用来构造Tensor，但是用户也可以"tag"任何iterator为一个memory
space，例如global memory，shared memory，register memory等。
这通过make_gmem_ptr(g)或者make_gmem_ptr<T>(g)来实现，将g标记为global memory
iterator，或者make_smem_ptr(s)或者make_smem_ptr<T>(s)来标记为shared memory
iterator。
*/

/*
Tensor Creation

创建Tensor分为两类：
1. Owning Tensor
2. Non-Owning Tensor
Owning Tensor是拥有自己的内存的Tensor，例如std::array，它的内存是动态分配的。
Non-Owning Tensor是view of existing
memory的Tensor，例如pointer，它的内存是外部提供的。

一般都是non-owning的

*/
void test_tensor_creation()
{
  float *A = new float[10];
  auto tensor_8 = make_tensor(A, make_layout(Int<8>{}));
  auto tensor_8s = make_tensor(A, Int<8>{});
  auto tensor_8d2 = make_tensor(A, 8, 2);
  print(tensor_8);
  std::cout << std::endl;

  // Global memory
  auto gmem_8s = make_tensor(make_gmem_ptr(A), Int<8>{});
  auto gmem_8d = make_tensor(make_gmem_ptr(A), 8);

  print(gmem_8s);
  std::cout << std::endl;
  print(gmem_8d);
  std::cout << std::endl;
  // shared memory
  // Layout smem_layout = make_layout(make_shape(Int<4>{}, Int<8>{}));
  // __shared__ float smem[decltype(cosize(smem_layout))::value];
  // Tensor smem_4x8_col = make_tensor(make_smem_ptr(smem), smem_layout);
  // print(smem_4x8_col);
  // std::cout << std::endl;
  // Tensor smem_4x8_row =
  //     make_tensor(make_smem_ptr(smem), shape(smem_layout), LayoutRight{});
  // print(smem_4x8_row);
  // std::cout << std::endl;
}

/*
Owning Tensors
Owning Tensor是拥有自己的内存的Tensor，例如std::array，它的内存是动态分配的。
必须是静态的，不能是动态的。CUTE并不支持动态内存分配，因为这并不常见。
*/
void test_owning_tensors()
{
  // Register memory ，直接在register上分配内存
  Tensor rmem_4x8_col = make_tensor<float>(Shape<_4, _8>{});

  print(rmem_4x8_col);
  std::cout << std::endl;

  Tensor rmem_4x8_row = make_tensor<float>(Shape<_4, _8>{}, LayoutRight{});

  print(rmem_4x8_row);
  std::cout << std::endl;

  Tensor rmem_4x8_pad = make_tensor<float>(Shape<_4, _8>{}, Stride<_32, _2>{});

  print(rmem_4x8_pad);
  std::cout << std::endl;

  Tensor rmem_4x8_like = make_tensor_like(rmem_4x8_pad);
  print(rmem_4x8_like);
  std::cout << std::endl;
}
/*
Accessing a Tensor
用户可以使用()或者[]来访问Tensor的元素。

用户提供logical
coord，Tensor会使用layout来计算offset，然后使用engine来访问元素。

template <class Coord>
decltype(auto) operator[](Coord const& coord) {
  return data()[layout()(coord)];
}
*/

void test_accessing_a_tensor()
{
  Tensor A = make_tensor<float>(Shape<Shape<_4, _5>, Int<13>>{},
                                Stride<Stride<_12, _1>, _64>{});
  float *b_ptr = new float[13 * 20];
  Tensor B = make_tensor(b_ptr, make_shape(13, 20));

  // 填充 A
  for (int m0 = 0; m0 < size<0, 0>(A); ++m0)
  { // size<0, 0>(A) == 4
    for (int m1 = 0; m1 < size<0, 1>(A); ++m1)
    {
      for (int n = 0; n < size<1>(A); ++n)
      {
        A[make_coord(make_coord(m0, m1), n)] = n + 2 * m0;
      }
    }
  }

  // 将A转置，并写入B
  for (int m = 0; m < size<0>(A); ++m)
  {
    for (int n = 0; n < size<1>(A); ++n)
    {
      B(n, m) = A(m, n);
    }
  }
  // 将B copy 到A
  for (int i = 0; i < A.size(); ++i)
  {
    A[i] = B[i];
  }

  print(A);
  print(B);
}

/*
Tiling a Tensor
许多对layout的操作，也可以用到tensor上
   composition(Tensor, Tiler)
logical_divide(Tensor, Tiler)
 zipped_divide(Tensor, Tiler)
  tiled_divide(Tensor, Tiler)
   flat_divide(Tensor, Tiler)
上面这些操作运行任意layout的操作，然后返回一个新的tensor。这在tiling threadgroup的时候很有用
tiling for MMAs
注意，_product 操作不会应用到tensor上，因为它改变了tensor的rank。
 */

/*
slicing a tensor
_ 表示保留这个mode
 */

void test_slicing_tensor()
{
  // ((_3,2),(2,_5,_2)):((4,1),(_2,13,100))
  float *ptr = new float[3 * 2 * 2 * 5 * 2];
  Tensor A = make_tensor(ptr, make_shape(make_shape(Int<3>{}, 2), make_shape(2, Int<5>{}, Int<2>{})),
                         make_stride(make_stride(4, 1), make_stride(Int<2>{}, 13, 100)));
  // ((2,_5,_2)):((_2,13,100))
  Tensor B = A(2, _);
  // ((_3,_2)):((4,1))
  Tensor C = A(_, 5);
  // (_3,2):(4,1)
  Tensor D = A(make_coord(_, _), 5);
  // (_3, _5): (4,13)
  Tensor E = A(make_coord(_, 1), make_coord(0, _, 1));
  // (2,2,_2):(1,_2,100)
  Tensor F = A(make_coord(2, _), make_coord(_, 3, _));
}

/*
Partitioning a Tensor
*/

int main()
{
  test_slicing_tensor();
  return 0;
}