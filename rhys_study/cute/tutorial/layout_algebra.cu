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

coalesce 是简化，类似simplify，它不改变size，其他都改

layout {shape, stride} + coord -> index

(_2, _4):(_1, _2) 在1-D coord的情况下，和_8:_1 的 index 是相同的

static-1 的维度，在natural coord的情况下，
它只有一个取值，那就是0，可以直接被忽略

对于二维layout s0:d0 和 s1:d1， coalesce 只有四种场景：
1. s0:d0 ++ _1:d1 => s0:d0, 如果shape为1，则可以直接忽略
2. _1:d0 += s1:d1 => s1:d1, 如果shape为1，则可以直接忽略
3. s0:d0 ++ s1:s0*d0 => s0*s1:d0,
如果第二个维度的stride是第一个维度的shape的整数倍，则可以将第一个第二个shape合并

4. s0:d0 ++ s1:d1 => (s0,s1):(d0,d1), 此外，都不能合并

举个例子，(_2, _4):(_1, _2) 符合第3个场景，可以合并为(_8):(_1)

*/
void test_coalesce()
{
  auto layout = Layout<Shape<_2, Shape<_1, _6>>, Stride<_1, Stride<_6, _2>>>{};

  print(layout);

  for (int i = 0; i < size(layout); i++)
  {
    print(crd2idx(i, layout.shape(),
                  layout.stride())); // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
  }

  auto result = coalesce(layout); // _12:_1

  print(result); // equal to layout but simpler
}

// by mode coalesce
void test_by_mode_coalesce()
{
  auto a = Layout<Shape<_2, Shape<_1, _6>>, Stride<_1, Stride<_6, _2>>>{};

  auto result = coalesce(a, Step<_1, _1>{});

  print(result); // equal to a but simpler
}

/*
Composition
这是cute的核心，被每一个高层的操作所使用
组合的公式可以如下定义：
R := A o B
R(c) := (A o B)(c) := A(B(c))
Example
A = (6,2):(8,2)
B = (4,3):(3,1)

R( 0) = A(B( 0)) = A(B(0,0)) = A( 0) = A(0,0) =  0
R( 1) = A(B( 1)) = A(B(1,0)) = A( 3) = A(3,0) = 24
R( 2) = A(B( 2)) = A(B(2,0)) = A( 6) = A(0,1) =  2
R( 3) = A(B( 3)) = A(B(3,0)) = A( 9) = A(3,1) = 26
R( 4) = A(B( 4)) = A(B(0,1)) = A( 1) = A(1,0) =  8
R( 5) = A(B( 5)) = A(B(1,1)) = A( 4) = A(4,0) = 32
R( 6) = A(B( 6)) = A(B(2,1)) = A( 7) = A(1,1) = 10
R( 7) = A(B( 7)) = A(B(3,1)) = A(10) = A(4,1) = 34
R( 8) = A(B( 8)) = A(B(0,2)) = A( 2) = A(2,0) = 16
R( 9) = A(B( 9)) = A(B(1,2)) = A( 5) = A(5,0) = 40
R(10) = A(B(10)) = A(B(2,2)) = A( 8) = A(2,1) = 18
R(11) = A(B(11)) = A(B(3,2)) = A(11) = A(5,1) = 42

如果将R写成layout，则可以写成：
R = ((2,2),3):((24,2),8) 这是观察得到的

并且：
compatible(B, R)
回顾一下compatible的定义：
compatible(A, B) := for all c in B, A(c) is defined
这意味着B的每一个坐标都可以作为R的坐标，且B的shape的size和R的shape的size相同


Composition 计算
B = (B_0, B_1, ...) B是由sublayouts组成的layout，比如 (4,3):(3,1) B_0 = (4,3)
B_1 = (3,1) B = (B_0, B_1)

Composition (A o B) 本质上使用B的stride d 来提取A的元素，再用B的
shape来截取前s个元素

左分配律：
A o B = A o (B_0, B_1, ...) = (A o B_0, A o B_1, ...)
假设B = s:d，A是完全展开的，简化后的layout, A = a:b
R = A o B = A(B) = a:b o s:d = s:(b*d)

如果A有多个mode，则按照两步走:
1. 使用B的stride 来除A的shape，得到一个中间的layput
For example,
(6,2) /  2 => (3,2)
(6,2) /  3 => (2,2)
(6,2) /  6 => (1,2)
(6,2) / 12 => (1,1)
(3,6,2,8) /  3 => (1,6,2,8)
(3,6,2,8) /  6 => (1,3,2,8)
(3,6,2,8) /  9 => (1,2,2,8)
(3,6,2,8) / 72 => (1,1,1,4)
假设A 是 (3, 6, 2, 8):(w, x, y, z)
中间layout的shape = (3, 6, 2, 8) / 72 = (1, 1, 1, 4)
中间layout的stride = (72*w, 24*x, 4*y, 2*z)

2. 使用B的shape 来截取中间layout的前s个元素
首先，从中间layout的最左侧维度开始，逐步用取模的方式，获取s个元素
计算新的shape：
For example,

(6,2) %  2 => (2,1)
(6,2) %  3 => (3,1)
(6,2) %  6 => (6,1)
(6,2) % 12 => (6,2)
(3,6,2,8) %  6 => (3,2,1,1)
(3,6,2,8) %  9 => (3,3,1,1)
(1,2,2,8) %  2 => (1,2,1,1)
(1,2,2,8) % 16 => (1,2,2,4)

最终layout的shape = (1, 1, 1, 4): (9*w, 3*x, y, z)
**/
void test_composition()
{
  auto a1 =
      Layout<Shape<_3, Shape<_6, _2>, _8>, Stride<_1, Stride<_6, _2>, _2>>{};
  auto b1 = Layout<Shape<_4, _3>, Stride<_3, _1>>{};

  auto a = coalesce(a1);
  auto b = coalesce(b1);

  std::cout << "a: ";
  print(a);
  std::cout << std::endl;
  std::cout << "b: ";
  print(b);
  std::cout << std::endl;
  auto result = composition(a, b);

  /*
  A = (3, (6,2), 8):(1, (6,2), 2) = (3, 6, 2, 8):(1, 6, 2, 2)
  B = (4, 3):(3, 1) = (B0, B1) = ((4:3), (3:1))
  B0 = (4:3)
  B1 = (3:1)

  A o B = (A o B0, A o B1)

  A o B0
  1. 中间layput
    shape = (3, 6, 2, 8) / 3 = (1, 6, 2, 8)
    stride = (3, 6, 2, 2)

  2. 取mod
    shape = (1, 6, 2, 8) % 4 = (1, 4, 1, 1)

  3. 最终layout
    shape = (1, 4, 1, 1)
    stride = (3, 6, 2, 2)

  A o B1
  1. 中间layput
    shape = (3, 6, 2, 8) / 1 = (3, 6, 2, 8)
    stride = (1, 6, 2, 2)

  2. 取mod
    shape = (3, 6, 2, 8) % 3 = (3, 1, 1, 1)


  3. 最终layout
    shape = (3, 1, 1, 1)
    stride = (1, 6, 2, 2)

  A o B = ((4:6),(3:1)) = (4,3):(6,1)

  */

  std::cout << "result: ";
  print(result);
  std::cout << std::endl;

  print(result); // equal to a but simpler
}

// by mode composition
// https://github.com/Rhys-Q/cutlass/blob/main/media/images/cute/composition1.png
// 这张图很好解释了composition的本质，就是从A中按照B定义的规则
// 进行截取，可以实现任意截取
void test_by_mode_composition()
{
  auto a = make_layout(make_shape(12, make_shape(4, 8)),
                       make_stride(59, make_stride(13, 1)));

  auto tiler = make_tile(Layout<_3, _4>{}, Layout<_8, _2>{});
  auto result = composition(a, tiler);

  print(result);
  std::cout << std::endl;

  auto same_result = make_layout(composition(layout<0>(a), get<0>(tiler)),
                                 composition(layout<1>(a), get<1>(tiler)));

  print(same_result);
  std::cout << std::endl;
}

/**
tiler可以是以下三种：
1. 一个Layout
2. tuple of Layout，即它是支持嵌套的
3. Shape，会被解释为tuple of Layout with stride-1

tile的作用就是用于composition的第二个参数，对A进行截取，可以实现任意截取

比如对一个MxNxL的tensor，获取它的3x5x8的子layout
也可以将一个8x16的tensor，reorder为32x4的tensor


 */

/*
Complement 补集

complement(A, M)
M是一个shape，一般是一个int，表示一个整数
补集合就是计算一个layout，使得A和补集合的并集是M

*/
void test_complement()
{
  auto layout = Layout<_1, _0>{};
  auto result = complement(layout, Int<16>{});
  print(result);
  std::cout << std::endl;
}

/*
Division (Tiling)

template <class LShape, class LStride,
          class TShape, class TStride>
auto logical_divide(Layout<LShape,LStride> const& layout,
                    Layout<TShape,TStride> const& tiler)
{
  return composition(layout, make_layout(tiler, complement(tiler,
size(layout))));
}

A⊘B := A o (B,B*)
B* = complement(B, size(A))

将A分为两个部分，第一个部分是B所截取的，第二个部分是B的补集所截取的，也就是B没有截取的

After the divide, the first mode of the result is the tile of data and the
second mode of the result iterates over each tile.

第一个mode是tile of data，第二个mode是iterates over each tile.
https://github.com/Rhys-Q/cutlass/blob/main/media/images/cute/divide1.png
*/
void test_logical_divide()
{
  auto layout = Layout<Shape<_4, _2, _3>, Stride<_2, _1, _8>>{};
  auto tiler = Layout<_4, _2>{};
  print(layout);
  print(tiler);
  // print_latex(layout);
  auto result = logical_divide(layout, tiler);
  auto result1 = tiled_divide(layout, tiler);
  print(result);
  print_latex(result);
  print_latex(result1);
  std::cout << std::endl;
}

/*
zipped Tiled Flat Divides

Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))
tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)
flat_divide    : (TileM, TileN, RestM, RestN, L, ...)

是对logical_divide的封装，将结果的第二个mode进行聚合，得到不同的结果

layout<0>(zipped_divide(a, b)) == composition(a, b)
layout<1>(zipped_divide(a, b)) == composition(a,complement(b, size(a)))
*/

/*
Product(Tiling)

logical_product(Layout, Layout)
A ⊗ B := (A, A* o B)

template <class LShape, class LStride,
          class TShape, class TStride>
auto logical_product(Layout<LShape,LStride> const& layout,
                     Layout<TShape,TStride> const& tiler)
{
  return make_layout(layout, composition(complement(layout,
size(layout)*cosize(tiler)), tiler));
}

I can not understand totally, maybe I need to read the paper again.
*/

/*
Zipped and Tiled Products
Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_product : ((M,TileM), (N,TileN), L, ...)
zipped_product  : ((M,N), (TileM,TileN,L,...))
tiled_product   : ((M,N), TileM, TileN, L, ...)
flat_product    : (M, N, TileM, TileN, L, ...)
*/
int main()
{
  test_logical_divide();
  return 0;
}