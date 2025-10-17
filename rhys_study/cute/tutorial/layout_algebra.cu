#include "cute/int_tuple.hpp"
// #include "cute/util/print_tensor.hpp"
#include "cute/util/type_traits.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"
#include <cstdint>
#include <cute/tensor.hpp>
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
void test_coalesce() {
  auto layout = Layout<Shape<_2, Shape<_1, _6>>, Stride<_1, Stride<_6, _2>>>{};

  print(layout);

  for (int i = 0; i < size(layout); i++) {
    print(crd2idx(i, layout.shape(),
                  layout.stride())); // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
  }

  auto result = coalesce(layout); // _12:_1

  print(result); // equal to layout but simpler
}

int main() {
  test_coalesce();
  return 0;
}