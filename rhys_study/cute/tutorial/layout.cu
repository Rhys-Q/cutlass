#include "cute/int_tuple.hpp"
#include "cute/util/print_tensor.hpp"
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

void test_integer() {

  int size1 = 0;
  size_t size2 = 1;
  int16_t size3 = 2;

  Int<1> size4;

  int size_5 = size4;
  _1 size6;
  int size_7 = size6;

  std::cout << is_integral<int>() << std::endl;        // true
  std::cout << is_integral<float>() << std::endl;      // false
  std::cout << is_std_integral<_1>() << std::endl;     // false
  std::cout << is_std_integral<Int<1>>() << std::endl; // false
  std::cout << is_std_integral<int>() << std::endl;    // true
}
void test_tuple() {
  tuple<int, std::string> t1(1, "2sadsad");
  std::cout << t1 << std::endl;

  // get the first element
  std::cout << get<0>(t1) << std::endl;
  // get the second element
  std::cout << get<1>(t1) << std::endl;

  // make a tuple
  auto t2 = make_tuple(1, 2.0, 21);
  std::cout << t2 << std::endl;

  // malloc a tuple
  auto t3 = new tuple<int, float>(1, 2.0);
  std::cout << *t3 << std::endl;

  // free the tuple
  delete t3;
}

void test_int_tuple() {
  std::vector<int> a = {6, 3, 4};
  auto tup = make_int_tuple<5>(a, a.size(), 0); // (6,3,4,0,0)
  std::cout << tup << std::endl;

  // get the first element
  std::cout << get<0>(tup) << std::endl;
  // get the second element
  std::cout << get<1>(tup) << std::endl;

  // make a nested tuple
  auto tup2 = make_tuple(make_tuple(1, 2.5), make_tuple(3, 4.0), 5);

  std::cout << get<0, 0>(tup2) << std::endl;
  std::cout << get<0, 1>(tup2) << std::endl;
  std::cout << get<1, 0>(tup2) << std::endl;
  std::cout << get<1, 1>(tup2) << std::endl;
  std::cout << get<2>(tup2) << std::endl;

  auto tup3 =
      make_tuple(uint16_t{42}, make_tuple(Int<1>{}, int32_t{3}), Int<17>{});
  std::cout << get<0>(tup3) << std::endl;
  std::cout << get<1, 0>(tup3) << std::endl;
  std::cout << get<1, 1>(tup3) << std::endl;
  std::cout << get<2>(tup3) << std::endl;

  // rank
  std::cout << rank(tup3) << std::endl;
  std::cout << rank<0>(tup3) << std::endl;
  std::cout << rank<1>(tup3) << std::endl;
  std::cout << rank<2>(tup3) << std::endl;

  // depth
  std::cout << depth(tup3) << std::endl;
  std::cout << depth<0>(tup3) << std::endl;
  std::cout << depth<1>(tup3) << std::endl;
  std::cout << depth<2>(tup3) << std::endl;

  // size
  std::cout << size(tup3) << std::endl;
  std::cout << size<0>(tup3) << std::endl;
  std::cout << size<1>(tup3) << std::endl;
  std::cout << size<2>(tup3) << std::endl;
}

void test_shape_strides() {
  auto shape = make_shape(Int<2>{}, Int<3>{});
  print(shape);

  auto stride = make_stride(Int<2>{}, Int<3>{});
  print(stride);
}

void test_layout_basic() {
  auto layout = make_layout(make_shape(Int<2>{}, Int<3>{}),
                            make_stride(Int<2>{}, Int<3>{}));
  print(layout);
  print("\n");
  print(rank(layout));
  print("\n");
  print(get<0>(layout));
  print("\n");
  print(depth(layout));
  print("\n");

  print(shape(layout));
  print("\n");
  print(stride(layout));
  print("\n");
  print(size(layout));
  print("\n");

  print(cosize(layout));
  print("\n");
}

void test_hierarchical_access() {
  auto layout = make_layout(
      make_shape(Int<2>{}, Int<3>{}, make_shape(Int<4>{}, Int<5>{})),
      make_stride(Int<2>{}, Int<3>{}, make_stride(Int<4>{}, Int<5>{})));
  print(get<2, 0>(layout));
  print("\n");
  print(rank<2, 0>(layout));
  print("\n");
}

void test_constructing_layout() {
  Layout s8 = make_layout(Int<8>{});
  print(s8);
  print("\n");
  Layout d8 = make_layout(8);
  print(d8);
  print("\n");
  Layout s2xs4 = make_layout(make_shape(Int<2>{}, Int<4>{}));
  print(s2xs4);
  print("\n");
  Layout s2xd4 = make_layout(make_shape(Int<2>{}, 4));
  print(s2xd4);
  print("\n");
  Layout s2xd4_a =
      make_layout(make_shape(Int<2>{}, 4), make_stride(Int<12>{}, Int<1>{}));
  print(s2xd4_a);
  print("\n");
  Layout s2xd4_col = make_layout(make_shape(Int<2>{}, 4), LayoutLeft{});
  print(s2xd4_col);
  print("\n");
  Layout s2xd4_row = make_layout(make_shape(Int<2>{}, 4), LayoutRight{});
  print(s2xd4_row);
  print("\n");
  Layout s2xh4 = make_layout(make_shape(2, make_shape(2, 2)),
                             make_stride(4, make_stride(2, 1)));
  print(s2xh4);
  print("\n");

  print(congruent(shape(s2xh4), stride(s2xh4)));
  print("\n");
}

template <class Shape, class Stride>
void print2D(Layout<Shape, Stride> const &layout) {
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m, n));
    }
    printf("\n");
  }
}

void test_print2D() {
  auto layout = make_layout(make_shape(Int<2>{}, Int<3>{}),
                            make_stride(Int<2>{}, Int<3>{}));
  print2D(layout);
  print("\n");
  Layout s2xh4 = make_layout(make_shape(2, make_shape(2, 2)),
                             make_stride(4, make_stride(2, 1)));
  print_layout(s2xh4);
}

int main() {
  test_print2D();
  return 0;
}