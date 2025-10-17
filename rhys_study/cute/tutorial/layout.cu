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

void test_natural_coord() {
  auto shape = Shape<_3, Shape<_2, _3>>{};
  print(idx2crd(16, shape));                                    // (1,(1,2))
  print(idx2crd(_16{}, shape));                                 // (_1,(_1,_2))
  print(idx2crd(make_coord(1, 5), shape));                      // (1,(1,2))
  print(idx2crd(make_coord(_1{}, 5), shape));                   // (_1,(1,2))
  print(idx2crd(make_coord(1, make_coord(1, 2)), shape));       // (1,(1,2))
  print(idx2crd(make_coord(_1{}, make_coord(1, _2{})), shape)); // (_1,(1,_2))
}

void test_index_mapping() {
  auto shape = Shape<_3, Shape<_2, _3>>{};
  auto stride = Stride<_3, Stride<_12, _1>>{};
  print(crd2idx(16, shape, stride));                              // 17
  print(crd2idx(_16{}, shape, stride));                           // _17
  print(crd2idx(make_coord(1, 5), shape, stride));                // 17
  print(crd2idx(make_coord(_1{}, 5), shape, stride));             // 17
  print(crd2idx(make_coord(_1{}, _5{}), shape, stride));          // _17
  print(crd2idx(make_coord(1, make_coord(1, 2)), shape, stride)); // 17
  print(
      crd2idx(make_coord(_1{}, make_coord(_1{}, _2{})), shape, stride)); // _17
}

void test_sublayout() {
  Layout a = Layout<Shape<_4, Shape<_3, _6>>>{}; // (4,(3,6)):(1,(4,12))
  Layout a0 = layout<0>(a);                      // 4:1
  Layout a1 = layout<1>(a);                      // (3,6):(4,12)
  Layout a10 = layout<1, 0>(a);                  // 3:4
  Layout a11 = layout<1, 1>(a);                  // 6:12
}

void test_concatenation() {
  Layout a = Layout<_3, _1>{};                  // 3:1
  Layout b = Layout<_4, _3>{};                  // 4:3
  Layout row = make_layout(a, b);               // (3,4):(1,3)
  Layout col = make_layout(b, a);               // (4,3):(3,1)
  Layout q = make_layout(row, col);             // ((3,4),(4,3)):((1,3),(3,1))
  Layout aa = make_layout(a);                   // (3):(1)
  Layout aaa = make_layout(aa);                 // ((3)):((1))
  Layout d = make_layout(a, make_layout(a), a); // (3,(3),3):(1,(1),1)
}

void test_concatenation_append() {
  Layout a = Layout<_3, _1>{}; // 3:1
  Layout b = Layout<_4, _3>{}; // 4:3
  Layout ab = append(a, b);    // (3,4):(1,3)
  Layout ba = prepend(a, b);   // (4,3):(3,1)
  Layout c = append(ab, ab);   // (3,4,(3,4)):(1,3,(1,3))
  Layout d = replace<2>(c, b); // (3,4,4):(1,3,3)
}

int main() {
  test_concatenation_append();
  return 0;
}