#include "cute/int_tuple.hpp"
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
int main() {
  test_int_tuple();
  return 0;
}