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

int main() {
  test_dynamic_integer();
  return 0;
}