# Overview

cutedsl 将python代码转为一个自定义的IR，进行JIT编译，生产cuda kernel。

核心cute抽象
- Layouts：描述数据如何组织
- Tensors：
- Atoms：mma或者copy
- Tiled Operations：定义mma如何用

按照最新版cutedsl 4.3.0，注意，python版本必须是3.12

cutedsl的ast+tracing的方式很牛啊
https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html

``` shell
pip install nvidia-cutlass-dsl --pre --upgrade
```


cutedsl对control flow的支持很不错，在编译时，会先将python code进行ast分子，将control flow转为IR。
如果是native python control flow，则直接在compile time决策
如果是dynamic 则会emit为IR

但是不能将IR的value传递给python的control flow，不然会报错。

# For loop
cutedsl 支持三种类型的for
- range：python内置的，会直接lower为IR
- cutlass.range：和python内置的range接近，但支持先进的unrolling和pipelining control。
- cutlass.range_constexpr：在编译期间展开

``` python
@cute.jit
def control_flow_examples(bound: cutlass.Int32):
    n = 10

    # ✅ This loop is Python loop, evaluated at compile time.
    for i in cutlass.range_constexpr(n):
        cute.printf("%d\\n", i)

    # ✅ This loop is dynamic, even when bound is Python value.
    for i in range(n):
        cute.printf("%d\\n", i)

    # ❌ This loop bound is a dynamic value, not allowed in Python loop.
    # Should use `range` instead.
    for i in cutlass.range_constexpr(bound):
        cute.printf("%d\\n", i)

    # ✅ This loop is dynamic, emitted IR loop.
    for i in range(bound):
        cute.printf("%d\\n", i)

    # ✅ This loop is dynamic, emitted IR loop with unrolling
    for i in cutlass.range(bound, unroll=2):
        cute.printf("%d\\n", i)
```

# Software pipelining
Software pipelining是用于优化loop的优化技术，一般形式如下：
``` python
@cute.jit
def example():
    ...
    # build a circular buffer
    buffer = ...

    # prefetch loop
    for i in range(prefetch_stages):
        cute.copy(atom, gmem[i], buffer[i], ...)

    # main loop
    for i in range(bound):
        if i + prefetch_stages < bound:
            cute.copy(atom, gmem[i + prefetch_stages], buffer[(i + prefetch_stages) % total_stages], ...)

        use(buffer[i % total_stages])

    ...
```
这样写很冗长，cute提供了简单的方式：
``` python
@cute.jit
def example():
    ...
    # build a circular buffer
    buffer = ...

    for i in cutlass.range(bound, prefetch_stages=prefetch_stages):
        # Compiler automatically handles the pipelining:
        # - Generates prefetch loop for initial stages
        # - In main loop, prefetches future data while using current data
        cute.copy(atom, gmem[i], buffer[i % total_stages], ...)
        use(buffer[i % total_stages])  # Uses data from previous iterations

    ...
```
编译器会自动生成prefetch loop，以及main loop。这个特性还在实验中，仅在sm90及以上的架构中用。
# If-Else Statements
标准的python if / elif / else是支持的。
如果不做任何标识，会将它lower到IR。
如果做了cutlass.const_expr标识，会在编译期间决策。
``` python
@cute.jit
def main(const_var: cutlass.Constexpr, dynamic_var: cutlass.Int32):
    # ✅ This branch is Python branch, evaluated at compile time.
    if cutlass.const_expr(const_var):
        cute.printf("Const branch\\n")
    else:
        cute.printf("Const else\\n")

    # ✅ This branch is dynamic branch, emitted IR branch.
    if dynamic_var == 10:
        cute.printf("Dynamic True\\n")
    else:
        cute.printf("Dynamic False\\n")

    # ❌ Using a dynamic value with `cutlass.const_expr` is not allowed.
    if cutlass.const_expr(dynamic_var == 10):
        cute.printf("Bound is 10\\n")
```

# While Loop
``` python
@cute.jit
def main(dynamic_var: cutlass.Int32):
    n = 0

    # ✅ This is Python while loop, evaluated at compile time.
    while cutlass.const_expr(n < 10):
        cute.printf("Const branch\\n")
        n += 1

    # ✅ This is dynamic while loop, emitted IR while loop.
    while dynamic_var == 10:
        cute.printf("Dynamic True\\n")
        n += 1

    # ❌ Using a dynamic value with `cutlass.const_expr` is not allowed.
    while cutlass.const_expr(n < dynamic_var):
        n += 1
```

# Compile-Time Metaprogramming
``` python
@cute.kernel
def gemm(..., do_relu: cutlass.Constexpr):
    # main GEMM work
    ...
    if cutlass.const_expr(do_relu):    # compile-time guard
        # ReLU code is emitted only when do_relu is True
        ...
```

# 动态control flow的限制
- break、continue、pass、或者报异常，目前还不支持
- control flow中的操作，仅当tracing为True时，才会被trace。
- 在control flow中定义的value，无法被外面获取
- 在control flow中改变value的类型是不被允许的

``` python
@cute.jit
def control_flow_negative_examples(predicate: cutlass.Boolean):
    n = 10

    # ❌ This loop is dynamic, early-exit isn't allowed.
    for i in range(n):
        if i == 5:
            break         # Early-exit

    if predicate:
        val = 10
        # ❌ return from control flow body is not allowed.
        return
        # ❌ Raising exception from control flow body is not allowed.
        raise ValueError("This is not allowed")
        # ❌ Using pass in control flow body is not allowed.
        pass

    # ❌ val is not available outside the dynamic if
    cute.printf("%d\\n", val)

    if predicate:
        # ❌ Changing type of a variable in control flow body is not allowed.
        n = 10.0
```

# JIT Function Argument Generation
- jit function 默认会被假设为dynamic arguments。
- 如果参数的类型被指定为cutlass.Constexpr，那它会被视为compile-time constant。
- 如果type annotation提供了，那么cutedsl会在编译期间检查类型
- cutedsl提供了runtime checkable protocols （JitArgument和DynamicExpression）来为自定义类型生成JIT Function

## Static argument vs Dynamic argument
cutedsl同时支持static argument和dynamic argument。
- 对于静态参数，会在编译期处理掉，并不会出现在jit函数的函数签名中
- 对于动态参数，则会出现在函数签名中

``` python
import cutlass
import cutlass.cute as cute

@cute.jit
def foo(x: cutlass.Int32, y: cutlass.Constexpr):
    print("x = ", x)        # Prints x = ?
    print("y = ", y)        # Prints y = 2
    cute.printf("x: {}", x) # Prints x: 2
    cute.printf("y: {}", y) # Prints y: 2

foo(2, 2)
```

## Type safety
cutedsl 会在编译时严格检查参数的类型。

## 自定义类型
cutedsl支持自定义类型，只需要提供两个东西：
- JitArguement：是用于python在host端进行调用
  - __c_pointers__ 生成一系列ctypes指针
  - __get_mlir_types__ 生成mlir类型
  - __new_from_mlir_values__ 创建一个新的对象
- DynamicExpression：用于在device层面调用：
  - __extract_mlir_values__：生成动态表示
  - __new_from_mlir_values__：新建一个对象


```python
import cutlass
import cutlass.cute as cute

# Customized type that implements the DynamicExpression protocol
class MyDynamicExpression:
    def __init__(self, tensor, offset):
        self._tensor = tensor # Dynamic argument
        self._offset = offset # Dynamic argument

    def __extract_mlir_values__(self):
        return [self._tensor.__extract_mlir_values__(), self._offset.__extract_mlir_values__()]

    def __new_from_mlir_values__(self, values):
        return MyDynamicExpression(values[0], values[1])

@cute.kernel
def my_kernel(x: MyDynamicExpression):
    ...
```


```python 
@cutlass.register_jit_arg_adapter(MyFrameworkObject)
class MyFrameworkObjectAdapter:
    """
    Convert a 3rd party framework object to a JIT function argument with JitArgument protocol
    """

    def __init__(self, arg):
        self._arg = arg

    def __c_pointers__(self):
        # Convert the framework object to a C-ABI compatible object
        # thru its C-ABI interface
        return [self._arg.get_cabi_pointer()]

    def __get_mlir_types__(self):
        # Return the list of MLIR types the framework object represents
        return [self._arg.get_data().mlir_type]

    def __new_from_mlir_values__(self, values):
        # Convert the MLIR values back to the framework object
        return MyFrameworkObject(values[0])
```

## static layout
编译好之后，cute function就只能针对固定的shape了


直接传递torch.Tensor就行了，它会被cutedsl打上cute.mark_layout_dynamic的标签，会用动态Layut
``` python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor):
    print(tensor.layout)  # Prints (?,?):(?,1) for dynamic layout

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint16)
compiled_func = cute.compile(foo, a)
compiled_func(a)

b = torch.tensor([[11, 12], [13, 14], [15, 16]], dtype=torch.uint16)
compiled_func(b)  # Reuse the same compiled function for different shape
```

静态layout 更有利于性能优化，动态Layout 更加灵活。

# JIT Caching
cutedsl kernel被编译后，会生成一个JIT Executor。
cutedsl 的constant 参数在JIT Executor中不需要传递

JIT executor会将python runtime的参数，转为C ABI-compatible的类型，然后调用host function。

cute.compile每次都会编译，不会cache，因此我们基于此可以自己实现caching：
``` python
@cute.jit
def add(b):
   return a + b

# Define a custom cache
custom_cache = {}

a = 1
compiled_add_1 = cute.compile(add, 2)
custom_cache[1] = compiled_add_1
compiled_add_1(2) # result = 3

a = 2
compiled_add_2 = cute.compile(add, 2)
custom_cache[2] = compiled_add_2
compiled_add_2(2) # result = 4

# Use the custom cache
custom_cache[1](2) # result = 3
custom_cache[2](2) # result = 4
```

默认，cutedsl的kernel编译是会用cache的，避免重复编译，cache的key的规则如下：
- cutedsl生成的mlir bytecode
- cutedsl python source files
- cutedsl shared libraries
- cutedsl 的环境变量
根据上述内容计算一个hash值


value就是编译好的JIT Executor 实例。

```shell 
# Disable file caching while keeping in-memory cache available, defaults to False.
export CUTE_DSL_DISABLE_FILE_CACHING=True

# Maximum number of cache files allowed, defaults to 1000.
export CUTE_DSL_FILE_CACHING_CAPACITY=1000
```

注意：由于python code和MLIR之间的一致性很难维持，比如存在全局变量。因此mlir每次都会生成。


# JIT Compilation Options
有很多编译选项，见：https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_compilation_options.html
