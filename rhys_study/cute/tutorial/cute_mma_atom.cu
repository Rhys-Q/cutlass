
/*
Cute's support for Matrix Multiply-Accumulate instructions

MMA是指矩阵乘加指令，每个不同型号的GPU，MMA的实现可能是不一样的。MMA的指令可能不一样。
但是cute layout封装了一下，用户可以使用cute提供的mma接口完成mma计算，而不用关心底层的实现细节。
实现步骤如下：
1. 我们将每个mma ptx 指令封装成一个 "Operation" 结构体。
2. 对于每个"Operation" 结构体， 我们定义了对应的"Traits"结构体，它包含了这个Operation所需要的meta-information。
3. 我们将Operation和Traits结合起来，定义了一个Atom 类，它提供了构造cute::Tensor fragments和 在cute::Tensor上执行Operation的方式
4. 因为有可能出现需要多个Atoms的场景，我们构造了TiledMMA类
*/

/*
CUTE MMA Atoms
Operation 结构定义了ptx指令，定义了这个指令需要的参数和接口。Operation结构体有最小的软件依赖：
它不使用layout、tensor、等非标准的数值数据类型，仅仅使用物理的输入输出。
不同的Operation结构体有不同的MMA指令。

对应的MMA_Traits结构体定义了Operation所需要的meta-information。比如计算的数据类型、shape、threads的layout等。
MMA_Traits 将Operation作为模版参数，cute为每个Operation都实现了它对应的MMA_Traits结构体。

这样的设计，可以将thread、layout、数据类型、shape等信息和具体的ptx指令解耦开来。

CUTE MMA 支持很多硬件level，包括：
1. 单独一个thread：比如fused multiply add (FMA) 指令
2. a quadpair (Volta)
3. 一个单独的warp（Ampere）
4， 一个warpgroup (Hopper)
*/

/*
Operation structs

SM75_16x8x8_F32F16F16F32_TN

表示M=16, N=8, K=8的矩阵乘加指令
MMA的计算可以表示为： D =A * B + C
对应的dtype
D FP32 A FP16 B FP16 C FP32
这在ptx的指令也有体现：mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32

TN表示 A是 row-major， B是col-major，这在ptx指令中也有体现。

Operation结构体有四个公开的type alias: DRegister、ARegister、BRegister、CRegister
例如SM70_8x8x4_F32F16F16F32_NT，定义如下
using DRegisters = float[8];
using ARegisters = uint32_t[2];
using BRegisters = uint32_t[2];
using CRegisters = float[8];

这样子8个thread就可以完成一次mma的操作，一个warp完成4次mma操作。

fma static member device functon
Operation结构体定义了一个fma函数 用CUTE_HOST_DEVICE 修饰符修饰，它实现具体的mma计算。
*/

/*
Traits

MMA_Traits定义了以下type alias:
ValTypeD: D的逻辑计算类型
ValTypeA: A的逻辑计算类型
ValTypeB: B的逻辑计算类型
ValTypeC: C的逻辑计算类型
Shape_MNK: MMA Operation的MNK shape
ThrID: MMA operation的thread mapping
ALayout: 用(thread, value)对A矩阵进行layout映射

BLayout: Mapping of (thread,value) pairs to coordinates in the NxK B matrix

CLayout: Mapping of (thread,value) pairs to coordinates in the MxN C matrix

比如：
template <>
struct MMA_Traits<SM70_8x8x4_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = SM70_QuadPair;
  using ALayout = SM70_8x4_Col;
  using BLayout = SM70_8x4_Col;
  using CLayout = SM70_8x8_32b;
};
*/

/*
Volta

8个thread组成一个quadpair，4个quadpair组成一个warp。一个quadpair可以完成一次MMA操作。

HMMA types：
using ValTypeD = float;
using ValTypeA = half_t;
using ValTypeB = half_t;
using ValTypeC = float;

shape：
// Logical shape of the MMA
using Shape_MNK = Shape <_8,_8,_4>;

Thread ID：
一个warp内的thread 逻辑排列为[0...31]，一个mma中会包含8个thread，比如[0,1,2,3] U [16,17,18,19]组成一个quadpair
MMA的8个thread，如何map到warp的8个thread上，这个mapping关系由ThrID定义
即从mma的logical thread，到warp的logical thread的映射关系

// Mapping from (logical thread id) -> (thread idx)
using ThrID = Layout<Shape <_4, _2>,
                    Stride<_1,_16>>;

至于第二个mma，则加上offset=4就行 第二个mma的8个thread是[4,5,6,7] U [20,21,22,23]
*/

/*
Accumulator Mapping
CLayout
每个thread 负责 C/D 8个元素的load or store。TensorCore的硬件决定了每个thread负责C的哪些元素。
根据这个硬件决定的规则，就可以计算出CLayout 、DLayout
*/

/*
A 和 B的Layout Mapping
A和B的layout，取决于是否需要进行transpose。
这里说一下NT、TN。这里有一个背景知识，cutlass假设用户提供的矩阵都是row-major的。
其实也有NN、TT，但是不常用。
其他的与C/D类似，都是TensorCore定死的。
*/

/*
Hopper
Hopper的MMA更大，称为Group MMA，是第一次出现。
GMMA需要128个thread来完成一次运算，即4个warp。因此这4个warp称为warpgroup。

Thread ID
using ThrID = Layout<_128, _1>;
Thread ID负责将gmma的logical thread 映射到warpgroup的logical thread上。
直接[0,1,...,127]映射=> [0,1,...,127]即可。

Accumulator Mapping
在GMMA中，acc的mapping是分层mapping。
首先是一个core matrix，在fp16的数据类型下，core matrix的shape是8x8。在fp32的情况下，shape是8x4。
有点复杂啊，后面遇到hopper再说吧！
*/

/*
TiledMMAs

我们可以将多个Atoms组合起来

  TiledMMA mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                                Layout<Shape <_2,_2>,
                                        Stride<_2,_1>>{});   // 2x2 n-major layout of Atoms
  print_latex(mma);

  这样就组合为一个16x16x4的mma操作。


  TiledMMA mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                                  Layout<Shape <_2,_2>,
                                         Stride<_2,_1>>{},  // 2x2 n-major layout of Atoms
                                  Tile<_32,_32,_4>{});      // 32x32x4 tiler
    print_latex(mma);

32x32x4的mma操作

*/
int main()
{
}