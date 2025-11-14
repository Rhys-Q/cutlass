
/*
这个文件是心啊了基础的矩阵乘法，包括：
1. 将矩阵partition，然后从global memory 加载到 CTA
2. 使用cute::copy和cute::gemm写mainloop，实现gemm运算
*/

/*
high level接口
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC,
            Alpha alpha, Beta beta)

有很多的模版参数：
1. ProlemShape： 矩阵乘法的MNK
2. CtaTiler：cute的tile的概念，决定如何从MNK problem shape中划分出CTA的shape
3. TA const* A, TB const* B, TC * C： 矩阵A，B，C的数据类型
4. AStride, BStride, CStride：矩阵A，B，C的stride
5. ASmemLayout, BSmemLayout, CSmemLayout：每个CTA，A、B、C矩阵在shared memory中的layout
6. Alpha alpha, Beta beta： 矩阵乘法的缩放系数
*/
