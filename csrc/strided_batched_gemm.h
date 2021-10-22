#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>

//#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

// symbol to be automatically resolved by PyTorch libs
extern THCState *state;

cublasOperation_t convertTransToCublasOperation(char trans) {
  if (trans == 't') return CUBLAS_OP_T;
  else if (trans == 'n') return CUBLAS_OP_N;
  else if (trans == 'c') return CUBLAS_OP_C;
  else {
    AT_ERROR("trans must be one of: t, n, c");
    return CUBLAS_OP_T;
  }
}

void CublasStridedBatchedGemm(THCState *state, char transa, char transb, long m, long n, long k,
                    float alpha, const half *a, long lda, long strideA, const half *b, long ldb, long strideB,
                    float beta, half *c, long ldc, long strideC, long batchCount, cublasGemmAlgo_t algo=CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
    cublasOperation_t opa = convertTransToCublasOperation(transa);
    cublasOperation_t opb = convertTransToCublasOperation(transb);

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);
    float fAlpha = alpha;
    float fBeta = beta;
    //THCublasCheck(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(handle,
                                     opa, opb, (int)m, (int)n, (int)k,
                                     (void*)&fAlpha, a, CUDA_R_16F, (int)lda, strideA,
                                     b, CUDA_R_16F, (int)ldb, strideB,
                                     (void*)&fBeta, c, CUDA_R_16F, (int)ldc, strideC,
                                     (int)batchCount, CUDA_R_32F, algo));
    //THCublasCheck(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
}
