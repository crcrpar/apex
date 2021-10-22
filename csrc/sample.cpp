#include <torch/extension.h>

#include "strided_batched_gemm.h"

void noop(const at::Tensor & x) { return; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("noop", &noop,
        "Computes and apply update for LAMB optimizer");
}
