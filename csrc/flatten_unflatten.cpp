#include <torch/extension.h>

at::Tensor flatten(std::vector<at::Tensor> tensors) {
  return at::flatten_dense_tensors(tensors);
}

/// Reference: https://github.com/pytorch/pytorch/blob/9b639f263d696ea571a61a13e712ffafa634545f/aten/src/ATen/native/TensorShape.cpp#L2709-L2727
std::vector<at::Tensor> unflatten(at::Tensor flat, std::vector<at::Tensor> tensors) {
  std::vector<at::Tensor> outputs;
  outputs.reserve(tensors.size());
  size_t offset{0};
  for (const auto & t : tensors) {
    const auto numel{t.numel()};
    // If unflatten an empty tensor, create a new empty tensor using
    // flat tensor Options.
    // This can avoid the unflattened empty tensor to share the same storage
    // with other unflatten tensors.
    if (numel == 0) {
      outputs.push_back(at::empty({0}, flat.options()));
    } else {
      outputs.push_back(flat.narrow(0, offset, numel).view(t.sizes()).to(t.suggest_memory_format()));
      offset += numel;
    }
  }
  return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flatten", &flatten, "Flatten dense tensors");
  m.def("unflatten", &unflatten, "Unflatten dense tensors");
}
