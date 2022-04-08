#include <torch/extension.h>
#include <torch/csrc/utils/tensor_flatten.h>
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/utils/tensor_flatten.h

at::Tensor flatten(std::vector<at::Tensor> tensors)
{
  return torch::utils::flatten_dense_tensors(tensors);
}

std::vector<at::Tensor> unflatten(at::Tensor flat, std::vector<at::Tensor> tensors)
{
  return torch::utils::unflatten_dense_tensors(flat, tensors);
}

std::vector<at::Tensor> unflatten_channels_last(at::Tensor flat, std::vector<at::Tensor> & tensors) {
  std::vector<at::Tensor> outputs;
  outputs.reserve(tensors.size());
  size_t offset = 0;
  for (const auto & tensor : tensors) {
    auto numel = tensor.numel();
    // If unflatten an empty tensor, create a new empty tensor using
    // flat tensor Options.
    // This can avoid the unflattened empty tensor to share the same storage
    // with other unflatten tensors.
    if (numel == 0) {
      outputs.push_back(at::empty({0}, flat.options()));
    } else {
      TORCH_CHECK(tensor.dim() == 4);
      outputs.push_back(flat.narrow(0, offset, numel).view(tensor.sizes()).to(c10::MemoryFormat::ChannelsLast));
      offset += numel;
    }
  }
  return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flatten", &flatten, "Flatten dense tensors");
  m.def("unflatten", &unflatten, "Unflatten dense tensors");
  m.def("unflatten_channels_last", &unflatten_channels_last, "Unflatten dense tensors");
}
