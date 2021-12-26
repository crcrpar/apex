#include <torch/extension.h>

#include <unordered_map>
#include <string>
#include <vector>

#include <ATen/native/utils/ParamsHash.h>

std::unordered_map<std::string, std::unordered_map<std::string, std::vector<at::Tensor>>
tensor_grouping(const std::vector<torch::Tensor>& tensors) {
  std::unordered_map<std::string, std::unordered_map<std::string, std::vector<at::Tensor>> grouped_tensor_dict;

  for (const auto &t : tensors) {
    const auto &device = t.device().str();
    const auto &dtype = t.scalar_type().toString();

    if (!grouped_tensor_dict.count(device)) {
      const auto element = std::unordered_map<std::string, std::vector<at::Tensor>>{device, std::vector<at::Tensor>{t}};
      grouped_tensor_dict.insert(device, element);
    } else {
      if (!grouped_tensor_dict[device].count(dtype)) {
        grouped_tensor_dict[device].insert(dtype, std::vector<at::Tensor>{t});
      } else {
        grouped_tensor_dict[device][dtype].push_back(t);
      }
    }
  }
  return grouped_tensro_dict;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tensor_grouping", &tensor_grouping, "Grouping tensors by device and dtype");
}
