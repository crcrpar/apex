from itertools import product
import unittest

import torch

import apex_C


class TestFlattenUnflatten(unittest.TestCase):

    def _make_inputs(self, device, memory_format, is_3d):
        if not is_3d:
            return [
                torch.rand(2, i, 4, i * 2).to(device=device, memory_format=memory_format) for i in range(5, 11)
            ]
        else:
            return [
                torch.rand(2, 3, i, 4, i * 2).to(device=device, memory_format=memory_format) for i in range(5, 11)
            ]

    def _get_formats(self, is_3d):
        return (
            (torch.contiguous_format, torch.channels_last_3d)
            if is_3d else (torch.contiguous_format, torch.channels_last)
        )

    def _test_impl(self, device, memory_format, is_3d):
        grads = self._make_inputs(device, memory_format, is_3d)
        flat_grads = apex_C.flatten(grads)
        unflatten_flat_grads = apex_C.unflatten(flat_grads, grads)

        torch.testing.assert_close(grads, unflatten_flat_grads)
        torch.testing.assert_close([t.stride() for t in grads], [t.stride() for t in unflatten_flat_grads])

    def test_cpu(self):
        for is_3d, memory_format in product((False, True), (torch.contiguous_format, torch.channels_last_3d)):
            for memory_format in self._get_formats(is_3d):
                with self.subTest(is_3d=is_3d, memory_format=memory_format):
                    self._test_impl(torch.device("cpu"), memory_format, is_3d)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA devices")
    def test_cuda(self):
        for is_3d, memory_format in product((False, True), (torch.contiguous_format, torch.channels_last_3d)):
            for memory_format in self._get_formats(is_3d):
                with self.subTest(is_3d=is_3d, memory_format=memory_format):
                    self._test_impl(torch.device("cuda"), memory_format, is_3d)


if __name__ == "__main__":
    unittest.main()
