import unittest

import torch

import apex_C


class TestFlattenUnflatten(unittest.TestCase):

    def test_unflatten_channels_last(self):
        grads = [torch.rand(2, i, 4, i * 2) for i in range(5, 11)]
        nhwc_grads = [t.to(memory_format=torch.channels_last) for t in grads]
        nhwc_flat_grads = apex_C.flatten(nhwc_grads)
        unflatten_nhwc_flat_grads = apex_C.unflatten_channels_last(nhwc_flat_grads, nhwc_grads)

        torch.testing.assert_close(nhwc_grads, unflatten_nhwc_flat_grads)
        torch.testing.assert_close([t.stride() for t in nhwc_grads], [t.stride() for t in unflatten_nhwc_flat_grads])


if __name__ == "__main__":
    unittest.main()
