# coding=utf-8
# Copyright (c) 2021-22, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO(mkozuki): Consider removing `timers`.

from functools import reduce
import operator
from typing import Union, Optional, Tuple

import torch

from apex.transformer import parallel_state
from apex.transformer.log_util import get_transformer_logger
from apex.transformer.utils import split_tensor_into_1d_equal_chunks
from apex.transformer.utils import gather_split_1d_tensor
from apex.transformer.pipeline_parallel.utils import Shape


_logger = get_transformer_logger(__name__)


class FutureTensor:
    def __init__(self, tensor: torch.Tensor, waitfunc) -> None:
        self.tensor = tensor
        self.waitfunc = waitfunc

    def get(self) -> torch.Tensor:
        if self.waitfunc is not None:
            res = self.waitfunc()
            if isinstance(res, torch.Tensor):
                self.tensor = res
            self.waitfunc = None
        return self.tensor


def _run_p2pops(
    tensor_send_prev: Union[torch.Tensor, None],
    tensor_send_next: Union[torch.Tensor, None],
    tensor_recv_prev: Union[torch.Tensor, None],
    tensor_recv_next: Union[torch.Tensor, None],
    async_comm: bool = False
):
    ops = []
    if tensor_send_prev is not None:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_prev,
            parallel_state.get_pipeline_model_parallel_prev_rank(),
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_prev,
            parallel_state.get_pipeline_model_parallel_prev_rank(),
        )
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_next,
            parallel_state.get_pipeline_model_parallel_next_rank(),
        )
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_next,
            parallel_state.get_pipeline_model_parallel_next_rank(),
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        if async_comm:
            assert len(reqs) == len(ops)
            tensor_send_prev_req = None if tensor_send_prev is None else reqs.pop(0)
            tensor_recv_prev_req = None if tensor_recv_prev is None else reqs.pop(0)
            tensor_send_next_req = None if tensor_send_next is None else reqs.pop(0)
            tensor_recv_next_req = None if tensor_recv_next is None else reqs.pop(0)
            return (tensor_send_prev_req, tensor_recv_prev_req, tensor_send_next_req, tensor_recv_next_req)
        else:
            for req in reqs:
                req.wait()
            return (None, None, None, None)
    return (None, None, None, None)


def _communicate(
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Optional[Shape] = None,
    dtype: Optional[torch.dtype] = None,
    *,
    async_comm: bool = False,
    disable_chunk_to_optimize_p2p: bool = False,
) -> Tuple[Union[torch.Tensor, FutureTensor, None], Union[torch.Tensor, FutureTensor, None]]:
    """Base function for communication of tensors between stages.


    .. note::
        Reference https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/cfd2e2160700b7f2c1bf35298ac14bc341f4c759/megatron/p2p_communication.py#L24-L159

    dtype logic: If none of ``dtype_``, ``params_dtype``, ``fp32_residual_connection`` is specified,
    torch.float32 is used.

    See https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/arguments.py#L145-L159
    for the details of arguments of ``dtype_``, ``params_dtype``, ``fp32_residual_connection``.

    Args:
        tensor_send_next: tensor to send to next rank (no tensor sent if set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if set to None).
        recv_prev: boolean for whether tensor should be received from previous rank.
        recv_next: boolean for whether tensor should be received from next rank.
        tensor_shape: optional, use when the input sequence contains less tokens than the default sequence length
        dtype: dtype of tensor.

    Keyword args:
        async_comm: An experimental optimization for slower network such as UCC.
        disable_chunk_to_optimize_p2p: Flag to disable split&gather among tensor parallel group
            before&after P2P communication.

    Returns:
        tuple containing

        - tensor_recv_prev: `torch.Tensor` if `recv_prev` is :obj:`True`, `None` otherwise.
        - tensor_recv_next: `torch.Tensor` if `recv_next` is :obj:`True`, `None` otherwise.
    """
    # Create placeholder tensors for receive in forward and backward directions if needed.
    tensor_recv_prev = None
    tensor_recv_next = None
    if tensor_shape is None:
        # In megatron, `tensor_shape` is set to `(args.seq_length, args.micro_batch_size, args.hidden_size)`
        raise RuntimeError(
            "`tensor_shape` must be specified. Common `tensor_shape` is `(seq_length, micro_batch_size, hidden_size)`")

    tensor_parallel_size = parallel_state.get_tensor_model_parallel_world_size()
    tensor_chunk_size = int(reduce(operator.mul, tensor_shape, 1))
    split_tensors_to_optimize_p2p = (not disable_chunk_to_optimize_p2p) and (tensor_chunk_size % tensor_parallel_size == 0)
    if split_tensors_to_optimize_p2p:
        tensor_chunk_shape = [tensor_chunk_size // tensor_parallel_size]
    else:
        tensor_chunk_shape = tensor_shape
    requires_grad = True

    if recv_prev:
        tensor_recv_prev = torch.empty(
            tensor_chunk_shape,
            requires_grad=requires_grad,
            device=torch.cuda.current_device(),
            dtype=dtype,
        )
    if recv_next:
        tensor_recv_next = torch.empty(
            tensor_chunk_shape,
            requires_grad=requires_grad,
            device=torch.cuda.current_device(),
            dtype=dtype,
        )

    if split_tensors_to_optimize_p2p:
        if tensor_send_next is not None:
            tensor_send_next = split_tensor_into_1d_equal_chunks(tensor_send_next)
        if tensor_send_prev is not None:
            tensor_send_prev = split_tensor_into_1d_equal_chunks(tensor_send_prev)

    # Send tensors in both the forward and backward directions as appropriate.
    _, tensor_recv_prev_req, _, tensor_recv_next_req = _run_p2pops(tensor_send_prev, tensor_send_next, tensor_recv_prev, tensor_recv_next, async_comm=async_comm)

    if async_comm:
        tensor_recv_prev_waitfunc = None
        tensor_recv_next_waitfunc = None
        # TODO: investigate whether this is necessary for correctness (ref: https://github.com/pytorch/pytorch/issues/38642)
        # see also: sync added for async_comm callbacks below in gather_recv_prev_wait and gather_recv_next_wait
        if tensor_recv_prev_req is not None:
            def tensor_recv_prev_wait():
                tensor_recv_prev_req.wait()
                torch.cuda.synchronize()
            tensor_recv_prev_waitfunc = tensor_recv_prev_wait
        if tensor_recv_next_req is not None:
            def tensor_recv_next_wait():
                tensor_recv_next_req.wait()
                torch.cuda.synchronize()
            tensor_recv_next_waitfunc = tensor_recv_next_wait
    else:
        # To protect against race condition when using batch_isend_irecv().
        torch.cuda.synchronize()

    # If using scatter-gather optimization, gather smaller chunks.
    if split_tensors_to_optimize_p2p:
        if not async_comm:
            if recv_prev:
                tensor_recv_prev = (
                    gather_split_1d_tensor(tensor_recv_prev)
                    .view(tensor_shape)
                    .requires_grad_()
                )

            if recv_next:
                tensor_recv_next = (
                    gather_split_1d_tensor(tensor_recv_next)
                    .view(tensor_shape)
                    .requires_grad_()
                )
        else:
            def gather_recv_prev_wait():
                tensor_recv_prev_req.wait()
                # From @Deepak's PR https://github.com/NVIDIA/Megatron-LM/commit/27fc468964064eeb33b703c9a0b2af938d80dd14
                # A sync seems to be needed before gather otherwise losses jump around e.g., in run_gpt_minimal_test
                torch.cuda.synchronize()
                return (
                    gather_split_1d_tensor(tensor_recv_prev)
                    .view(tensor_shape)
                    .requires_grad_()
                )
            def gather_recv_next_wait():
                tensor_recv_next_req.wait()
                torch.cuda.synchronize()
                return (
                    gather_split_1d_tensor(tensor_recv_next)
                    .view(tensor_shape)
                    .requires_grad_()
                )
            tensor_recv_prev_waitfunc = gather_recv_prev_wait
            tensor_recv_next_waitfunc = gather_recv_next_wait
    if async_comm:
        future_tensor_recv_prev = None
        future_tensor_recv_next = None
        if tensor_recv_prev is not None:
            future_tensor_recv_prev = FutureTensor(tensor_recv_prev, tensor_recv_prev_waitfunc)
        if tensor_recv_next is not None:
            future_tensor_recv_next = FutureTensor(tensor_recv_next, tensor_recv_next_waitfunc)
        return future_tensor_recv_prev, future_tensor_recv_next
    return tensor_recv_prev, tensor_recv_next


def log_wrapper(func):

    name = func.__name__

    def wrapper(*args, **kwargs):

        _logger.debug(f"[{name}] start")
        func(*args, **kwargs)
        _logger.debug(f"[{name}] done")

    return wrapper


@log_wrapper
def recv_forward(
    tensor_shape: Shape,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    disable_chunk_to_optimize_p2p: bool = False,
) -> Union[torch.Tensor, FutureTensor, None]:
    """Receive tensor from previous rank in pipeline (forward receive)."""
    if parallel_state.is_pipeline_first_stage():
        return None
    input_tensor, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=True,
        recv_next=False,
        tensor_shape=tensor_shape,
        dtype=dtype,
        async_comm=async_comm,
        disable_chunk_to_optimize_p2p=disable_chunk_to_optimize_p2p,
    )
    return input_tensor


@log_wrapper
def recv_backward(
    tensor_shape: Shape = None,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    disable_chunk_to_optimize_p2p: bool = False,
) -> Union[torch.Tensor, FutureTensor, None]:
    """Receive tensor from next rank in pipeline (backward receive)."""
    if parallel_state.is_pipeline_last_stage():
        return None
    _, output_tensor_grad = _communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=True,
        tensor_shape=tensor_shape,
        dtype=dtype,
        async_comm=async_comm,
        disable_chunk_to_optimize_p2p=disable_chunk_to_optimize_p2p,
    )
    return output_tensor_grad


def send_forward(
    output_tensor: torch.Tensor,
    tensor_shape: Shape = None,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    disable_chunk_to_optimize_p2p: bool = False,
) -> None:
    """Send tensor to next rank in pipeline (forward send)."""
    if parallel_state.is_pipeline_last_stage():
        return
    _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=False,
        tensor_shape=tensor_shape,
        dtype=dtype,
        async_comm=async_comm,
        disable_chunk_to_optimize_p2p=disable_chunk_to_optimize_p2p,
    )


def send_backward(
    input_tensor_grad: torch.Tensor,
    tensor_shape: Shape,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    disable_chunk_to_optimize_p2p: bool = False,
) -> None:
    """Send tensor to previous rank in pipeline (backward send)."""
    if parallel_state.is_pipeline_first_stage():
        return
    _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=False,
        tensor_shape=tensor_shape,
        dtype_=dtype,
        async_comm=async_comm,
        disable_chunk_to_optimize_p2p=disable_chunk_to_optimize_p2p,
    )


def send_forward_recv_backward(
    output_tensor: torch.Tensor,
    tensor_shape: Shape,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    disable_chunk_to_optimize_p2p: bool = False,
) -> Union[torch.Tensor, FutureTensor, None]:
    """Batched send and recv with next rank in pipeline."""
    if parallel_state.is_pipeline_last_stage():
        return None
    _, output_tensor_grad = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=True,
        tensor_shape=tensor_shape,
        dtype=dtype,
        async_comm=async_comm,
        disable_chunk_to_optimize_p2p=disable_chunk_to_optimize_p2p,
    )
    return output_tensor_grad


def send_backward_recv_forward(
    input_tensor_grad: torch.Tensor,
    tensor_shape: Shape,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    disable_chunk_to_optimize_p2p: bool = False,
) -> Union[torch.Tensor, FutureTensor, None]:
    """Batched send and recv with previous rank in pipeline."""
    if parallel_state.is_pipeline_first_stage():
        return None
    input_tensor, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=True,
        recv_next=False,
        tensor_shape=tensor_shape,
        dtype=dtype,
        async_comm=async_comm,
        disable_chunk_to_optimize_p2p=disable_chunk_to_optimize_p2p,
    )
    return input_tensor


def send_forward_recv_forward(
    output_tensor: torch.Tensor,
    recv_prev: bool,
    tensor_shape: Shape,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    disable_chunk_to_optimize_p2p: bool = False,
) -> Union[torch.Tensor, FutureTensor]:
    """Batched recv from previous rank and send to next rank in pipeline."""
    input_tensor, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape,
        dtype=dtype,
        async_comm=async_comm,
        disable_chunk_to_optimize_p2p=disable_chunk_to_optimize_p2p,
    )
    return input_tensor


def send_backward_recv_backward(
    input_tensor_grad: torch.Tensor,
    recv_next: bool,
    tensor_shape: Shape,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    disable_chunk_to_optimize_p2p: bool = False,
) -> Union[torch.Tensor, FutureTensor]:
    """Batched recv from next rank and send to previous rank in pipeline."""
    _, output_tensor_grad = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        dtype=dtype,
        async_comm=async_comm,
        disable_chunk_to_optimize_p2p=disable_chunk_to_optimize_p2p,
    )
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
    output_tensor: torch.Tensor,
    input_tensor_grad: torch.Tensor,
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    disable_chunk_to_optimize_p2p: bool = False,
) -> Tuple[Union[torch.Tensor, FutureTensor], Union[torch.Tensor, FutureTensor]]:
    """Batched send and recv with previous and next ranks in pipeline."""
    input_tensor, output_tensor_grad = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        dtype=dtype,
        async_comm=async_comm,
        disable_chunk_to_optimize_p2p=disable_chunk_to_optimize_p2p,
    )
    return input_tensor, output_tensor_grad
