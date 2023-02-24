.. module:: apex.transformer

apex.transformer
================

Functionalities based off of https://github.com/nvidia/megatron-lm with some modifications for NeMo Megatron.

.. autosummary::
   :toctree:
   :nosignatures:

   apex.transformer.functional.fused_softmax.FusedScaleMaskSoftmax
   apex.transformer.functional.fused_softmax.GenericFusedScaleMaskSoftmax
   apex.transformer.parallel_state.initialize_model_parallel
   apex.transformer.pipeline_parallel.get_forward_backward_func
   apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining.forward_backward_no_pipelining
   apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_with_interleaving._forward_backward_pipelining_with_interleaving
   apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving.forward_backward_pipelining_without_interleaving
   apex.transformer.tensor_parallel.vocab_parallel_cross_entropy
   apex.transformer.tensor_parallel.broadcast_data
   apex.transformer.tensor_parallel.ColumnParallelLinear
   apex.transformer.tensor_parallel.RowParallelLinear
   apex.transformer.tensor_parallel.VocabParallelEmbedding
   apex.transformer.tensor_parallel.model_parallel_cuda_manual_seed
   apex.transformer.utils.split_tensor_into_1d_equal_chunks
   apex.transformer.utils.gather_split_1d_tensor
