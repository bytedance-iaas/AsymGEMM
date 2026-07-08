  For each layer/tensor, decide:

  If tensor is small:
      keep it

  If tensor is large and cheap to recompute:
      recompute it

  If tensor is large and expensive to recompute, and it has long lifetime:
      offload it

  If tensor is large but needed very soon:
      keep it if possible




  For a N-layer model:

  layers 0% - 50%:
      allow CPU offload for large expensive tensors
      allow recompute for cheap tensors

  layers 50% - 80%:
      recompute cheap tensors
      offload only if tensor is huge and reload can be prefetched

  layers 80% - 100%:
      avoid CPU offload
      keep in HBM if possible
      only recompute cheap elementwise activations if needed