# Two-GPU Parallel Training

After preparing the one shared hard-detected fault manifest, launch the two
independent training processes concurrently. Assign fanin-mean to physical
GPU 0 and level-GAT-GRU to physical GPU 1 through per-process
CUDA_VISIBLE_DEVICES. Each child therefore continues to use its existing
cuda:0 device logic while addressing a different physical GPU.

Keep separate output, checkpoint, TensorBoard and log directories. Use 20
rounds while preserving the 2000-backtrack contract, seeds and resume behavior.
Prefix interleaved console lines by model while keeping log files unchanged.
If either child fails, terminate the other and do not build the benchmark
bundle. Build the bundle only after both best models exist. Record physical
GPU assignments in run metadata and reject duplicate, negative or unavailable
GPU IDs before preparation starts.
