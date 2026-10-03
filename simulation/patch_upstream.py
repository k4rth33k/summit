"""Small, guarded adaptations of the pinned Apache-2.0 Phantora sources.

Keep the current replay/VRAM implementation, but link its Rust estimator to
tch 0.20 (PyTorch 2.7.1). FSDP calls ncclBroadcast; upstream implements the
equivalent in-place ncclBcast event but leaves this entry point unimplemented.
No tensor bytes are computed by either simulator collective.
"""
from pathlib import Path
import sys


def patch(root):
    cargo = root / "phantora/phantora/Cargo.toml"
    source = cargo.read_text()
    old = 'rev = "f7fb153bd46c0df5f1107f3748b5595d6c39a1dc"'
    if source.count(old) != 1:
        raise RuntimeError("Phantora tch dependency changed")
    cargo.write_text(source.replace(old, 'rev = "c71bac49e98e06c5427f4ea8ddcf6fda704e1e30"'))
    nccl = root / "stub/nccl.c"
    source = nccl.read_text()
    start, end = source.index("ncclBroadcast("), source.index("ncclSend(")
    block = source[start:end]
    if block.count("NOT_IMPLEMENTED;") != 1:
        raise RuntimeError("Phantora ncclBroadcast implementation changed")
    block = block.replace("NOT_IMPLEMENTED;", "return ncclBcast(recvbuff, count, datatype, root, comm, stream);")
    nccl.write_text(source[:start] + block + source[end:])
    patch_cuda_loader(root)
    patch_triangular_solve(root)


def patch_cuda_loader(root):
    cuda = root / "stub/cuda.c"
    source = cuda.read_text()
    if source.count("\ncuStreamWriteValue32(") != 1:
        raise RuntimeError("Phantora CUDA stream-write adapter anchor changed")
    # CUDA 12.8 headers rename the existing definition to *_v2. PyTorch 2.7.1
    # dlsym still requests the legacy spelling while initializing its driver
    # table, including on the OOM diagnostic path. Supply that symbol without
    # pretending to implement stream writes if an unqualified path calls it.
    source += '''
#undef cuStreamWriteValue32
CUresult CUDAAPI
cuStreamWriteValue32(CUstream stream, CUdeviceptr addr,
                     cuuint32_t value, unsigned int flags)
{
    return CUDA_ERROR_NOT_SUPPORTED;
}
'''
    cuda.write_text(source)


def patch_triangular_solve(root):
    cublas = root / "stub/cublas_noimpl.c"
    source = cublas.read_text()
    start = source.index("cublasStrsmBatched(")
    end = source.index("cublasDtrsmBatched(", start)
    block = source[start:end]
    if block.count("NOT_IMPLEMENTED;") != 1:
        raise RuntimeError("Phantora batched triangular-solve adapter anchor changed")
    # Qwen3.5's reference Gated DeltaNet uses float32 triangular solves.
    # As with upstream's GEMM stubs, PyTorch allocates the actual tensors and
    # drives autograd; no numerical kernel executes here. Validate scalar
    # dimensions/enums, but never read fake device pointer-array contents.
    block = block.replace("NOT_IMPLEMENTED;", '''
    if ((side != CUBLAS_SIDE_LEFT && side != CUBLAS_SIDE_RIGHT) ||
        (uplo != CUBLAS_FILL_MODE_LOWER && uplo != CUBLAS_FILL_MODE_UPPER) ||
        (trans != CUBLAS_OP_N && trans != CUBLAS_OP_T && trans != CUBLAS_OP_C) ||
        (diag != CUBLAS_DIAG_NON_UNIT && diag != CUBLAS_DIAG_UNIT) ||
        m < 0 || n < 0 || batchCount < 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    int order = side == CUBLAS_SIDE_LEFT ? m : n;
    if (lda < (order > 0 ? order : 1) || ldb < (m > 0 ? m : 1))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0 || batchCount == 0)
        return CUBLAS_STATUS_SUCCESS;
    if (!alpha || !A || !B)
        return CUBLAS_STATUS_INVALID_VALUE;
    return CUBLAS_STATUS_SUCCESS;''')
    cublas.write_text(source[:start] + block + source[end:])


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
