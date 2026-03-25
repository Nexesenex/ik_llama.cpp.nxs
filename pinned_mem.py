import torch
import sys

def test_max_pinned_memory(max_test_mb: int = 196608):
    if not torch.cuda.is_available():
        print("CU not available. Make sure the NVIDIA driver is loaded and your A4000 is device 0.")
        sys.exit(1)

    print(f"Testing maximum pinned memory on: {torch.cuda.get_device_name(0)} (CUDA device 0 = your A4000 in TCC)")
    print("Using torch.empty(..., pin_memory=True) → this is exactly what llama.cpp calls.\n")

    low = 0
    high = max_test_mb
    max_success_mb = 0

    while low <= high:
        mid = (low + high) // 2
        size_bytes = mid * 1024**2   # mid MiB in bytes

        try:
            tensor = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True, device='cpu')
            print(f"✓ SUCCESS: {mid} MiB pinned")
            max_success_mb = mid
            del tensor
            torch.cuda.empty_cache()   # free it immediately
            low = mid + 1
        except RuntimeError as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or "cuda" in err_str:
                print(f"✗ FAILED at {mid} MiB → {e}")
                high = mid - 1
            else:
                print(f"⚠ Unexpected error: {e}")
                break

    print("\n" + "="*60)
    print(f"MAXIMUM SUCCESSFUL PINNED ALLOCATION = {max_success_mb} MiB ({max_success_mb / 1024:.2f} GiB)")
    print("="*60)
    print("Use this number to size your llama.cpp parameters:")
    print(f"   -cuda ...offload-batch-size={max_success_mb - 10240 if max_success_mb > 20480 else 32768}...")
    print("   (leave ~10 GiB headroom for OS + other buffers)")

if __name__ == "__main__":
    test_max_pinned_memory()