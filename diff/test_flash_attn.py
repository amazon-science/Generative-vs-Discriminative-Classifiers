import torch
from flash_attn import flash_attn_varlen_qkvpacked_func
import random

def test_flash_attention(batch_size=4, max_seq_len=128, head_dim=64, n_heads=12, seed=42, dtype=torch.bfloat16):
    """
    Test flash attention with fixed and variable length sequences,
    including both forward and backward passes
    """
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device('cuda')

    def run_flash_attn(qkv, cu_seqlens, max_seqlen, name=""):
        """Run forward and backward pass and collect statistics"""
        try:
            # Forward pass
            qkv.requires_grad = True
            
            # Ensure inputs are in correct dtype
            qkv = qkv.to(dtype)
            # Ensure cu_seqlens is int32
            cu_seqlens = cu_seqlens.to(torch.int32)
            
            print(f"\n{name} Input stats:")
            print(f"QKV dtype: {qkv.dtype}")
            print(f"cu_seqlens dtype: {cu_seqlens.dtype}")
            print(f"QKV shape: {qkv.shape}")
            print(f"cu_seqlens: {cu_seqlens}")
            
            with torch.cuda.amp.autocast(dtype=dtype):
                out = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens,
                    max_seqlen,
                    0.,
                    causal=False
                )
            
            forward_stats = {
                'min': out.min().item(),
                'max': out.max().item(),
                'mean': out.mean().item(),
                'has_nan': torch.isnan(out).any().item()
            }
            print(f"\n{name} Forward pass stats:")
            print(forward_stats)

            # Backward pass
            try:
                # Use mean as loss for simplicity
                loss = out.mean()
                loss.backward()
                
                grad_stats = {
                    'min': qkv.grad.min().item(),
                    'max': qkv.grad.max().item(),
                    'mean': qkv.grad.mean().item(),
                    'has_nan': torch.isnan(qkv.grad).any().item(),
                    'grad_norm': qkv.grad.norm().item()
                }
                print(f"{name} Backward pass stats:")
                print(grad_stats)
                
                return {
                    'success': True,
                    'forward_stats': forward_stats,
                    'grad_stats': grad_stats
                }
            
            except Exception as e:
                print(f"{name} Backward pass failed with error:", str(e))
                return {
                    'success': False,
                    'forward_stats': forward_stats,
                    'error': str(e)
                }

        except Exception as e:
            print(f"{name} Forward pass failed with error:", str(e))
            print(f"QKV shape: {qkv.shape}")
            print(f"cu_seqlens: {cu_seqlens}")
            return {
                'success': False,
                'error': str(e)
            }

    # Generate random QKV tensors
    def generate_qkv(total_tokens):
        qkv = torch.randn(total_tokens, 3, n_heads, head_dim, device=device, dtype=dtype)
        qkv = qkv / qkv.norm(dim=-1, keepdim=True)  # normalize
        return qkv

    print("\n=== Testing Fixed Length Sequences ===")
    # All sequences have length max_seq_len
    total_tokens = batch_size * max_seq_len
    qkv_fixed = generate_qkv(total_tokens)
    cu_seqlens_fixed = torch.arange(
        0, (batch_size + 1) * max_seq_len, 
        step=max_seq_len, 
        dtype=torch.int32,  # Explicitly set dtype
        device=device
    )
    print(f"Fixed length cu_seqlens: {cu_seqlens_fixed}")
    result_fixed = run_flash_attn(qkv_fixed, cu_seqlens_fixed, max_seq_len, "Fixed Length")

    print("\n=== Testing Variable Length Sequences ===")
    # Generate variable sequence lengths
    seq_lens = torch.tensor(
        [random.randint(max_seq_len//2, max_seq_len) for _ in range(batch_size)],
        dtype=torch.int32,  # Explicitly set dtype
        device=device
    )
    total_tokens_var = seq_lens.sum().item()
    qkv_var = generate_qkv(total_tokens_var)
    cu_seqlens_var = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),  # Explicitly set dtype
        seq_lens.cumsum(0)
    ]).to(torch.int32)  # Ensure int32
    print(f"Variable length cu_seqlens: {cu_seqlens_var}")
    print(f"Sequence lengths: {seq_lens}")
    result_var = run_flash_attn(qkv_var, cu_seqlens_var, max_seq_len, "Variable Length")

    print("\n=== Testing Very Uneven Sequences ===")
    # Generate very uneven sequence lengths
    seq_lens_uneven = torch.tensor(
        [max_seq_len] + [random.randint(1, max_seq_len//4) for _ in range(batch_size-1)],
        dtype=torch.int32,  # Explicitly set dtype
        device=device
    )
    total_tokens_uneven = seq_lens_uneven.sum().item()
    qkv_uneven = generate_qkv(total_tokens_uneven)
    cu_seqlens_uneven = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),  # Explicitly set dtype
        seq_lens_uneven.cumsum(0)
    ]).to(torch.int32)  # Ensure int32
    print(f"Uneven length cu_seqlens: {cu_seqlens_uneven}")
    print(f"Sequence lengths: {seq_lens_uneven}")
    result_uneven = run_flash_attn(qkv_uneven, cu_seqlens_uneven, max_seq_len, "Uneven Length")

    # Test with different scale factors
    print("\n=== Testing Different Scale Factors with Variable Length ===")
    scales = [0.1, 1.0, 10.0, 100.0]
    for scale in scales:
        print(f"\nScale factor: {scale}")
        qkv_scaled = generate_qkv(total_tokens_var) * scale  # Create new tensor for each scale
        result_scaled = run_flash_attn(
            qkv_scaled, 
            cu_seqlens_var, 
            max_seq_len, 
            f"Scale {scale}"
        )

    # Test with padding token positions
    print("\n=== Testing with Padding Token Positions ===")
    attention_mask = torch.ones(batch_size, max_seq_len, device=device)
    for i in range(batch_size):
        pad_len = random.randint(0, max_seq_len//2)
        attention_mask[i, -pad_len:] = 0
    
    seq_lens_pad = attention_mask.sum(dim=1).to(torch.int32)  # Explicitly set dtype
    total_tokens_pad = seq_lens_pad.sum().item()
    qkv_pad = generate_qkv(total_tokens_pad)
    cu_seqlens_pad = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),  # Explicitly set dtype
        seq_lens_pad.cumsum(0)
    ]).to(torch.int32)  # Ensure int32
    print(f"Padding mask sequence lengths: {seq_lens_pad}")
    result_pad = run_flash_attn(qkv_pad, cu_seqlens_pad, max_seq_len, "Padding Masks")

def print_separator():
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    # Test with both fp16 and bf16
    for dtype in [torch.float16, torch.bfloat16]:
        print_separator()
        print(f"Testing with dtype: {dtype}")
        
        print_separator()
        print("Testing with small batch size")
        test_flash_attention(batch_size=4, dtype=dtype)
        
        print_separator()
        print("Testing with medium batch size")
        test_flash_attention(batch_size=8, dtype=dtype)
        
        print_separator()
        print("Testing with large batch size")
        test_flash_attention(batch_size=16, dtype=dtype)
