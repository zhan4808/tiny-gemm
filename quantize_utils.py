import torch
import numpy as np

def quantize_weights_to_int4(weights, per_channel=True):
    """
    Quantize weights to INT4 format.
    
    Args:
        weights: Tensor to quantize
        per_channel: If True, quantize per output channel
        
    Returns:
        quantized_weights: INT4 quantized weights
        scales: Quantization scales
        zeros: Zero points
    """
    org_device = weights.device
    if org_device != torch.device('cpu'):
        weights = weights.to('cpu')
    
    # Determine quantization axis
    if per_channel:
        if weights.dim() == 2:  # Linear layer weights
            axis = 0  # Quantize per output channel for linear layers
        elif weights.dim() == 4:  # Conv layer weights
            axis = 0  # Quantize per output channel for conv layers
        else:
            axis = 0  # Default
    else:
        axis = None  # Quantize entire tensor
    
    # Get min and max values
    if axis is not None:
        # Perform per-channel quantization
        view_shape = [1] * weights.dim()
        view_shape[axis] = -1
        dims = list(range(weights.dim()))
        dims.remove(axis)
        
        w_min = weights.min(dim=dims[0], keepdim=True)[0]
        for d in dims[1:]:
            w_min = w_min.min(dim=d, keepdim=True)[0]
        
        w_max = weights.max(dim=dims[0], keepdim=True)[0]
        for d in dims[1:]:
            w_max = w_max.max(dim=d, keepdim=True)[0]
    else:
        # Perform per-tensor quantization
        w_min = weights.min()
        w_max = weights.max()
    
    # Calculate scales and zero points
    # For INT4, range is [-8, 7]
    scales = (w_max - w_min) / 15
    zeros = torch.round(-w_min / scales)
    
    # Clamp zeros to ensure they're in the valid range for INT4
    zeros = torch.clamp(zeros, 0, 15)
    
    # Quantize weights
    quantized_weights = torch.round(weights / scales + zeros)
    quantized_weights = torch.clamp(quantized_weights, 0, 15)
    
    # Pack quantized weights if possible
    # Note: Actual INT4 packing would require bit manipulation
    # This is a simulated version that returns a tensor with values 0-15
    
    # Convert to the required device
    quantized_weights = quantized_weights.to(org_device)
    scales = scales.to(org_device)
    zeros = zeros.to(org_device)
    
    return quantized_weights, scales, zeros

def dequantize_weights_from_int4(quantized_weights, scales, zeros):
    """
    Dequantize INT4 weights back to floating point.
    
    Args:
        quantized_weights: INT4 quantized weights (values 0-15)
        scales: Quantization scales
        zeros: Zero points
        
    Returns:
        weights: Dequantized weights
    """
    # Dequantize
    return scales * (quantized_weights - zeros)

def simulate_int4_matmul(a, b, scales, zeros):
    """
    Simulate matrix multiplication with one INT4 quantized operand.
    
    Args:
        a: First tensor (normal float)
        b: Second tensor (simulated INT4, values 0-15)
        scales: Quantization scales for b
        zeros: Zero points for b
        
    Returns:
        result: a @ dequantize(b)
    """
    # Dequantize b
    b_dequantized = dequantize_weights_from_int4(b, scales, zeros)
    
    # Perform matrix multiplication
    return torch.matmul(a, b_dequantized)

def prepare_transformer_for_int4(model):
    """
    Quantize weights of a transformer model to INT4.
    
    Args:
        model: A FusedTransformer model
        
    Returns:
        model: Model with quantized weights
        weight_data: Dict containing quantization parameters
    """
    weight_data = {}
    
    # Process each transformer layer
    for i, layer in enumerate(model.layers):
        # Quantize attention weights
        wq_q, wq_scales, wq_zeros = quantize_weights_to_int4(layer.wq.data)
        wk_q, wk_scales, wk_zeros = quantize_weights_to_int4(layer.wk.data)
        wv_q, wv_scales, wv_zeros = quantize_weights_to_int4(layer.wv.data)
        wo_q, wo_scales, wo_zeros = quantize_weights_to_int4(layer.wo.data)
        
        # Store quantized attention weights
        layer.wq.data = wq_q
        layer.wk.data = wk_q
        layer.wv.data = wv_q
        layer.wo.data = wo_q
        
        # Quantize FFN weights
        w1_q, w1_scales, w1_zeros = quantize_weights_to_int4(layer.w1.data)
        w2_q, w2_scales, w2_zeros = quantize_weights_to_int4(layer.w2.data)
        
        # Store quantized FFN weights
        layer.w1.data = w1_q
        layer.w2.data = w2_q
        
        # Store quantization parameters
        weight_data[f'layer_{i}'] = {
            'wq': {'scales': wq_scales, 'zeros': wq_zeros},
            'wk': {'scales': wk_scales, 'zeros': wk_zeros},
            'wv': {'scales': wv_scales, 'zeros': wv_zeros},
            'wo': {'scales': wo_scales, 'zeros': wo_zeros},
            'w1': {'scales': w1_scales, 'zeros': w1_zeros},
            'w2': {'scales': w2_scales, 'zeros': w2_zeros},
        }
    
    # Quantize embedding and output weights if desired
    # (not done here to maintain accuracy for inputs/outputs)
    
    return model, weight_data

def test_int4_quantization(weight_shape=(512, 2048), seed=42):
    """
    Test INT4 quantization and dequantization.
    
    Args:
        weight_shape: Shape of test weights
        seed: Random seed
        
    Returns:
        error: Quantization error statistics
    """
    torch.manual_seed(seed)
    
    # Create test weights
    weights = torch.randn(weight_shape)
    
    # Quantize weights
    quantized_weights, scales, zeros = quantize_weights_to_int4(weights)
    
    # Dequantize weights
    dequantized_weights = dequantize_weights_from_int4(quantized_weights, scales, zeros)
    
    # Calculate error
    error = weights - dequantized_weights
    
    return {
        'mean_error': error.abs().mean().item(),
        'max_error': error.abs().max().item(),
        'min_error': error.abs().min().item(),
        'std_error': error.std().item(),
        'orig_std': weights.std().item(),
        'relative_error': error.abs().mean().item() / weights.abs().mean().item()
    }

def quantize_tensor(tensor, num_bits=8, per_channel=False, axis=0):
    """Quantize a tensor to a specified number of bits."""
    qmin = 0.
    qmax = 2.**num_bits - 1.

    if per_channel:
        # Per-channel quantization
        mins = tensor.min(dim=axis, keepdim=True)[0]
        maxs = tensor.max(dim=axis, keepdim=True)[0]
    else:
        # Per-tensor quantization
        mins = tensor.min()
        maxs = tensor.max()

    scales = (maxs - mins) / (qmax - qmin)
    zero_points = qmin - mins / scales

    # Quantize
    q_tensor = torch.round(tensor / scales + zero_points)
    q_tensor = torch.clamp(q_tensor, qmin, qmax).to(torch.int8)

    return q_tensor, scales, zero_points

def dequantize_tensor(q_tensor, scales, zero_points):
    """Dequantize a tensor from its quantized form."""
    return scales * (q_tensor - zero_points)

if __name__ == "__main__":
    print("Testing INT4 quantization...")
    
    # Test on different weight shapes
    shapes = [
        (768, 768),    # Attention projection
        (768, 3072),   # FFN first layer
        (3072, 768),   # FFN second layer
        (50000, 768)   # Embedding
    ]
    
    for shape in shapes:
        results = test_int4_quantization(shape)
        print(f"\nWeight shape {shape}:")
        print(f"Mean abs error: {results['mean_error']:.6f}")
        print(f"Max abs error: {results['max_error']:.6f}")
        print(f"Relative error: {results['relative_error']:.2%}")
        print(f"Original std: {results['orig_std']:.6f}, Error std: {results['std_error']:.6f}")
    
    # Test simulated matmul
    print("\nTesting simulated INT4 matmul...")
    a = torch.randn(8, 512)
    b = torch.randn(512, 2048)
    
    # Reference matmul
    ref_result = torch.matmul(a, b)
    
    # Quantize b
    b_quantized, scales, zeros = quantize_weights_to_int4(b)
    
    # Simulated INT4 matmul
    sim_result = simulate_int4_matmul(a, b_quantized, scales, zeros)
    
    # Calculate error
    matmul_error = ref_result - sim_result
    
    print(f"Matmul mean abs error: {matmul_error.abs().mean().item():.6f}")
    print(f"Matmul max abs error: {matmul_error.abs().max().item():.6f}")
    print(f"Matmul relative error: {matmul_error.abs().mean().item() / ref_result.abs().mean().item():.2%}") 