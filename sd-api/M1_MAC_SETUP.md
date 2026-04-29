# M1 Mac Studio Setup Guide

## Overview

The `sd_api.py` file has been modified to support Apple Silicon (M1/M2/M3) Macs. The main changes include:

1. **Automatic device detection**: The code now automatically detects and uses MPS (Metal Performance Shaders) on Apple Silicon Macs
2. **Device-specific memory management**: Optimized memory handling for M1 Macs' unified memory architecture
3. **Fallback support**: Falls back to CPU if MPS is not available

## Key Changes Made

### 1. Device Detection Function
Added a `get_device()` function that:
- Prioritizes MPS on Apple Silicon Macs
- Falls back to CUDA on NVIDIA GPUs
- Uses CPU as a last resort

### 2. Pipeline Initialization
Modified pipeline initialization to:
- Move pipelines to MPS device when available
- Use appropriate memory management for MPS (unified memory)
- Maintain compatibility with CUDA and CPU

### 3. All Pipeline Types Updated
The following pipeline types now support MPS:
- Main Stable Diffusion pipeline
- Img2Img pipeline
- Inpaint pipeline
- ControlNet pipelines
- SDXL pipelines

## Installation Instructions

### 1. Install PyTorch with MPS Support

**Important**: Do NOT install the CUDA version of PyTorch. Install the standard version which includes MPS support:

```bash
pip install torch torchvision torchaudio
```

Verify MPS is available:
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

You should see: `MPS available: True`

### 2. Install Other Dependencies

```bash
pip install -r requirements-m1-mac.txt
```

Or install manually:
```bash
pip install fastapi uvicorn[standard] diffusers transformers safetensors accelerate rembg pillow colorama opencv-python-headless numpy scikit-image optimum-quanto requests huggingface_hub[cli] onnxruntime sentencepiece
```

### 3. Run the API

```bash
python sd_api.py
```

The API will automatically detect and use MPS on your M1 Mac Studio.

## Performance Considerations

### M1 Mac Studio Advantages
- **Unified Memory**: M1 Macs have unified memory architecture, which can be more efficient for some operations
- **Metal Performance Shaders**: Hardware-accelerated inference using Apple's Metal framework
- **Power Efficiency**: Lower power consumption compared to discrete GPUs

### Limitations
- **Speed**: MPS may be slower than high-end NVIDIA GPUs (like RTX 3090) for some operations
- **Compatibility**: Some operations may still fall back to CPU if MPS doesn't support them
- **Memory**: M1 Mac Studio typically has 32GB-128GB unified memory, which is shared between CPU and GPU

### Optimization Tips
1. **Model Size**: Consider using smaller models or quantized versions for faster inference
2. **Batch Size**: Start with smaller batch sizes and increase if memory allows
3. **Attention Slicing**: Already enabled in the code to reduce memory usage
4. **Image Resolution**: Lower resolutions will be faster

## Troubleshooting

### MPS Not Available
If you see "No GPU acceleration available, using CPU":
1. Ensure you're running macOS 12.3+ (Monterey or later)
2. Verify PyTorch version: `python -c "import torch; print(torch.__version__)"` (should be 2.0+)
3. Check MPS availability: `python -c "import torch; print(torch.backends.mps.is_available())"`

### Out of Memory Errors
1. Reduce image resolution
2. Use smaller models
3. Enable more aggressive memory optimizations (already enabled in code)
4. Close other applications to free up unified memory

### Slow Performance
1. Check that MPS is actually being used (check logs for "Using MPS" message)
2. Some operations may be faster on CPU - this is normal for certain PyTorch operations
3. Consider using quantized models (optimum-quanto is already included)

## Differences from Linux/CUDA Version

1. **No CUDA dependencies**: PyTorch is installed without CUDA support
2. **Device handling**: Uses `torch.device("mps")` instead of `torch.device("cuda")`
3. **Memory management**: Unified memory means different optimization strategies
4. **Performance**: Generally slower than high-end NVIDIA GPUs but still much faster than CPU

## Testing

After installation, test the API:

```bash
# Start the server
python sd_api.py

# In another terminal, test the endpoint
curl -X POST "http://localhost:8000/sd" \
  -F "prompt=a beautiful landscape" \
  -F "steps=20" \
  -F "width=512" \
  -F "height=512" \
  --output test_image.jpg
```

Check the logs to confirm MPS is being used:
```
Using MPS (Metal Performance Shaders) device for Apple Silicon
Pipeline moved to MPS device
```

## Notes

- The code maintains backward compatibility with Linux/CUDA systems
- If running on a non-Apple Silicon Mac, it will fall back to CPU
- The modifications are minimal and don't change the API interface
- All existing endpoints work the same way

