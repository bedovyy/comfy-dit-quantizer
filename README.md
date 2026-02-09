## Simple quantization scripts for ComfyUI diffusion models

### **"Quick Start"**

1. Activate ComfyUI's venv.

2. Copy config from `configs/` and customize yours. \
   *Rules are matched top-to-bottom (first match = highest priority/fallback)*
   ```json
   {
     "format": "comfy_quant",
     "block_names": ["net.blocks."],
     "rules": [
       { "policy": "keep", "match": ["blocks.0."] },
       { "policy": "float8_e4m3fn", "match": ["v_proj", "adaln_modulation", ".mlp"] },
       { "policy": "nvfp4", "match": ["k_proj", "q_proj", "output_proj"] }
     ]
   }
   ```

3. Run quantize.py with:
   ```bash
   python quantize.py your_config.json model.safetensors output.safetensors
   ```

---

### NVFP4 Calibration (in a clumsy way)

NVFP4 requires input scale calibration.

1. **Setup ComfyUI**:
   - Apply patches from `patch/` directory to ComfyUI
   - Copy `patch/extra/calibration.py` → `ComfyUI/extra/`
   - Run ComfyUI with:  
     ```bash
     COMFY_CALIB=1 python3 main.py ...
     ```

2. **Generate calibration data**:
   - Use your NVFP4-quantized model for normal image generation
   - This creates `CALIB_DATA.json`

3. **Apply calibration**:
   ```bash
   python add_input_scale.py CALIB_DATA.json uncalibrated_model.safetensors output.safetensors

4. **Verify**: Test generation with the calibrated model.
