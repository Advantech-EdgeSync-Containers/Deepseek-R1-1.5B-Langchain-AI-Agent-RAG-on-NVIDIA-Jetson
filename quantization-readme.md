# Post-Training Model Quantization Using Llama.cpp

## Overview

This guide provides step-by-step instructions to build and run `llama.cpp`, an efficient inference engine & quantization tool for LLMs.
Steps to be performed:

- Install/build Llamacpp on your machine
- Conversion/quantization using an automated Python script
- Convert/quantize the model using the online GGML repository as an alternative
- Users can also follow manual steps to perform model conversion and quantization.

## System Requirements

- OS Preferred: Ubuntu 20.04+ / Debian / macOS
- Alternatives: Windows via WSL2 (with Ubuntu) 
- RAM: At least 16GB for models up to 3B/7B parameters
- GPU: Supports multiple backends like NVIDIA, AMD, etc. Refer to [Supported Backends.](https://github.com/ggml-org/llama.cpp?tab=readme-ov-file#supported-backends)
- Recommendation: It is not recommended to perform quantization/conversion steps on Jetson Orin NX. Use non-ARM architectures for reliable results.


## Installation
Here are the possible ways to install it on your machine:

- Install `llama.cpp` using [brew, nix, or winget.](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md)
- Run with Docker—see [Docker documentation.](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository—check out [the build guide.](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)


## Option 1: Conversion & Quantization via Automated Script

### Recommended Python Version

- Use Python 3.10.12 version

### Create a Virtual Environment & Install the Following Packages

```
###(Optional) Setup python virtual environment
python -m venv .venv
source .venv/bin/activate

##Install following python packages
pip install \
huggingface-hub==0.31.1 \
transformers==4.46.3 \
torch==2.4.1 \
sentencepiece==0.2.0 
```

### Use Automated Script to handle model download, GGUF conversion, and quantization. (Recommended)

 Edit these configuration variables in the Python script below and run the script.
```
repo_id = <Hugging face Repo Id>
llama_cpp_dir = <Llama CPP directory>
model_dir = <Model directory>
output_dir = <Output directory to store model gguf and quantized models>
model_name = <Model Name>
quant_type = <Quantization Type>
HF_TOKEN = <Hugging face Token>
```
Refer to the sample below:
```
repo_id = "meta-llama/Llama-3.2-1B-Instruct"
llama_cpp_dir = "./llama.cpp"
model_dir = "./models/Llama-3.2-1B-Instruct"
output_dir = "./output"
model_name = "Llama-3.2-1B-Instruct"
quant_type = "Q4_K_M"
HF_TOKEN = <Hugging face Token>
```
### Python Script
```
import os
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download


# ==== CONFIGURATION ====
repo_id = "<Hugging face Repo Id>"
llama_cpp_dir = "<Llama CPP dir>"
model_dir = "<Model directory>"
output_dir = "<Output directory to store model gguf and quantized models>"
model_name = "<Model Name>"
quant_type = "<Quantization Type>"
HF_TOKEN = "<Hugging face Token>"
# ==========================

def run(cmd):
    print(">>> Running:", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)
    print("Done\n")


def download_model():
    os.environ["HF_TOKEN"] = HF_TOKEN
    model_path = (f"{model_dir}/{model_name}")
    if not os.path.exists(model_path):
        snapshot_download(
            repo_id=repo_id,
            local_dir=f"models/{model_name}",
            use_auth_token=True
        )
    else:
        print("Model Path already exists")

def convert_to_gguf():
    convert_script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    output_path = Path(output_dir) / f"{model_name}.f16.gguf"
    cmd = [
        "python3", str(convert_script),
        str(model_dir),
        "--outfile", str(output_path),
        "--outtype", "f16"
    ]
    run(cmd)
    return output_path

def quantize_model(f16_path):
    quantize_bin = Path(llama_cpp_dir) / "build" / "bin" / "llama-quantize"
    quant_output = f16_path.with_name(f"{f16_path.stem}.{quant_type.lower()}.gguf")
    cmd = [
        str(quantize_bin),
        str(f16_path),
        str(quant_output),
        quant_type
    ]
    run(cmd)
    return quant_output

if name == "__main__":
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Downloading HuggingFace model...")
    download_model()

    print("Converting HuggingFace model to GGUF...")
    f16_path = convert_to_gguf()

    print("Quantizing to", quant_type, "...")
    quant_path = quantize_model(f16_path)

    print("Full GGUF path:", f16_path)
    print("Quantized path:", quant_path)

```

### The script performs:
- Model download using Hugging Face Hub
- Conversion to .gguf using convert_hf_to_gguf.py
- Quantization using Llama-quantize

### Available Quantization Types
| 4-bit | 5-bit | 6–8-bit | Integer Quant (IQ) | True Quant (TQ) | Float |
| --------- | --------- | ----------- | ---------------------- | ------------------- | --------- |
| `Q4_0`    | `Q5_0`    | `Q6_K`      | `IQ1_S`                | `TQ1_0`             | `F16`     |
| `Q4_1`    | `Q5_1`    | `Q8_0`      | `IQ1_M`                | `TQ2_0`             | `BF16`    |
| `Q4_K`    | `Q5_K`    |             | `IQ2_XXS`              |                     | `F32`     |
| `Q4_K_S`  | `Q5_K_S`  |             | `IQ2_XS`               |                     |           |
| `Q4_K_M`  | `Q5_K_M`  |             | `IQ2_S`                |                     |           |
| `Q3_K`    |           |             | `IQ2_M`                |                     |           |
| `Q3_K_S`  |           |             | `IQ3_XXS`              |                     |           |
| `Q3_K_M`  |           |             | `IQ3_XS`               |                     |           |
| `Q3_K_L`  |           |             | `IQ3_S`, `IQ3_M`       |                     |           |
|           |           |             | `IQ4_NL`, `IQ4_XS`     |                     |           |

## Option 2: Conversion & Quantization using online GGUF Repository

This online tool can be used to convert & quantize Hugging Face models with ease & pushes the final model into the user's own Hugging Face account. Though this tool at times faces lots of traffic and may not be available all the time.
[GGUF Repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo)

![ggm-repo](..%2Fdata%2Fimages%2Fggml-repo.png)

## Option 3: Manual Steps to Handle Model Download, GGUF Conversion, and Quantization.

### Download the Model from Hugging Face

### Authentication token
User Access Tokens are the preferred way to authenticate an application or notebook to Hugging Face services. Create a Hugging Face token in your [account settings](https://huggingface.co/settings/tokens) and export it:

![hugging-face-token](..%2Fdata%2Fimages%2Fhugging-face-token.png)


```
export HF_TOKEN=<your-token>
```

### Use `snapshot_download` to download the model from huggingface_hub

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    local_dir="models/Llama-3.2-1B-Instruct",
    use_auth_token=True
)
```

### Conversion using local machine
The downloaded llama.cpp contains the file convert_hf_to_gguf.py to convert the downloaded model to .gguf format.
Use the script in `llama.cpp`:

```
python3 llama.cpp/convert_hf_to_gguf.py models/Llama-3.2-1B-Instruct   --outfile models/Llama-3.2-1B-Instruct.gguf
```

See the below screenshot for reference (apply on your model):
![gguf-convert](..%2Fdata%2Fimages%2Fgguf-convert.png)


### Quantize the Model
If the build of llama.cpp is successful, there must be a build folder; inside the build folder, the bin directory contains the different tools. Use llama-quantize to quantize the model.
### Quantization Tool

```
./llama.cpp/build/bin/llama-quantize <input> <output> <quant-type>
```

Example:

```
./llama.cpp/build/bin/llama-quantize   models/Llama-3.2-1B-Instruct.gguf   models/Llama-3.2-1B-Instruct.gguf Q4_K_M
```
![Quantization](..%2Fdata%2Fimages%2Fquantization.png)

### Check the llama-quantize help document.

```
./llama.cpp/build/bin/llama-quantize --help
```
![Quantize Help](..%2Fdata%2Fimages%2Fquantize-help.png)

### Supported Quantization Types
Key Points:

- Bit-width Columns (4-bit, 5-bit, 6–8-bit): These represent traditional quantization levels where model weights are reduced in size to lower precision (e.g., Q4_0, Q5_K_M), enabling faster inference and smaller memory footprint—especially useful for edge deployments

- Integer Quant (IQ): A series of newer quantization schemes (e.g., IQ1_S, IQ3_XS) designed for highly efficient inference. These formats vary in size, quality, and hardware compatibility.

- True Quant (TQ): A more recent and experimental family (e.g., TQ1_0) aiming to balance model performance and quality, though less commonly supported

- Float Column: Includes high-precision formats (F16, BF16, F32) that offer the highest quality but require significantly more resources—typically suited for training or high-end inference. 

Refer above table `Available Quantization types`

## LLM Model Quantization Levels: Size vs. Quality Trade-offs

Q-formats and floating-point types along with their size, quality, and recommendation:

| Format     | Size         | Quality Impact              | Notes / Recommendation                    |
|------------|--------------|-----------------------------|-------------------------------------------|
| Q2_K   | Smallest     | Extreme quality loss        | Not recommended                         |
| Q3_K_S | Very small   | Very high quality loss      | Use only for extreme compression needs    |
| Q3_K_M | Very small   | Very high quality loss      | Preferred over Q4_0 for ultra-small models |
| Q3_K_L | Small        | Substantial quality loss    | Preferred over Q4_1                        |
| Q3_K   | Alias        | —                           | Alias for Q3_K_M                           |
| Q4_K_S | Small        | Significant quality loss    | Not recommended                                           |
| Q4_K_M | Medium       | Balanced quality            | Recommended                             |
| Q4_K   | Alias        | —                           | Alias for Q4_K_M                           |
| Q4_0   | Small        | Very high quality loss      | prefer Q3_K_M                  |
| Q4_1   | Small        | Substantial quality loss    | prefer Q3_K_L                  |
| Q5_K_S | Large        | Low quality loss            | Recommended                             |
| Q5_K_M | Large        | Very low quality loss       | Recommended                             |
| Q5_K   | Alias        | —                           | Alias for Q5_K_M                           |
| Q5_0   | Medium       | Balanced quality            | prefer Q4_K_M                  |
| Q5_1   | Medium       | Low quality loss            | prefer Q5_K_M                  |
| Q6_K   | Very large   | Extremely low quality loss  | Large but good quality                                           |
| Q8_0   | Very large   | Extremely low quality loss  | Good for smaller models                         |
| F16    | Extremely large | Virtually no quality loss | Not recommended due to larger size for edge                        |
| F32    | Absolutely huge | Lossless                 | Not recommended due to larger size for edge                        |                        |
