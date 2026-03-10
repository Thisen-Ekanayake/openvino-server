# OpenVINO Model Testing Repository

This repository contains examples and tools for running OpenVINO-optimized large language models (LLMs) and vision-language models (VLMs) locally. It includes basic inference scripts, chat interfaces, model compression guides, and an OpenAI-compatible API server.

## Features

- **Basic Inference**: Simple text generation with Mistral-7B
- **Interactive Chat**: Command-line chat interface for text models
- **Vision-Language Server**: OpenAI-compatible API server for Qwen2.5-VL-3B-Instruct
- **Model Compression**: Instructions for optimizing models with INT4 quantization
- **Multi-Device Support**: CPU, GPU, and NPU execution

## Included Models

### Mistral-7B (Text Generation)
- Location: `mistral-7b-ov/`
- Optimized for text generation tasks
- Used in `basic_inference.py` and `chat.py`

### Qwen2.5-VL-3B-Instruct (Vision-Language)
- Location: `Qwen2.5-VL-3B-Instruct-ov/`
- Supports both text and image inputs
- Powers the OpenAI-compatible server in `openvino_server/`

## Prerequisites

- Python 3.8+
- OpenVINO Runtime
- For GPU/NPU: Appropriate drivers installed

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Thisen-Ekanayake/openvino-server.git
   cd openvino_test
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install openvino-genai
   ```

   For the server component:
   ```bash
   cd openvino_server
   pip install -r requirements.txt
   ```

## Usage

### Basic Text Inference

Run simple text generation with Mistral-7B:

```bash
python basic_inference.py
```

### Interactive Chat

Start a chat session with Mistral-7B:

```bash
python chat.py
```

Type your messages and get responses. Type "exit" or "quit" to end.

### Vision-Language Server

For detailed setup and usage of the OpenAI-compatible server, see [openvino_server/README.md](openvino_server/README.md).

Quick start:
```bash
cd openvino_server
python server.py --model-dir ../Qwen2.5-VL-3B-Instruct-ov --device CPU --port 8000
```

The server provides:
- `/v1/chat/completions` endpoint
- `/v1/models` endpoint
- OpenAI-compatible request/response format
- Support for text and image inputs

## Model Compression

See `how_to_compress.txt` for commands to compress models using Optimum and NNCF.

Example for Mistral-7B:
```bash
pip install "optimum[openvino]" nncf
optimum-cli export openvino \
  --model mistralai/Mistral-7B-v0.1 \
  --task text-generation-with-past \
  --weight-format int4 \
  ./mistral-7b-ov
```

## Performance

Approximate token generation speeds (may vary by hardware):

| Device | Speed | Notes |
|--------|-------|-------|
| CPU | ~5-10 tok/s | Works on any machine |
| GPU | ~15-25 tok/s | Intel Arc/Iris Xe required |
| NPU | ~20-30 tok/s | Intel Core Ultra only |

First run may be slower due to model compilation and caching.

## Project Structure

```
.
├── basic_inference.py          # Simple text generation demo
├── chat.py                     # Interactive chat interface
├── how_to_compress.txt         # Model compression instructions
├── mistral-7b-ov/             # Mistral-7B OpenVINO model files
├── Qwen2.5-VL-3B-Instruct-ov/ # Qwen2.5-VL OpenVINO model files
└── openvino_server/           # OpenAI-compatible API server
    ├── model.py               # Model wrapper class
    ├── server.py              # FastAPI server
    ├── requirements.txt       # Server dependencies
    └── README.md              # Detailed server documentation
```

## Troubleshooting

- **Import errors**: Ensure `openvino-genai` is installed (`pip install openvino-genai>=2025.3.0`)
- **Slow first run**: Normal - OpenVINO compiles and caches optimized kernels
- **Device not found**: Check that required drivers are installed for GPU/NPU
- **Memory issues**: Try CPU mode or reduce `max_new_tokens` parameter
- **Model loading fails**: Verify model directory path and ensure all model files are present

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this collection of OpenVINO examples.