# Qwen2.5-VL-3B OpenVINO Local Server

Runs Qwen2.5-VL-3B-Instruct locally via OpenVINO and exposes an
OpenAI-compatible `/v1/chat/completions` endpoint for UI-TARS.

---

## 1. Export the model (do this once)

```bash
pip install "optimum[openvino]" nncf

# INT4 + AWQ — good balance of size and accuracy
optimum-cli export openvino \
  -m Qwen/Qwen2.5-VL-3B-Instruct \
  --weight-format int4 \
  --group-size -1 \
  --sym \
  --awq \
  --scale-estimation \
  --dataset contextual \
  ./Qwen2.5-VL-3B-Instruct-ov

# Or download the pre-converted version (faster):
# pip install huggingface-hub[cli]
# huggingface-cli download llmware/Qwen2.5-VL-3B-Instruct-ov-int4-npu \
#   --local-dir ./Qwen2.5-VL-3B-Instruct-ov
```

---

## 2. Install server dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Start the server

```bash
# On CPU (works everywhere)
python server.py --model-dir ./Qwen2.5-VL-3B-Instruct-ov --device CPU --port 8000

# On Intel integrated/discrete GPU
python server.py --model-dir ./Qwen2.5-VL-3B-Instruct-ov --device GPU --port 8000

# On Intel Core Ultra NPU (requires latest NPU driver)
python server.py --model-dir ./Qwen2.5-VL-3B-Instruct-ov --device NPU --port 8000
```

First run will compile and cache the model (~1-2 min). Subsequent runs load from cache instantly.

---

## 4. Configure UI-TARS .env

```dotenv
VLM_PROVIDER=huggingface
VLM_BASE_URL=http://localhost:8000/v1
VLM_API_KEY=local
VLM_MODEL_NAME=qwen2.5-vl-3b
```

> `VLM_API_KEY` can be any non-empty string — the server doesn't validate it.

---

## 5. Test the endpoint manually

```bash
curl http://localhost:8000/v1/models

# Chat with image (base64)
python - <<'EOF'
import base64, json, requests

with open("screenshot.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "qwen2.5-vl-3b",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "What UI elements do you see?"}
        ]
    }],
    "max_tokens": 256,
    "temperature": 0.0
})
print(resp.json()["choices"][0]["message"]["content"])
EOF
```

---

## Device performance notes

| Device | Speed    | Notes                                      |
|--------|----------|--------------------------------------------|
| CPU    | ~5 tok/s | Works on any machine, no drivers needed    |
| GPU    | ~15 tok/s| Intel Arc / Iris Xe, requires OpenCL       |
| NPU    | ~20 tok/s| Intel Core Ultra only, needs NPU driver    |

## Troubleshooting

- **`openvino_genai` not found**: Run `pip install openvino-genai>=2025.3.0`
- **Slow first run**: Normal — OV compiles kernels and caches them in `ov_cache/`
- **OOM on NPU**: Use `--device CPU` or resize images smaller in `model.py`
- **UI-TARS can't connect**: Make sure `VLM_BASE_URL` ends with `/v1` (no trailing slash after)