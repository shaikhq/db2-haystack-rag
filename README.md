# haystack-db2-rag

Retrieval-augmented generation with [Haystack](https://haystack.deepset.ai/), **IBM Db2** as the
vector store, and a **local llama.cpp server** for embeddings and generation. No cloud services,
no API keys, no network egress at query time.

Recreated from the IBM Community tutorial
[Agentic Workflows with Haystack and IBM Db2](https://community.ibm.com/community/user/blogs/dhruv-chaturvedi/2026/07/10/agentic-workflows-with-haystack-and-ibm-db2),
which used cloud Db2 and watsonx.ai.

## What changed from the tutorial

| Tutorial | Here |
| --- | --- |
| Cloud Db2 (`BLUDB`, port 50001, `SECURITY=SSL`) | Local Db2 instance, `SAMPLE`, port 50000, no SSL |
| `Db2DocumentStore`, `Db2EmbeddingRetriever` | `IBMDb2DocumentStore`, `IBMDb2EmbeddingRetriever` — the class names in `ibm-db-haystack` 0.2.0; the blog's names are from an earlier release |
| `WatsonxDocumentEmbedder` / `WatsonxTextEmbedder` (`ibm/slate-125m-english-rtrvr`) | `OpenAIDocumentEmbedder` / `OpenAITextEmbedder` → llama.cpp, bge-small-en-v1.5, **dim 384** |
| `WatsonxChatGenerator` (`ibm/granite-3-2b-instruct`) | `OpenAIChatGenerator` → llama.cpp, Qwen2.5-3B-Instruct |
| `PromptBuilder` | `ChatPromptBuilder` (Haystack 3.x chat interface) |
| watsonx API key + project ID | Dummy key + `api_base_url` — llama.cpp ignores the key |

llama.cpp's server speaks the OpenAI API, so Haystack's stock OpenAI components work against it
unchanged. One `llama-server` process serves one model, so embeddings and generation run as two
processes on two ports.

## Requirements

- Db2 **12.1.2 or later** (native `VECTOR` type; verified on 12.1.5.0)
- Python 3.10+ (verified on 3.12)
- A C++ toolchain and CMake for building llama.cpp
- ~2.1 GB of disk for the two models

## Setup

### 1. llama.cpp and the models

Build `llama-server` from a pinned tag, CPU-only:

```bash
sudo dnf install -y cmake            # or: pip install --user cmake
git clone --depth 1 --branch b9913 https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build ~/llama.cpp/build --target llama-server -j"$(nproc)"
```

> `-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF` are required here. Without them the build
> downloads a prebuilt WebUI bundle from Hugging Face that does not match tag b9913, and
> `llama-ui-embed` aborts the build with `missing required asset(s): loading.html`. We only need
> the `/v1` API, not the browser UI. If you hit that error after a partial build, delete
> `~/llama.cpp/build/tools/ui` before rebuilding — the stale asset directory is re-validated.

Embedding model (bge-small-en-v1.5, ~37 MB):

```bash
mkdir -p ~/models/bge-small-en-v1.5
curl -fSL -o ~/models/bge-small-en-v1.5/bge-small-en-v1.5-q8_0.gguf \
  "https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf"
```

Generation model (Qwen2.5-3B-Instruct, ~2 GB):

```bash
mkdir -p ~/models/qwen2.5-3b-instruct
curl -fSL -o ~/models/qwen2.5-3b-instruct/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
```

Sanity-test each on a throwaway port. `--pooling cls` is required for bge — the wrong pooling
silently degrades quality:

```bash
~/llama.cpp/build/bin/llama-server -m ~/models/bge-small-en-v1.5/bge-small-en-v1.5-q8_0.gguf \
  --embedding --pooling cls --ctx-size 512 --host 127.0.0.1 --port 8099 >/tmp/sanity.log 2>&1 &
until curl -sf -o /dev/null http://127.0.0.1:8099/health; do sleep 1; done
curl -s http://127.0.0.1:8099/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input":"hello"}' \
  | python3 -c "import sys,json;print('dim', len(json.load(sys.stdin)['data'][0]['embedding']))"
fuser -k 8099/tcp
```

Expect `dim 384`.

```bash
~/llama.cpp/build/bin/llama-server -m ~/models/qwen2.5-3b-instruct/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --ctx-size 2048 --host 127.0.0.1 --port 8099 >/tmp/sanity.log 2>&1 &
until curl -sf -o /dev/null http://127.0.0.1:8099/health; do sleep 1; done
curl -s http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with one word: hello"}]}' \
  | python3 -c "import sys,json;print('reply:', json.load(sys.stdin)['choices'][0]['message']['content'])"
fuser -k 8099/tcp
```

A short reply means it works; failures land in `/tmp/sanity.log`.

> Use `curl -sf`, not `curl -s`, in the readiness loop. `/health` answers **503** while the model
> is still loading, and without `-f` curl treats that as success — the loop exits early and the
> first request fails with a confusing error.

### 2. The long-running servers

```bash
scripts/llama-servers.sh start     # embeddings :8081, chat :8080
scripts/llama-servers.sh status
scripts/llama-servers.sh stop
```

Logs go to `logs/`.

### 3. Db2

```bash
db2start
db2 connect to SAMPLE
```

The document store creates its own table (`HAYSTACK_DOCUMENTS` by default) on first use. Db2
requires a real password for TCP connections (`AUTHENTICATION=SERVER`), so `DB2_PASSWORD` in
`.env` must be the OS password of the instance user.

### 4. Python

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env      # then fill in DB2_PASSWORD
```

## Usage

Two scripts. Store the documents, then ask questions.

```bash
export PYTHONPATH=src

.venv/bin/python -m haystack_db2_rag.index
.venv/bin/python -m haystack_db2_rag.ask "Which savings account has no minimum balance?"
```

Pass a product type as a second argument to filter on metadata before the vector search:

```bash
.venv/bin/python -m haystack_db2_rag.ask "What are the risks?" investment
```

Example output:

```
Q: Which savings account has no minimum balance, and what rate does it pay?

A: The Everyday Savings Account has no minimum balance and pays 2.1% APY.

Retrieved:
  [0.209] savings: The Everyday Savings Account pays 2.1% APY with no minimum balance and...
  [0.225] savings: The Premier High-Yield Savings Account pays 4.35% APY but requires a $...
  [0.333] checking: The Student Checking Account has no minimum balance, no overdraft fee ...
```

Lower scores are closer — they are cosine *distances*, not similarities.

Verify the vectors landed, straight from SQL:

```bash
db2 connect to SAMPLE
db2 "SELECT COUNT(*) FROM HAYSTACK_DOCUMENTS"
db2 "SELECT COLNAME, TYPENAME, LENGTH FROM SYSCAT.COLUMNS WHERE TABNAME='HAYSTACK_DOCUMENTS'"
```

The `EMBEDDING` column comes back as `VECTOR` with length 384 — Db2 is storing the vectors
natively, not as a blob.

## Layout

```
src/haystack_db2_rag/
  settings.py     everything read from .env
  documents.py    sample banking-product corpus with filterable metadata
  store.py        connects to Db2 and creates the table
  index.py        embedder -> writer
  ask.py          text_embedder -> retriever -> prompt_builder -> generator
scripts/
  llama-servers.sh
```

The code is deliberately minimal — no error handling, no retries, no edge cases — so each file
reads top to bottom. `index.py` recreates the table on every run, which keeps it repeatable.
