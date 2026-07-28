# haystack-db2-rag

Retrieval-augmented generation with [Haystack](https://haystack.deepset.ai/), **IBM Db2** as the
vector store, and **IBM watsonx.ai** for embeddings and chat generation.

Started from the IBM Community tutorial
[Agentic Workflows with Haystack and IBM Db2](https://community.ibm.com/community/user/blogs/dhruv-chaturvedi/2026/07/10/agentic-workflows-with-haystack-and-ibm-db2)
(banking-assistant knowledge retrieval), then extended.

## Stack

| Piece | What it does |
| --- | --- |
| `Db2DocumentStore` | Creates/manages the vector table in Db2 12.1.2+ (native `VECTOR` type) |
| `Db2EmbeddingRetriever` | Similarity search — COSINE / EUCLIDEAN / MANHATTAN, with metadata filters |
| `WatsonxDocumentEmbedder` / `WatsonxTextEmbedder` | Vectorize documents and queries (`ibm/slate-125m-english-rtrvr`) |
| `WatsonxChatGenerator` | Answer generation (`ibm/granite-3-2b-instruct`) |
| `PromptBuilder` | Assembles retrieved context into the prompt |

## Requirements

- Db2 **12.1.2 or later** (the `VECTOR` data type is not available in earlier releases)
- A watsonx.ai project (API key + project ID)
- Python 3.10+

On Linux, the `ibm-db` driver needs build tooling:

```bash
sudo apt-get install -y build-essential libxml2-dev
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your credentials
```

## Layout

```
src/haystack_db2_rag/   pipeline code
notebooks/              exploratory notebooks
data/                   sample documents (gitignored except samples)
```

## Status

Scaffold only — tutorial code goes in next.
