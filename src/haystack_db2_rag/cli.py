"""Command line entry point.

    python -m haystack_db2_rag.cli preflight
    python -m haystack_db2_rag.cli index [--recreate]
    python -m haystack_db2_rag.cli ask "question" [--product-type savings] [--region US] [--top-k 3]
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

from .config import chat_endpoint, db2_settings, embed_endpoint
from .documents import SAMPLE_DOCUMENTS
from .pipelines import ask, build_indexing_pipeline, build_query_pipeline, document_store


def _post_json(url: str, payload: dict, timeout: float = 60.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer sk-no-key-required"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def cmd_preflight(_: argparse.Namespace) -> int:
    failures = []

    db2 = db2_settings()
    try:
        store = document_store()
        count = store.count_documents()
        print(f"  Db2         OK    {db2.database} on {db2.hostname}:{db2.port}, "
              f"table {db2.table_name}, {count} documents")
    except Exception as exc:  # noqa: BLE001 — preflight reports, it does not handle
        failures.append(f"Db2 connection failed: {exc}")
        print(f"  Db2         FAIL  {exc}")

    embed = embed_endpoint()
    try:
        data = _post_json(f"{embed.base_url}/embeddings", {"input": "preflight", "model": embed.model})
        dim = len(data["data"][0]["embedding"])
        if dim != db2.embedding_dim:
            failures.append(f"Embedding dim {dim} != EMBED_DIM {db2.embedding_dim}")
            print(f"  Embeddings  FAIL  returned dim {dim}, EMBED_DIM says {db2.embedding_dim}")
        else:
            print(f"  Embeddings  OK    {embed.base_url} ({embed.model}), dim {dim}")
    except (urllib.error.URLError, KeyError, TimeoutError) as exc:
        failures.append(f"Embedding endpoint unreachable: {exc}")
        print(f"  Embeddings  FAIL  {embed.base_url}: {exc}")

    chat = chat_endpoint()
    try:
        data = _post_json(
            f"{chat.base_url}/chat/completions",
            {"model": chat.model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5},
        )
        data["choices"][0]["message"]["content"]
        print(f"  Chat        OK    {chat.base_url} ({chat.model})")
    except (urllib.error.URLError, KeyError, TimeoutError) as exc:
        failures.append(f"Chat endpoint unreachable: {exc}")
        print(f"  Chat        FAIL  {chat.base_url}: {exc}")

    if failures:
        print(f"\n{len(failures)} check(s) failed.", file=sys.stderr)
        return 1
    print("\nAll checks passed.")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    store = document_store(recreate_table=args.recreate)
    pipeline = build_indexing_pipeline(store)
    result = pipeline.run({"embedder": {"documents": SAMPLE_DOCUMENTS}})
    written = result["writer"]["documents_written"]
    print(f"Indexed {written} documents into {db2_settings().table_name} "
          f"({store.count_documents()} total).")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    filters = None
    conditions = []
    if args.product_type:
        conditions.append({"field": "meta.product_type", "operator": "==", "value": args.product_type})
    if args.region:
        conditions.append({"field": "meta.region", "operator": "==", "value": args.region})
    if args.max_min_balance is not None:
        conditions.append(
            {"field": "meta.min_balance", "operator": "<=", "value": args.max_min_balance}
        )
    if conditions:
        filters = conditions[0] if len(conditions) == 1 else {"operator": "AND", "conditions": conditions}

    store = document_store()
    pipeline = build_query_pipeline(store, top_k=args.top_k)
    result = ask(pipeline, args.question, filters=filters)

    print(f"\nQ: {args.question}")
    if filters:
        print(f"   filters: {json.dumps(filters)}")
    print(f"\nA: {result['answer']}\n")
    print("Retrieved:")
    for doc in result["documents"]:
        score = f"{doc.score:.4f}" if doc.score is not None else "n/a"
        print(f"  [{score}] {doc.meta.get('product_type')}/{doc.meta.get('region')}: "
              f"{doc.content[:80]}...")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="haystack_db2_rag")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preflight", help="check Db2 and both llama.cpp endpoints").set_defaults(
        func=cmd_preflight
    )

    index_parser = sub.add_parser("index", help="embed and store the sample documents")
    index_parser.add_argument("--recreate", action="store_true", help="drop and recreate the table")
    index_parser.set_defaults(func=cmd_index)

    ask_parser = sub.add_parser("ask", help="run the RAG query pipeline")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--product-type")
    ask_parser.add_argument("--region")
    ask_parser.add_argument("--max-min-balance", type=int)
    ask_parser.add_argument("--top-k", type=int, default=3)
    ask_parser.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
