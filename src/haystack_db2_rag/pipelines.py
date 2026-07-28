"""Indexing and query pipelines.

Same shape as the tutorial, with the watsonx components swapped for Haystack's
OpenAI-compatible ones pointed at local llama.cpp servers:

    indexing : OpenAIDocumentEmbedder -> DocumentWriter -> IBMDb2DocumentStore
    query    : OpenAITextEmbedder -> IBMDb2EmbeddingRetriever -> ChatPromptBuilder
               -> OpenAIChatGenerator
"""

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.ibm_db import IBMDb2EmbeddingRetriever
from haystack_integrations.document_stores.ibm_db import IBMDb2DocumentStore

from .config import chat_endpoint, db2_settings, embed_endpoint

ANSWER_TEMPLATE = """You are a banking product assistant. Answer the question using only the
product documents below. If they do not contain the answer, say so plainly — do not invent
products, rates, or fees.

{% for doc in documents %}
Document {{ loop.index }} ({{ doc.meta.product_type }}, {{ doc.meta.region }}):
{{ doc.content }}
{% endfor %}

Question: {{ question }}
Answer:"""


def document_store(recreate_table: bool = False) -> IBMDb2DocumentStore:
    settings = db2_settings()
    return IBMDb2DocumentStore(recreate_table=recreate_table, **settings.store_kwargs())


def _document_embedder() -> OpenAIDocumentEmbedder:
    endpoint = embed_endpoint()
    return OpenAIDocumentEmbedder(
        api_key=endpoint.api_key,
        model=endpoint.model,
        api_base_url=endpoint.base_url,
        # bge models are trained with a query prefix on the query side only;
        # documents are embedded bare.
        meta_fields_to_embed=["product_type", "region"],
    )


def _text_embedder() -> OpenAITextEmbedder:
    endpoint = embed_endpoint()
    return OpenAITextEmbedder(
        api_key=endpoint.api_key,
        model=endpoint.model,
        api_base_url=endpoint.base_url,
        prefix="Represent this sentence for searching relevant passages: ",
    )


def build_indexing_pipeline(store: IBMDb2DocumentStore) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("embedder", _document_embedder())
    pipeline.add_component(
        "writer", DocumentWriter(document_store=store, policy=DuplicatePolicy.OVERWRITE)
    )
    pipeline.connect("embedder.documents", "writer.documents")
    return pipeline


def build_query_pipeline(store: IBMDb2DocumentStore, top_k: int = 3) -> Pipeline:
    endpoint = chat_endpoint()
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", _text_embedder())
    pipeline.add_component(
        "retriever", IBMDb2EmbeddingRetriever(document_store=store, top_k=top_k)
    )
    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(
            template=[ChatMessage.from_user(ANSWER_TEMPLATE)],
            required_variables=["question", "documents"],
        ),
    )
    pipeline.add_component(
        "generator",
        OpenAIChatGenerator(
            api_key=endpoint.api_key,
            model=endpoint.model,
            api_base_url=endpoint.base_url,
            generation_kwargs={"temperature": 0.2, "max_tokens": 400},
        ),
    )
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "generator.messages")
    return pipeline


def ask(pipeline: Pipeline, question: str, filters: dict | None = None) -> dict:
    """Run the query pipeline and return the answer plus the retrieved documents."""
    retriever_input: dict = {}
    if filters:
        retriever_input["filters"] = filters

    result = pipeline.run(
        {
            "text_embedder": {"text": question},
            "retriever": retriever_input,
            "prompt_builder": {"question": question},
        },
        include_outputs_from={"retriever"},
    )
    return {
        "answer": result["generator"]["replies"][0].text,
        "documents": result["retriever"]["documents"],
    }
