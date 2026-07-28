"""Turn the sample documents into vectors and store them in Db2.

    PYTHONPATH=src .venv/bin/python -m haystack_db2_rag.index

The pipeline is two components:  embedder -> writer
"""

from haystack import Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret

from . import settings
from .documents import SAMPLE_DOCUMENTS
from .store import document_store

# recreate_table=True gives a clean table every run, so this script is repeatable.
store = document_store(recreate_table=True)

embedder = OpenAIDocumentEmbedder(
    api_key=Secret.from_token(settings.API_KEY),
    model=settings.EMBED_MODEL,
    api_base_url=settings.EMBED_BASE_URL,
)

pipeline = Pipeline()
pipeline.add_component("embedder", embedder)
pipeline.add_component("writer", DocumentWriter(document_store=store))
pipeline.connect("embedder", "writer")

result = pipeline.run({"embedder": {"documents": SAMPLE_DOCUMENTS}})

print(f"Stored {result['writer']['documents_written']} documents in {settings.DB2_TABLE}.")
