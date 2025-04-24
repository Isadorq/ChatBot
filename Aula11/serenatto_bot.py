import os
from typing import List
from llama_index.core import SimpleDirectoryReader, StorageContext,
VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatSummaryMemoryBuffer
import chromadb
from tempfile import TemporaryDirectory
from PyPDF2 import PdfReader

class ChromaEmbeddingWrapper:
 def __init__(self, model_name: str):
 self.model = HuggingFaceEmbedding(model_name=model_name)
 def __call__(self, input: List[str]) -> List[List[float]]:
 return self.model.embed_documents(input)

class SerenattoBot:
 def __init__(self):
 self.embed_model =
 HuggingFaceEmbedding(model_name='intfloat/multilingual-e5-large')
 self.embed_model_chroma =
 ChromaEmbeddingWrapper(model_name='intfloat/multilingual-e5-large')
 chroma_client = chromadb.PersistentClient(path='./chroma_db')
 collection_name = 'documentos_serenatto'
 chroma_collection = chroma_client.get_or_create_collection(
 name=collection_name,
 embedding_function=self.embed_model_chroma
)
