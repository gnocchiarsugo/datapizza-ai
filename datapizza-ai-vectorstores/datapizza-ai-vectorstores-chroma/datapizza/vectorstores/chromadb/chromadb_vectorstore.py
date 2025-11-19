import logging
from typing import Any
from pathlib import Path

from datapizza.core.vectorstore import VectorConfig, Vectorstore
from datapizza.type import (
    Chunk,
    DenseEmbedding,
    Embedding,
    EmbeddingFormat,
    SparseEmbedding,
)
# from qdrant_client import AsyncQdrantClient, QdrantClient, models
from chromadb import PersistentClient
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.client import Client
from chromadb.config import Settings
import chromadb.errors

log = logging.getLogger(__name__)

"""
    Code Philosophy:
        ChromaDB exports a variaety of Clients: PersistentClient, CloudClient, HttpClient, AsyncHttpClient, EphemeralClient, etc..
        but their implementation is always the same, they differ in Settings.
        The Client themselves are just functions that return a Client, but thier argument forces the user to specify certain settings:
        e.g. host / port for HttpClient or path for PersistentClient.
        We adopt this framework.
"""


# def ChromaDBPersistentVectorstore(
#     path: str | Path = "./chroma",
#     settings: Settings | None = None,
#     tenant: str | None = None,
#     database: str | None = None,
# ) -> ClientAPI:
#     return PersistentClient(
#         path=path
#         tenant=tenant, 
#         database=database, 
#         settings=settings
#     )
    


class ChromaDBPersistentVectorstore(Vectorstore):

    def __init__(
        self,
        path:str | Path = "./chroma",
        settings: Settings | None = None,
        **kwargs: Any,
    ):
        #Initialize the ChromaDBPersistentVectorstore.
        # tenant and database are supposed to be inside **kwargs
        self.client: ClientAPI
        self.client_kwargs : dict[str, Any]
        self.client_kwargs['path'] = path
        self.client_kwargs['settings'] = settings
        self.client_kwargs.update(kwargs or {})


        # self.client: ClientAPI = PersistentClient(
        #     path=path,
        #     settings=settings,
        #     **kwargs
        # )
        
    def get_client(self) -> ClientAPI:
        if not hasattr(self, "client"):
            self._init_client()
        return self.client

    def _init_client(self):
        self.client = PersistentClient(**self.client_kwargs)

        
    def add(self, 
            chunk: Chunk | list[Chunk], 
            collection_name: str | None = None
        ) :
        
        client = self.get_client()
        chunks = chunk if isinstance(chunk, list) else [chunk]
        
        # Propagate chromadb.errors.NotFoundError
        collection: Collection = client.get_collection(collection_name)
        
        for c in chunks:
            if isinstance(c, DenseEmbedding):
                collection.add(
                    ids=c.id,
                    embeddings=c.vector
                )   
            else:
                raise ValueError(f"Unsupported embedding type: {type(e)}")
                   
        



        

