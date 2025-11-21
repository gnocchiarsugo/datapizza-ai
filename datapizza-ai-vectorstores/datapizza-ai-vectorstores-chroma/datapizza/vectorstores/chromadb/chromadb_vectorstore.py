import logging
from typing import Any, Literal

from datapizza.core.vectorstore import VectorConfig, Vectorstore, Distance
from datapizza.type import (
    Chunk,
    DenseEmbedding,
    EmbeddingFormat
)

import chromadb
from chromadb.api.client import Client
from chromadb.api.models.Collection import Collection
from chromadb.api.types import QueryResult
from chromadb.config import Settings

Type = Literal['persistent', 'http']

log = logging.getLogger(__name__)

"""
    Guess what this code outputs:
    
    ```python
        settings = Settings(
            persist_directory="./chroma",
            is_persistent=True
        )
        admin = AdminClient(settings=settings)
        admin.create_tenant('meme')
        admin.create_database('pizza', tenant='meme')
        print(admin.list_databases())
    ```
    That's right: 
        [{'id': UUID('00000000-0000-0000-0000-000000000000'), 'name': 'default_database', 'tenant': 'default_tenant'}]

    What about this?
    ```python
        settings = Settings()
        admin = AdminClient(settings=settings)
        admin.create_tenant('meme')
        admin.create_database('pizza', tenant='meme')
        ec = EphemeralClient(settings=settings, tenant='meme', database='pizza')
        print(ec.get_user_identity().tenant)
        print(ec.get_user_identity().databases)
    ```
    That's right: 
        default_tenant
        ['default_database']
"""

class ChromaDBVectorstore(Vectorstore):
    
    def __init__(
        self,
        type: Type | None = None,
        batch_size:int = 100,
        **kwargs
    ):
        self.client: Client
        # self.a_client: AsyncClientAPI
        self.a_client: Client
        self.type = type
        self.batch_size = batch_size
        self.supported_types = ['persistent', 'http']
        self.kwargs: dict[str, Any] = kwargs
        
    # region Client Generation
    
    def get_client(self) -> Client:
        if not hasattr(self, "client"):
            self._init_client()
        return self.client

    def _get_a_client(self) -> Client:
        if not hasattr(self, "a_client"):
            self._init_a_client()
        return self.a_client

    def _init_client(self):
        settings: Settings = self.kwargs.get('settings') or Settings()
        if self.type == 'persistent':
            self.kwargs['path'] = self.kwargs.get('path') or './chroma'
            settings.persist_directory = str(self.kwargs['path'])
            settings.is_persistent = True
            # self.kwargs['settings'] = settings
        elif self.type == 'http':
            settings.chroma_api_impl = "chromadb.api.fastapi.FastAPI"
        
            if settings.chroma_server_host and settings.chroma_server_host != self.kwargs.get('host'):
                raise ValueError(
                    f"Chroma server host provided in settings[{settings.chroma_server_host}] is different to the one provided in HttpClient: [{self.kwargs.get('host')}]"
                )
            settings.chroma_server_host = self.kwargs.get('host')
            if settings.chroma_server_http_port and settings.chroma_server_http_port != self.kwargs.get('port'):
                raise ValueError(
                    f"Chroma server http port provided in settings[{settings.chroma_server_http_port}] is different to the one provided in HttpClient: [{self.kwargs.get('port')}]"
                )
            settings.chroma_server_http_port = self.kwargs.get('port')
            settings.chroma_server_ssl_enabled = self.kwargs.get('ssl')
            settings.chroma_server_headers = self.kwargs.get('headers')
            # self.kwargs['settings'] = settings
        else:
            raise ValueError(f"Client type {self.type} is not supported.")
        
        self.client = chromadb.Client(settings=settings)


    """
        AsyncHttpClient breaks API since _init_a_client would become async.
        Async for now is just at this layer, we don't use the native Async API of ChromaDB.
        We use native sync ChromaDB operations in async python functions
    """
    def _init_a_client(self):
        self._init_client()

    # endregion

    # region Override methods

    def add(self, chunk: Chunk | list[Chunk], collection_name: str | None = None, **kwargs):
        """
        Adds chunks to the collection

        Args:
            chunk (Chunk | list[Chunk]): Chunk or list of Chunks to add
            collection_name (str | None, optional): _description_. Defaults to None.

        Raises:
            ValueError: If collection_name is not set
            ValueError: If the collection doesn't exist
            ValueError: If there is a lenght mismatch between chunk and kwargs
        """
        
        # Validate collection_name
        if not collection_name:
            raise ValueError("Collection name must be set.")
        
        
        chunks = chunk if isinstance(chunk, list) else [chunk]
        client = self.get_client()
        collection = client.get_collection(collection_name)
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            
            # Creates a dictionary of lists to batch insert into the DB
            # The dict key must have the same same as the collection.upsert() expected parameters
            _to_insert: dict={'ids':[], 'documents':[], 'embeddings':[], 'metadatas':[]}
            for c in batch:
                if isinstance(c.embeddings, list):
                    raise ValueError("Only one embedding per chunk.")
                if isinstance(c.embeddings, DenseEmbedding):
                    _to_insert['ids'].append(c.id)
                    _to_insert['documents'].append(c.text)
                    _to_insert['embeddings'].append(c.embeddings.vector)
                    _to_insert['metadatas'].append(c.metadata if c.metadata != {} else None)
                else:
                    raise ValueError("Currently ChromaDB does not suport Embeddings other than DenseEmbedding.")
            _to_insert.update(kwargs)
            collection.upsert(**_to_insert)
   
        
    async def a_add(self, chunk: Chunk | list[Chunk], collection_name: str | None = None):
        self.add(chunk=chunk, collection_name=collection_name)
        
    
    def update(self, collection_name:str, chunk: Chunk | list[Chunk], **kwargs):
        client:Client = self.get_client()
        if collection_name not in [c.name for c in client.list_collections()]:
            raise ValueError(f"Collection {collection_name} does not exist for tenant {client.tenant} in database {client.database}.")
        
        # Since add() uses upsert we can use it.
        # Flashback to the fitting everything in the square hole meme
        self.add(
            chunk=chunk,
            collection_name=collection_name,
            **kwargs
        )
        
        
    def remove(self, collection_name:str, ids:list[str], **kwargs):
        client:Client = self.get_client()
        collection:Collection = client.get_collection(collection_name)
        collection.delete(ids=ids, **kwargs)

            
    def search(self, collection_name: str, query_vector:list[float], k:int = 10, vector_name:str | None = None, **kwargs) -> list[Chunk]:
        client:Client = self.get_client()
        collection = client.get_collection(collection_name)
        result:QueryResult = collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=['documents', 'metadatas','embeddings', 'distances'],
            **kwargs
        )
        chunks = []
        for i in range(len(result['ids'][0])):
            # the returned embedding is a numpy ndarray
            chunks.append(
                Chunk(
                    id=result.get('ids')[0][i] or None,
                    text=result.get('documents')[0][i] or None,
                    embeddings=result.get('embeddings')[0][i].tolist()[0] or None,
                    metadata=result.get('metadata')[0][i] if result.get('metadata') else None,
                )
            )
        return chunks

    
    async def a_search(self, collection_name, query_vector, k = 10, vector_name = None, **kwargs):
        return self.search(
            collection_name=collection_name, 
            query_vector=query_vector,
            vector_name=vector_name, 
            k=k,
            **kwargs
        )


    def retrieve(self, collection_name:str, ids:list[str], **kwargs):
        """
        Retrieve chunks from a collection by their IDs.
        """
        client:Client = self.get_client()
        collection:Collection = client.get_collection(collection_name)
        result = collection.get(
            ids=ids,
            include=['metadatas','embeddings','documents'],
            **kwargs
        )
        chunks = []
        for i in range(len(ids)):
            chunks.append(
                Chunk(
                    id=result.get('ids')[i],
                    text=result.get('documents')[i],
                    embeddings=result.get('embeddings')[i],
                    metadata=result.get('metadata')[i],
                )
            )
        return chunks
    
    # endregion
    
    def create_collection(self, collection_name:str, vector_config: list[VectorConfig]):
        client:Client = self.get_client()
        
        if collection_name in [c.name for c in client.list_collections()]:
            log.warning(
                f"Collection {collection_name} already exists, skipping creation"
            )
            return
        if any(v.format == EmbeddingFormat.SPARSE for v in vector_config):
            raise ValueError("ChromaDB only supports DENSE embeddings.")
        
        if len(vector_config) != 1:
            raise ValueError("Only one configuration per collection.")
        
        vector_config:VectorConfig = vector_config[0]
        
        if vector_config.distance == Distance.COSINE:
            dist = 'cosine'
        elif vector_config.distance == Distance.EUCLIDEAN:
            dist = 'l2'
        else:
            raise ValueError(f"Distance {vector_config.distance} not supported.")
        
        # Instead of just create_collection use get_or_create, this assures that the collection is created
        client.get_or_create_collection(
            metadata={'hnsw:space':dist},
            name=collection_name
        )
        
    def get_collections(self) -> list[Collection]:
        client:Client = self.get_client()
        return client.list_collections()
    
    def delete_collection(self, collection_name:str):
        client:Client = self.get_client()
        client.delete_collection(collection_name)
        
    def get_distances(self, collection_name: str, query_vector:list[float], k:int = 10) -> list[float]:
        client:Client = self.get_client()
        collection = client.get_collection(collection_name)
        result:QueryResult = collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=['distances'],
        )
        return result['distances'][0]
    
    