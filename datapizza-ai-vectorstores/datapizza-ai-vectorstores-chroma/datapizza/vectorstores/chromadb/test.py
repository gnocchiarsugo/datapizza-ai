
from chromadb_vectorstore import ChromaDBVectorstore
from datapizza.core.vectorstore import VectorConfig
from datapizza.type.type import Chunk, DenseEmbedding
from datapizza.core.vectorstore import Distance
import uuid


def persistent_vectorstore() -> ChromaDBVectorstore:
    vectorstore = ChromaDBVectorstore(type='persistent')
    vectorstore.create_collection(
        collection_name="test",
        vector_config=[VectorConfig(dimensions=1536, name="dense_emb_name")],
    )

    return vectorstore


def test_search_distance(persistent_vectorstore:ChromaDBVectorstore):
    chunk_list:list[Chunk] = [
        Chunk(
            id=str(uuid.uuid4()),
            text="Hello world",
            embeddings=DenseEmbedding(name="dense_emb_name", vector=[1.,-1.]),
        ),
        Chunk(
            id=str(uuid.uuid4()),
            text="Hello world",
            embeddings=DenseEmbedding(name="dense_emb_name", vector=[1.,1.]),
        )
    ]
    
    
    persistent_vectorstore.create_collection(
        collection_name="cosine",
        vector_config=[VectorConfig(name='', dimensions=2, distance=Distance.COSINE)],
    )

    persistent_vectorstore.add(
        chunk=chunk_list,
        collection_name="cosine"
    )
    
    persistent_vectorstore.create_collection(
        collection_name="euclidean",
        vector_config=[VectorConfig(name='', dimensions=2, distance=Distance.EUCLIDEAN)],
    )

    persistent_vectorstore.add(
        chunk=chunk_list,
        collection_name="euclidean"
    )
    
    
    cosine_dist:list[float] = persistent_vectorstore.get_distances('cosine', [1.,1.], k=2)
    euclidean_dist:list[float] = persistent_vectorstore.get_distances('euclidean', [1.,1.], k=2)
    
    
    # Since the cos distance take the scalar product, this is zero
    assert cosine_dist[0] <= 1e-6       and cosine_dist[0] >= -1e-6
    assert cosine_dist[1] <= 1 + 1e-6   and cosine_dist[1] >= 1-1e-6
    # while this is (1+1)^2 + (1-1)^2 = 4
    assert euclidean_dist[0] <= 1e-6        and euclidean_dist[0] >= -1e-6
    assert euclidean_dist[1] <= 4 + 1e-6    and euclidean_dist[1] >= 4 - 1e-6
    
    
    persistent_vectorstore.delete_collection('cosine')
    persistent_vectorstore.delete_collection('euclidean')

    

test_search_distance(persistent_vectorstore())