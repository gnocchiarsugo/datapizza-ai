import uuid

import pytest
from datapizza.core.vectorstore import VectorConfig
from datapizza.type import EmbeddingFormat
from datapizza.type.type import Chunk, DenseEmbedding
from datapizza.core.vectorstore import Distance

# from datapizza.vectorstores.chromadb import ChromaDBVectorstore
from datapizza.vectorstores.chromadb import ChromaDBVectorstore



@pytest.fixture
def persistent_vectorstore() -> ChromaDBVectorstore:
    vectorstore = ChromaDBVectorstore(type='persistent')
    vectorstore.create_collection(
        collection_name="test",
        vector_config=[VectorConfig(dimensions=1536, name="dense_emb_name")],
    )

    return vectorstore


def test_vectorstore_init():
    vectorstore = ChromaDBVectorstore(type='persistent')
    assert vectorstore is not None


def test_vectorstore_create_collection(persistent_vectorstore):
    persistent_vectorstore.create_collection(
        collection_name="test",
        vector_config=[VectorConfig(dimensions=1536, name="test")],
    )
    colls = persistent_vectorstore.get_collections()
    assert len(colls) == 1
    colls[0].name == 'test'
    persistent_vectorstore.delete_collection('test')

def test_delete_collection(persistent_vectorstore):
    # Test persistence
    colls = persistent_vectorstore.get_collections()
    assert len(colls) == 1
    colls[0].name == 'test'
    persistent_vectorstore.delete_collection(collection_name="test")
    colls = persistent_vectorstore.get_collections()
    assert len(colls) == 0
    
def test_vectorstore_add(persistent_vectorstore):
    vector:list[float] = [0.0] * 1536
    id:str = str(uuid.uuid4())
    text:str = "Hello world"
    chunks = [
        Chunk(
            id=id,
            text=text,
            embeddings=DenseEmbedding(name="dense_emb_name", vector=vector),
        )
    ]
    
    try:
        persistent_vectorstore.add(chunks, collection_name="test")
        # Should throw since the collection doesn't exist yet
        assert 0 == 1
    except:
        pass
        
    persistent_vectorstore.create_collection('test', vector_config=[VectorConfig(dimensions=1536, name='test')])
    persistent_vectorstore.add(chunks, collection_name='test')
    res:list[Chunk] = persistent_vectorstore.search(collection_name="test", query_vector=vector)
    assert len(res) == 1
    
    res[0].embeddings == vector
    res[0].id == id
    res[0].text == text
    res[0].metadata == {}
    


def test_create_collection_with_sparse_vector(persistent_vectorstore):
    try:
        persistent_vectorstore.create_collection(
            collection_name="test3",
            vector_config=[
                VectorConfig(dimensions=1536, name="test3", format=EmbeddingFormat.SPARSE)
            ],
        )
        # Should throw since Chroma dows not support Sparse Embedding
        assert 1 == 0
    except:
        pass



def test_collection_with_multiple_vectors(persistent_vectorstore):
    try:
        persistent_vectorstore.create_collection(
            collection_name="multi_vector_test",
            vector_config=[
                VectorConfig(dimensions=1536, name="dense_emb_name"),
                VectorConfig(name="sparse", format=EmbeddingFormat.SPARSE),
            ],
        )
        # Should throw since Chroma supports one embedding per chunk
        assert 1 == 0
    except:
        pass


def test_search_with_dense_vector(persistent_vectorstore):
    persistent_vectorstore.create_collection(
        collection_name="dense_test",
        vector_config=[VectorConfig(dimensions=1536, name="dense_emb_name")],
    )

    persistent_vectorstore.add(
        chunk=[
            Chunk(
                id=str(uuid.uuid4()),
                text="Hello world",
                embeddings=DenseEmbedding(name="dense_emb_name", vector=[0.0] * 1536),
            )
        ],
        collection_name="dense_test",
    )

    results = persistent_vectorstore.search(
        collection_name="dense_test",
        query_vector=[0.0] * 1536,
        vector_name="dense_emb_name",
    )
    assert len(results) == 1

    res_no_name = persistent_vectorstore.search(
        collection_name="dense_test",
        query_vector=[0.0] * 1536,
    )
    assert len(res_no_name) == 1
    
    persistent_vectorstore.delete_collection('dense_test')


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

