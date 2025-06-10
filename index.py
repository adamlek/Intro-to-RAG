from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client
import qdrant_client.models
from typing import Any, Tuple, List

from ingest import parse_and_get_nodes
from IPython import embed

# set a simple embedding model for the vector store
Settings.embed_model = HuggingFaceEmbedding(
    model_name="KB/bert-base-swedish-cased"
)

def create_qdrant_client(storage_path: str = "./qdrant_db") -> qdrant_client.QdrantClient:
    """
    Create and return a Qdrant client for local storage.

    Args:
        storage_path (str): Path to the Qdrant database directory.

    Returns:
        qdrant_client.QdrantClient: The Qdrant client instance.
    """
    return qdrant_client.QdrantClient(path=storage_path)

def create_db_collection(
        client: qdrant_client.QdrantClient, 
        name: str = 'MyCoolVectorStore'
        ) -> Tuple[QdrantVectorStore, str]:
    """
    Create a Qdrant vector store collection.

    Args:
        client (qdrant_client.QdrantClient): The Qdrant client.
        name (str): Name of the collection.

    Returns:
        Tuple[QdrantVectorStore, str]: The vector store and collection name.
    """
    return QdrantVectorStore(client=client, collection_name=name), name

def setup_storage(vector_store: QdrantVectorStore) -> StorageContext:
    """
    Set up the storage context for the vector store.

    Args:
        vector_store (QdrantVectorStore): The vector store.

    Returns:
        StorageContext: The storage context.
    """
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

def fill_vectordb(
        data: List[Any], 
        storage_context: StorageContext
        ) -> VectorStoreIndex:
    """
    Fill the vector database with nodes.

    Args:
        data (List[Any]): List of nodes to insert.
        storage_context (StorageContext): The storage context.

    Returns:
        VectorStoreIndex: The vector store index.
    """
    return VectorStoreIndex(
        nodes=data,
        storage_context=storage_context,
    )

def get_retriever(vector_store: VectorStoreIndex) -> Any:
    """
    Get a retriever from the vector store index.

    Args:
        vector_store (VectorStoreIndex): The vector store index.

    Returns:
        Any: The retriever object.
    """
    return vector_store.as_retriever(
            similarity_top_k=25,
            )

if __name__ == '__main__':
    """
    Main execution block:
    - Initializes Qdrant client and vector store.
    - Loads and inserts nodes into the vector database.
    - Prints the number of vectors before and after insertion.
    - Retrieves and prints results for a sample query.
    """
    
    client = create_qdrant_client()
    vector_store, collection_name = create_db_collection(client)
    storage_context = setup_storage(vector_store)

    data = parse_and_get_nodes('data/')
    initial_data = data[:50]

    # create initial vector database
    vector_index = fill_vectordb(initial_data, storage_context)
    
    num_vectors = client.count(
        collection_name=collection_name,
        exact=True # Use exact=True for a precise number
    ).count
    print('Initial vectors/nodes in DB:', num_vectors)

    # test that we can add more data
    additional_data = data[50:]
    print(f'Adding {len(additional_data)} nodes')
    vector_index.insert_nodes(additional_data)

    num_vectors = client.count(
        collection_name=collection_name,
        exact=True # Use exact=True for a precise number
    ).count
    print('Updated vectors/nodes in DB:', num_vectors)

    retriever = get_retriever(vector_index)
    results = retriever.retrieve('Vad är en gemensam digital assistent för offentlig sektor?')

    for i, r in enumerate(results):
        print(f'\nResult {i}:')
        print(r)

    #client.delete_collection(collection_name)