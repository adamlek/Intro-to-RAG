from ingest import parse_and_get_nodes
from index import get_vector_index, get_retriever

def main():
    # get data
    data = parse_and_get_nodes('data/')
    
    # get qdrant client, the collection name and our vector index object
    client, collection_name, vector_index = get_vector_index(data)
    
    # set up a retriever for the vector index
    # post_processors don't really work with retriever.as_retriever() 
    # for some reason, set as query_engine instead
    retriever = get_retriever(vector_index)

    question = 'Vad är en gemensam digital assistent för offentlig sektor?'
    print(f'+ We ask the following quesion:\n{question}\n')

    results = retriever.query(question)

    print('+ Retrieved chunks')
    for i, r in enumerate(results.source_nodes):
        print(f'\n++ Retrieved chunk {i}:')
        print(r)

    client.delete_collection(collection_name)

if __name__ == "__main__":
    main()

