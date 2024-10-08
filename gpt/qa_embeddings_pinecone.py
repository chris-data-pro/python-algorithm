import os
import time
import pinecone

from gpt.common import env_config
from gpt.common import var_config
from gpt.helper import langchain_chunk
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_pinecone_index(name):
    """Creates a pinecone index.

    Parameters
    ----------
    name : str, required
        The name of the index.
    """

    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

    # Initialize Pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )

    print("\nCreating a pinecone index named '{}' for this course".format(name))
    create_tic = time.perf_counter()

    if name not in pinecone.list_indexes():
        pinecone.create_index(name, dimension=1536, metric="euclidean", pod_type="p1")

    create_toc = time.perf_counter()
    print(f"The index was created in {create_toc - create_tic:0.4f} seconds")


def delete_pinecone_index(name):
    """Deletes the index to avoid exceeding the quota of 1 pods by 1 pods, for free account.

    Parameters
    ----------
    name : str, required
        The name of the index.
    """

    print("\nDeleting the pinecone index named '{}' for this course".format(name))
    delete_tic = time.perf_counter()

    pinecone.delete_index(name)

    delete_toc = time.perf_counter()
    print(f"The index was deleted in {delete_toc - delete_tic:0.4f} seconds")


def get_vector_store(index, embeddings, reports):
    """Retrieves the report vector embeddings from Pinecone, after upserting reports to Pinecone.

    Parameters
    ----------
    index : str, required
        The name of the index.
    embeddings: OpenAIEmbeddings, required
        The embeddings into pinecone.
    reports : list, required
        The list of chunked report.

    Returns
    -------
    vector store
        The vector store for Q&A similarity search
    """

    # Upsert reports to Pinecone via LangChain.
    for chunks in reports:
        Pinecone.from_texts([chunk.page_content for chunk in chunks], embeddings, index_name=index)

    return Pinecone.from_existing_index(index_name=index, embedding=embeddings)


if __name__ == '__main__':
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

    course = var_config.course_dir.split('/')[-1]
    chunked_reports = langchain_chunk.get_chunked_reports('data/' + course)

    # Create a pinecone index for this course using the course name.
    # index_name = var_setup.course_dir.split('/')[-1]
    create_pinecone_index(course)

    vectorstore = get_vector_store(course, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), chunked_reports)

    # Q&A about the report
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm)

    query = ''
    while query != 'exit':
        print('-' * 33)
        print('Please enter your Question here: ')
        print('-' * 33)
        query = input()

        if query == 'exit' or query == '':
            break

        docs = vectorstore.similarity_search(query, include_metadata=True)
        response = chain.run(input_documents=docs, question=query)

        print('-' * 8)
        print('Answer: ')
        print('-' * 8)
        print(response)

    delete_pinecone_index(course)
