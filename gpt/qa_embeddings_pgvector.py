import os
from typing import List

from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector

from common import var_config
from common import env_config
from helper import video_transcript_2_txt

if __name__ == '__main__':
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

    # postgresql+psycopg://localhost:5432/pgvector_example
    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver=os.environ.get("PGVECTOR_DRIVER", "psycopg"),
        host=os.environ.get("PGVECTOR_HOST", "localhost"),
        port=int(os.environ.get("PGVECTOR_PORT", "5432")),
        database=os.environ.get("PGVECTOR_DATABASE", "pgvector_example"),
        user=os.environ.get("PGVECTOR_USER", "chrjiang"),
        password=os.environ.get("PGVECTOR_PASSWORD", ""),
    )

    course = var_config.course_dir.split('/')[-1]
    data_dir = var_config.cwd + '/data/' + course + '/'

    # 1
    # documents is a list of Documents. Each one is Document(page_content="xxx", metadata={'source': 'xxx'})
    # len(documents) is the same as number of txt files inder data_dir. e.g. 18 for i2ds
    loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    docs: List[Document] = text_splitter.split_documents(documents)

    # 2
    # docs = [video_transcript_2_txt.txt2document(data_dir, file_name) for file_name in os.listdir(data_dir)]

    # to print out the chunks:
    print(len(docs))
    for i in range(5):
        print(docs[i])

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # The PGVector Module will try to create a table with the name of the collection.
    # So, make sure that the collection name is unique and the user has the permission to create a table.
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=course,
        connection_string=CONNECTION_STRING,
    )

    # Q&A about the report
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm)

    # query = ''
    while True:
        print('\n' + '-' * 33)
        print('Please enter your Question here: ')
        print('-' * 33)
        query = input()

        if query == 'exit' or query == '':
            break

        neighbors: List[Document] = db.similarity_search(query, include_metadata=True)
        for neighbor in neighbors:
            print(neighbor)
        response = chain.run(input_documents=neighbors, question=query)

        print('-' * 8 + '\nAnswer: \n' + '-' * 8)
        print(response)
