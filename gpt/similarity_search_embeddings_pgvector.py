import os
import openai
from gpt.common import env_config
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, insert, select, text, Integer, String, Text
from sqlalchemy.orm import declarative_base, mapped_column, Session

if __name__ == '__main__':
    engine = create_engine('postgresql+psycopg://localhost/pgvector_example')
    with engine.connect() as conn:
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        conn.commit()

    Base = declarative_base()


    class Document(Base):
        __tablename__ = 'document'

        id = mapped_column(Integer, primary_key=True)
        content = mapped_column(Text)
        embedding = mapped_column(Vector(1536))


    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    input_data = [
        'The dog is barking',
        'The cat is purring',
        'The bear is growling',
        'I like ice cream',
        'New York is cold',
        'What is your favorite show',
        'A story that ends all stories',
        'The cat sat on the windowsill and watched the birds outside.',
        'It was a dark and stormy night, and the wind howled through the trees.',
        'John went to the store to buy some milk and bread, but he forgot his wallet at home.',
        'The sky was a brilliant shade of orange and pink as the sun set over the ocean.',
        'She picked up the book and began to read, losing herself in the story.',
        'The children ran and played in the park, laughing and shouting with joy.',
        'He sat at his desk, staring blankly at the computer screen, trying to think of something to write.',
        'The old man shuffled slowly down the street, his cane tapping against the pavement with each step.',
        'The smell of freshly baked cookies wafted through the air, making her mouth water.',
        'The airplane flew high above the clouds, the passengers looking down at the world far below them.'
    ]

    openai.api_key = os.environ.get('OPENAI_API_KEY')

    embeddings = [v['embedding'] for v in openai.Embedding.create(input=input_data, model='text-embedding-ada-002')['data']]
    documents = [dict(content=input_data[i], embedding=embedding) for i, embedding in enumerate(embeddings)]

    session = Session(engine)
    session.execute(insert(Document), documents)

    # doc = session.get(Document, 1)
    # neighbors = session.scalars \
    #     (select(Document).filter(Document.id != doc.id).order_by(Document.embedding.max_inner_product(doc.embedding)).limit
    #         (5))
    # for neighbor in neighbors:
    #     print(neighbor.content)

    query = ''
    while query != 'exit':
        print('-' * 30)
        print('Please enter your Query here: ')
        print('-' * 30)
        query = input()

        if query == 'exit' or query == '':
            break

        new_embedding = openai.Embedding.create(input=[query], model='text-embedding-ada-002')['data'][0]['embedding']

        # Modify the document with the new content
        doc = Document(content=query, embedding=new_embedding)
        session.add(doc)

        # Perform similarity search for the modified document
        neighbors = session.scalars(
            select(Document)
            .filter(Document.id != doc.id)
            .order_by(Document.embedding.max_inner_product(doc.embedding))
            .limit(5)
        )

        print('-' * 50)
        print(f'Top 5 similar documents to "{query}": ')
        print('-' * 50)
        for neighbor in neighbors:
            print(neighbor.content)
