import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_chunked_reports(data_path):
    """Generates chunked reports from the text content feeds.

    Parameters
    ----------
    data_path : str, required
        The path of the txt files.

    Returns
    ----------
    list
        A list of lists. Each list inside is a chunked report, aka a list of Documents
        e.g.
        [
            Document(page_content="xxx", metadata={'source': 'xxx'}),
            Document(page_content='quick_nav', metadata={'source': '/Users/chrjiang/github/python-algorithm/gpt/data/i2ds/m2.txt'})
            ...
            Document(page_content="xxx", metadata={'source': 'xxx'})
        ]
    """

    reports = []
    for txt in os.listdir(data_path):
        loader = TextLoader(data_path + '/' + txt)
        document = loader.load()
        reports.append(document)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    chunked_reports = []
    for report in reports:
        # Chunk the report
        texts = text_splitter.split_documents(report)
        # Add the chunks to chunked_annual_reports, which is a list of lists
        chunked_reports.append(texts)
        print(f"chunked_report length: {len(texts)}")
        # print(texts)

    return chunked_reports
