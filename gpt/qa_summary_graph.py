import nest_asyncio

nest_asyncio.apply()
from common import env_config
from common import var_config
from pathlib import Path
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index.composability.joint_qa_summary import QASummaryGraphBuilder
from llama_index import SimpleDirectoryReader, ServiceContext, LLMPredictor, GPTSimpleVectorIndex
from llama_index.composability import ComposableGraph
from langchain.chat_models import ChatOpenAI


def build_gpt_model(model="gpt-3.5-turbo"):
    """Builds the gpt model.

    If the argument `model` isn't passed in, the default model is used.

    Parameters
    ----------
    model : str, optional
        The name of the model (default is gpt-3.5-turbo).

    Returns
    ----------
    ServiceContext
        The ServiceContext for graph Q&A query.
    """

    llm_predictor_chatgpt = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name=model))
    service_context_chatgpt = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt, chunk_size_limit=1024)

    return service_context_chatgpt


def save_gpt_index(course_name, sc_chatgpt):

    index_dir = var_config.cwd + '/indices/' + course_name
    data_dir = var_config.cwd + '/data/' + course_name
    reader = SimpleDirectoryReader(
        data_dir
    )

    # create data path directory if not exists
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    documents = reader.load_data()

    # NOTE: can also specify an existing docstore, service context, summary text, qa_text, etc.
    graph_builder = QASummaryGraphBuilder(service_context=sc_chatgpt)
    graph = graph_builder.build_graph_from_documents(documents)

    # save to disk
    graph.save_to_disk(index_dir + '/qa_summary_graph.json')


def get_query_configs():
    """Builds the query configs.

    Returns
    ----------
    list
        A list of dicts to set query config.
    """

    return [
        {
            "index_struct_type": "simple_dict",
            "query_mode": "default",
            "query_kwargs": {
                "similarity_top_k": 1
            },
        },
        {
            "index_struct_type": "list",
            "query_mode": "default",
            "query_kwargs": {
                "response_mode": "tree_summarize",
                "use_async": True,
                "verbose": True
            },
        },
        {
            "index_struct_type": "tree",
            "query_mode": "default",
            "query_kwargs": {
                "verbose": True
            },
        },
    ]


if __name__ == '__main__':
    course = var_config.course_dir.split('/')[-1]

    service_context_chatgpt = build_gpt_model("gpt-3.5-turbo")

    # save the graph index to disk
    # save_gpt_index(course, service_context_chatgpt)

    # load from disk
    graph = ComposableGraph.load_from_disk('indices/' + course + '/qa_summary_graph.json')

    # set query config
    query_configs = get_query_configs()

    input1 = 'nonEmpty'
    while input1:
        print('-' * 33)
        print('Please enter your Question here: ')
        print('-' * 33)
        input1 = input()

        if input1 == 'exit' or input1 == '':
            break

        response = graph.query(
            input1,
            query_configs=query_configs,
            service_context=service_context_chatgpt
        )
        print('-' * 8)
        print('Answer: ')
        print('-' * 8)
        print(response)
