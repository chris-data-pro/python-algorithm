from llama_index import Document
from llama_index.schema import MetadataMode

if __name__ == '__main__':
    document = Document(
        text="This is a super-customized document",
        metadata={
            "file_name": "super_secret_document.txt",
            "category": "finance",
            "author": "LlamaIndex"
        },
        excluded_llm_metadata_keys=['file_name'],
        metadata_seperator="\n",
        metadata_template="{key} => {value}",
        text_template="Metadata: \n{metadata_str}\n-----\nContent: \n{content}",
    )

    print("The LLM sees this: \n", document.get_content(metadata_mode=MetadataMode.LLM))
    print()
    print("The Embedding model sees this: \n", document.get_content(metadata_mode=MetadataMode.EMBED))
