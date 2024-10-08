import webvtt
from llama_index import Document
from llama_index.schema import MetadataMode


def vtt2text(input_path):
    vtt = webvtt.read(input_path)
    transcript = ""

    lines = []
    for line in vtt:
        lines.extend(line.text.strip().splitlines())

    previous = None
    for line in lines:
        if line == previous:
            continue
        transcript += " " + line
        previous = line

    for caption in vtt:
        start_time = caption.start
        end_time = caption.end
        text = caption.text

        # Process the transcript and timestamps as desired
        transcript += ("\n" + f"Timestamp: {start_time} - {end_time}")
        transcript += (f" Text: {text}")

    return transcript


def txt2document(txt_file_path: str, txt_file_name: str) -> Document:
    """Transform Course Text Contents txt file into customized Document.

    Parameters
    ----------
    txt_file_path : str, required
        The path to the txt file ending with '/'.
    txt_file_name : str, required
        The name of the txt file, e.g. abc.txt
    """
    with open(txt_file_path + txt_file_name, "r") as my_input_file:
        data: str = my_input_file.read()
        category: str = "module" if txt_file_name.startswith("m") else "video"
        course: str = txt_file_path.rsplit('/', 2)[1]
        return Document(
            text=data,
            metadata={
                "file_name": txt_file_name,
                "category": category,
                "course": course
            },
            excluded_llm_metadata_keys=['file_name'],
            metadata_seperator="\n",
            metadata_template="{key} => {value}",
            text_template="Metadata: \n{metadata_str}\n-----\nContent: \n{content}",
        )


if __name__ == '__main__':
    document = txt2document("/Users/chrjiang/github/python-algorithm/gpt/data/bnet/", "6249848734001_5.2.1 - Devices in a Bubble__eng.txt")
    print(document.get_content(metadata_mode=MetadataMode.EMBED))
#     vtt = webvtt.read('/Users/chrjiang/Documents/Work/i2ds/English VTT Files/1.0.1 - The Data Analyst.vtt')
#     transcript = ""
#
#     lines = []
#     for line in vtt:
#         lines.extend(line.text.strip().splitlines())
#
#     previous = None
#     for line in lines:
#         if line == previous:
#            continue
#         transcript += " " + line
#         previous = line
#
#     print(transcript)
