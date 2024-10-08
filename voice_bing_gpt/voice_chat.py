import re
import os
import boto3
import pydub
import openai
import whisper
import asyncio
import env_config
import var_config
from gtts import gTTS
from typing import List
from pydub import playback
import speech_recognition as sr
from EdgeGPT import Chatbot, ConversationStyle
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector

# Initialize the OpenAI API
openai.api_key = os.environ.get('OPENAI_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Create a recognizer object and wake word variables
recognizer = sr.Recognizer()
BING_WAKE_WORD = "bing"
GPT_WAKE_WORD = "gpt"
SFA_WAKE_WORD = "sfa"

# postgresql+psycopg://localhost:5432/pgvector_example
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg"),
    host=os.environ.get("PGVECTOR_HOST", "localhost"),
    port=int(os.environ.get("PGVECTOR_PORT", "5432")),
    database=os.environ.get("PGVECTOR_DATABASE", "pgvector_example"),
    user=os.environ.get("PGVECTOR_USER", "chrjiang"),
    password=os.environ.get("PGVECTOR_PASSWORD", ""),
)


def get_wake_word(phrase):
    if BING_WAKE_WORD in phrase.lower():
        return BING_WAKE_WORD
    elif GPT_WAKE_WORD in phrase.lower():
        return GPT_WAKE_WORD
    elif SFA_WAKE_WORD in phrase.lower():
        return SFA_WAKE_WORD
    else:
        return None


# audio filepath to text file
def transcribe_audio_to_text(filename, lang='en-US'):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio, language=lang)
        except:
            print("")
            #print('Skipping unknown error')


# text to audio file
def synthesize_speech_gtts(text, output_filename, language='en'):
    myobj = gTTS(text=text, lang=language, slow=False)
    myobj.save(output_filename)


# text to audio file
def synthesize_speech_aws(text, output_filename):
    polly = boto3.client('polly', region_name='us-west-2')
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Salli',
        Engine='neural'
    )

    with open(output_filename, 'wb') as f:
        f.write(response['AudioStream'].read())


def play_audio(file):
    sound = pydub.AudioSegment.from_file(file, format="mp3")
    playback.play(sound)


async def main():
    course = var_config.course_dir.split('/')[-1]
    data_dir = '/Users/chrjiang/github/python-algorithm/gpt/data/' + course
    loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader)

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    docs: List[Document] = text_splitter.split_documents(documents)

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

    while True:

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print(f"\nWaiting for wake words 'ok Bing' or 'ok GPT' or 'ok SFA'...")
            while True:
                source.pause_threshold = 1
                audio = recognizer.listen(source, timeout=2, phrase_time_limit=2)
                try:
                    phrase = recognizer.recognize_google(audio)
                    print(f"You: {phrase}")

                    wake_word = get_wake_word(phrase)
                    if wake_word is not None:
                        break
                    else:
                        print("Not a wake word. Try again.")
                except Exception as e:
                    print("Error transcribing audio: {0}".format(e))
                    continue

            print("Bot: What can I help you with?")
            # after 1st run the open_response.mp3 is saved and won't change, so commented out
            # synthesize_speech_gtts('What can I help you with?', var_config.cwd + '/open_response.mp3')
            play_audio(var_config.cwd + '/open_response.mp3')

            source.pause_threshold = 1
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=6)
            # user_prompt_audiofile = env_config.cwd + "/audio_prompt.wav"
            try:
                # with open(env_config.cwd + "/audio_prompt.wav", "wb") as f:
                #     f.write(audio.get_wav_data())
                # user_input = transcribe_audio_to_text(user_prompt_audiofile)
                user_input = recognizer.recognize_google(audio)
                print(f"You: {user_input}")
            except Exception as e:
                print("Error transcribing audio: {0}".format(e))
                continue

            if wake_word == BING_WAKE_WORD:
                bot = Chatbot(cookiePath=var_config.cwd + '/cookies.json')
                response = await bot.ask(prompt=user_input, conversation_style=ConversationStyle.precise)
                #
                # for message in response["item"]["messages"]:
                #     if message["author"] == "bot":
                #         bot_response = message["text"]
                #
                # bot_response = re.sub('\[\^\d+\^\]', '', bot_response).replace('*', '')
                # Select only the bot response from the response dictionary
                for message in response["item"]["messages"]:
                    if message["author"] == "bot":
                        bot_response = message["text"]
                # Remove [^#^] citations in response
                bot_response = re.sub('\[\^\d+\^\]', '', bot_response).replace('*', '')

            elif wake_word == GPT_WAKE_WORD:
                # Send prompt to GPT-3.5-turbo API
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_input},
                    ],
                    temperature=0.5,
                    max_tokens=150,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    n=1,
                    stop=["\nUser:"],
                )
                bot_response = response["choices"][0]["message"]["content"]

            else:
                neighbors: List[Document] = db.similarity_search(user_input, include_metadata=True)
                bot_response = chain.run(input_documents=neighbors, question=user_input)

        print("Bot:", bot_response)
        synthesize_speech_gtts(bot_response, var_config.cwd + '/response.mp3')
        play_audio(var_config.cwd + '/response.mp3')

    await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
