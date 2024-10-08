import openai
import asyncio
import re
import os
import whisper
import boto3
import pydub
from gtts import gTTS
from pydub import playback
import speech_recognition as sr
from EdgeGPT import Chatbot, ConversationStyle
import env_config

# Initialize the OpenAI API
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Create a recognizer object and wake word variables
recognizer = sr.Recognizer()
BING_WAKE_WORD = "bingo"
GPT_WAKE_WORD = "gpt"


def get_wake_word(phrase):
    if BING_WAKE_WORD in phrase.lower():
        return BING_WAKE_WORD
    elif GPT_WAKE_WORD in phrase.lower():
        return GPT_WAKE_WORD
    else:
        return None


# audio to text file
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


# text to audio file
def synthesize_speech_gtts(text, output_filename, language='zh'):
    myobj = gTTS(text=text, lang=language, slow=False)
    myobj.save(output_filename)


def play_audio(file):
    sound = pydub.AudioSegment.from_file(file, format="mp3")
    playback.play(sound)


async def main():
    while True:

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print(f"\nWaiting for wake words 'ok bingo' or 'ok gpt'...")
            while True:
                source.pause_threshold = 1
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                try:
                    # with open(env_config.cwd + "/audio.wav", "wb") as f:
                    #     f.write(audio.get_wav_data())
                    # # Use the preloaded tiny_model
                    # model = whisper.load_model("tiny")
                    # result = model.transcribe(env_config.cwd + "/audio.wav", fp16=False)
                    # phrase = result["text"]
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

            print("Bot: 有什么可以帮到您？")
            synthesize_speech_gtts('有什么可以帮到您？', env_config.cwd + '/response.mp3')
            play_audio(env_config.cwd + '/response.mp3')
            source.pause_threshold = 1
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=8)
            user_prompt_audiofile = env_config.cwd + "/audio_prompt.wav"

            try:
                with open(user_prompt_audiofile, "wb") as f:
                    f.write(audio.get_wav_data())
                # model = whisper.load_model("base")
                # result = model.transcribe(env_config.cwd + "/audio_prompt.wav", fp16=False)
                # user_input = result["text"]
                user_input = transcribe_audio_to_text(user_prompt_audiofile, lang='zh-CN')
                print(f"You: {user_input}")
            except Exception as e:
                print("Error transcribing audio: {0}".format(e))
                continue

            if wake_word == BING_WAKE_WORD:
                bot = Chatbot(cookiePath=env_config.cwd + '/cookies.json')
                response = await bot.ask(prompt=user_input, conversation_style=ConversationStyle.precise)

                for message in response["item"]["messages"]:
                    if message["author"] == "bot":
                        bot_response = message["text"]

                bot_response = re.sub('\[\^\d+\^\]', '', bot_response)
                # Select only the bot response from the response dictionary
                for message in response["item"]["messages"]:
                    if message["author"] == "bot":
                        bot_response = message["text"]
                # Remove [^#^] citations in response
                bot_response = re.sub('\[\^\d+\^\]', '', bot_response)

            else:
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

        print("Bot:", bot_response)
        synthesize_speech_gtts(bot_response, env_config.cwd + '/response.mp3')
        play_audio(env_config.cwd + '/response.mp3')

    await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
