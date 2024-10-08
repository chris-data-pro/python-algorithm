# Import the required module for text
# to speech conversion
from gtts import gTTS

# This module is imported so that we can
# play the converted audio
import os
import pydub
from pydub import playback


def text2speech_gtts(text, language='en'):
    # Passing the text and language to the engine,
    # here we have marked slow=False. Which tells
    # the module that the converted audio should
    # have a high speed
    myobj = gTTS(text=text, lang=language, slow=False)

    # Saving the converted audio in a mp3 file named
    # welcome
    myobj.save("welcome.mp3")


def play_audio(file):
    sound = pydub.AudioSegment.from_file(file, format="mp3")
    playback.play(sound)


if __name__ == '__main__':
    text2speech_gtts('Welcome to whatever you wanna hear from me')

    # Playing the converted file
    play_audio('welcome.mp3')

