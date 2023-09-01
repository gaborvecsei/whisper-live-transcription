"""
How does it work?

- step = 1
- length = 4

------------------------------------------
1st second: [t,   0,   0,   0] --> "Hi"
2nd second: [t-1, t,   0,   0] --> "Hi I am"
3rd second: [t-2, t-1, t,   0] --> "Hi I am the one"
4th second: [t-3, t-2, t-1, t] --> "Hi I am the one and only Gabor"
5th second: [t,   0,   0,   0] --> "How" --> Here we started the process again, and the output is in a new line
6th second: [t-1, t,   0,   0] --> "How are"
------------------------------------------
"""

import queue
import re
import threading

import numpy as np
import pyaudio
from faster_whisper import WhisperModel

# Yeah I could do this config with argparse, but I won't...

# Audio settings
STEP_IN_SEC: int = 1    # We'll increase the processable audio data by this
LENGHT_IN_SEC: int = 6    # We'll process this amount of audio data together maximum
NB_CHANNELS = 1
RATE = 16000
CHUNK = RATE

# Whisper settings
WHISPER_LANGUAGE = "en"
WHISPER_THREADS = 4

# Visualization (expected max number of characters for LENGHT_IN_SEC audio)
MAX_SENTENCE_CHARACTERS = 80

# This queue holds all the 1-second audio chunks
audio_queue = queue.Queue()

# This queue holds all the chunks that will be processed together
# If the chunk is filled to the max, it will be emptied
length_queue = queue.Queue(maxsize=LENGHT_IN_SEC)

# Whisper model
whisper = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=WHISPER_THREADS, download_root="./models")


def producer_thread():
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=NB_CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,    # 1 second of audio
    )

    print("-" * 80)
    print("Microphone initialized, recording started...")
    print("-" * 80)
    print("TRANSCRIPTION")
    print("-" * 80)

    while True:
        audio_data = b""
        for _ in range(STEP_IN_SEC):
            chunk = stream.read(RATE)    # Read 1 second of audio data
            audio_data += chunk

        audio_queue.put(audio_data)    # Put the 5-second audio data into the queue


# Thread which gets items from the queue and prints its length
def consumer_thread():
    while True:
        if length_queue.qsize() >= LENGHT_IN_SEC:
            with length_queue.mutex:
                length_queue.queue.clear()
                print()

        audio_data = audio_queue.get()
        length_queue.put(audio_data)

        # Concatenate audio data in the lenght_queue
        audio_data_to_process = b""
        for i in range(length_queue.qsize()):
            # We index it so it won't get removed
            audio_data_to_process += length_queue.queue[i]

        # convert the bytes data toa  numpy array
        audio_data_array: np.ndarray = np.frombuffer(audio_data_to_process, np.int16).astype(np.float32) / 255.0
        # audio_data_array = np.expand_dims(audio_data_array, axis=0)

        segments, _ = whisper.transcribe(audio_data_array,
                                         language=WHISPER_LANGUAGE,
                                         beam_size=5,
                                         vad_filter=True,
                                         vad_parameters=dict(min_silence_duration_ms=1000))
        segments = [s.text for s in segments]
        transcription = " ".join(segments)
        # remove anything from the text which is between () or [] --> these are non-verbal background noises/music/etc.
        transcription = re.sub(r"\[.*\]", "", transcription)
        transcription = re.sub(r"\(.*\)", "", transcription)
        # We do this for the more clean visualization (when the next transcription we print would be shorter then the one we printed)
        transcription = transcription.ljust(MAX_SENTENCE_CHARACTERS, " ")
        print(transcription, end='\r', flush=True)

        audio_queue.task_done()


if __name__ == "__main__":
    producer = threading.Thread(target=producer_thread)
    producer.start()

    consumer = threading.Thread(target=consumer_thread)
    consumer.start()

    try:
        producer.join()
        consumer.join()
    except KeyboardInterrupt:
        print("Exiting...")
