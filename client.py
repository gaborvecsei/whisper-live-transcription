import queue
import re
import sys
import threading
import time
from typing import Dict, List

import numpy as np
import pyaudio
import requests

# Yeah I could do this config with argparse, but I won't...

# Audio settings
STEP_IN_SEC: int = 1    # We'll increase the processable audio data by this
LENGHT_IN_SEC: int = 6    # We'll process this amount of audio data together maximum
NB_CHANNELS = 1
RATE = 16000
CHUNK = RATE

# Visualization (expected max number of characters for LENGHT_IN_SEC audio)
MAX_SENTENCE_CHARACTERS = 80

TRANSCRIPTION_API_ENDPOINT = "http://localhost:8008/predict"

# This queue holds all the 1-second audio chunks
audio_queue = queue.Queue()

# This queue holds all the chunks that will be processed together
# If the chunk is filled to the max, it will be emptied
length_queue = queue.Queue(maxsize=LENGHT_IN_SEC)


def send_audio_to_server(audio_data) -> str:
    response = requests.post(TRANSCRIPTION_API_ENDPOINT,
                             data=audio_data,
                             headers={'Content-Type': 'application/octet-stream'})
    result = response.json()
    return result["prediction"]


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
def consumer_thread(stats):
    while True:
        if length_queue.qsize() >= LENGHT_IN_SEC:
            with length_queue.mutex:
                length_queue.queue.clear()
                print()

        audio_data = audio_queue.get()
        transcription_start_time = time.time()
        length_queue.put(audio_data)

        # Concatenate audio data in the lenght_queue
        audio_data_to_process = b""
        for i in range(length_queue.qsize()):
            # We index it so it won't get removed
            audio_data_to_process += length_queue.queue[i]

        try:
            transcription = send_audio_to_server(audio_data_to_process)
            # remove anything from the text which is between () or [] --> these are non-verbal background noises/music/etc.
            transcription = re.sub(r"\[.*\]", "", transcription)
            transcription = re.sub(r"\(.*\)", "", transcription)
        except:
            transcription = "Error"

        transcription_end_time = time.time()

        # We do this for the more clean visualization (when the next transcription we print would be shorter then the one we printed)
        transcription_to_visualize = transcription.ljust(MAX_SENTENCE_CHARACTERS, " ")

        transcription_postprocessing_end_time = time.time()

        sys.stdout.write('\033[K' + transcription_to_visualize + '\r')

        audio_queue.task_done()

        overall_elapsed_time = transcription_postprocessing_end_time - transcription_start_time
        transcription_elapsed_time = transcription_end_time - transcription_start_time
        postprocessing_elapsed_time = transcription_postprocessing_end_time - transcription_end_time
        stats["overall"].append(overall_elapsed_time)
        stats["transcription"].append(transcription_elapsed_time)
        stats["postprocessing"].append(postprocessing_elapsed_time)


if __name__ == "__main__":
    stats: Dict[str, List[float]] = {"overall": [], "transcription": [], "postprocessing": []}

    producer = threading.Thread(target=producer_thread)
    producer.start()

    consumer = threading.Thread(target=consumer_thread, args=(stats,))
    consumer.start()

    try:
        producer.join()
        consumer.join()
    except KeyboardInterrupt:
        print("Exiting...")
        # print out the statistics
        print("Number of processed chunks: ", len(stats["overall"]))
        print(f"Overall time: avg: {np.mean(stats['overall']):.4f}s, std: {np.std(stats['overall']):.4f}s")
        print(
            f"Transcription time: avg: {np.mean(stats['transcription']):.4f}s, std: {np.std(stats['transcription']):.4f}s"
        )
        print(
            f"Postprocessing time: avg: {np.mean(stats['postprocessing']):.4f}s, std: {np.std(stats['postprocessing']):.4f}s"
        )
        # We need to add the step_in_sec to the latency as we need to wait for that chunk of audio
        print(f"The average latency is {np.mean(stats['overall'])+STEP_IN_SEC:.4f}s")
