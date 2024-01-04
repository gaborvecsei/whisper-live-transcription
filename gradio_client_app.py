import datetime
import io
import re
import time

import gradio as gr
import librosa
import numpy as np
import requests

STEPS_IN_SEC: int = 1    # How frequently we should run the transcription
LENGHT_IN_SEC: int = 6    # We'll process this amount of audio data together maximum

# TRANSCRIPTION_API_ENDPOINT = "http://localhost:8008/predict_other"
TRANSCRIPTION_API_ENDPOINT = "http://localhost:8008/predict"


def send_audio_to_server(audio_data: np.ndarray) -> str:
    audio_data_bytes = audio_data.astype(np.int16).tobytes()

    # Send octet stream
    response = requests.post(TRANSCRIPTION_API_ENDPOINT,
                             data=audio_data_bytes,
                             headers={
                                 "accept": "application/json",
                                 "Content-Type": "application/octet-stream"
                             })

    result = response.json()
    return result["prediction"]


def dummy_function(stream, new_chunk, transcription_display):
    sampling_rate, y = new_chunk

    y = y.astype(np.float32)

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y

    # Perform the transcription every second
    # TODO: can be problematic - as in theory we could get a chunk which is 0.5 sec long
    if len(stream) % sampling_rate != 0:
        return stream, transcription_display

    print(f"[*] {datetime.datetime.now()} | Stream length: {len(stream)}")

    transcription = "Error"
    try:
        # We need to resample the audio chunk to 16kHz
        # (https://github.com/jonashaag/audio-resampling-in-python)
        # We need to resample here, because if we resample chunk by chunk, the audio will be distorted
        stream_resampled = librosa.resample(stream, orig_sr=sampling_rate, target_sr=16000)

        transcription = send_audio_to_server(stream_resampled)
        print(transcription)
        # remove anything from the text which is between () or [] --> these are non-verbal background noises/music/etc.
        # transcription = re.sub(r"\[.*\]", "", transcription)
        # transcription = re.sub(r"\(.*\)", "", transcription)
    except:
        print("[*] There is an error with the transcription")

    display_text = transcription_display + "\n\n" + transcription

    # Reset the stream if it's already exceeding the maximum length
    if len(stream) > sampling_rate * LENGHT_IN_SEC:
        stream = None
        print("RESET")

    return stream, display_text


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Live Transcription")

    # This state stores the audio data that we'll process
    stream_state = gr.State()

    mic_audio_input = gr.Audio(sources=["microphone"], streaming=True)
    gr.Markdown("## Transcription")
    transcription_display = gr.Markdown("Transcription will appear here")

    # In gradio the default samplign rate is 48000 (https://github.com/gradio-app/gradio/issues/6526)
    # and the chunks size varies between 24000 and 48000 - so between 0.5sec and 1 sec
    mic_audio_input.stream(dummy_function, [stream_state, mic_audio_input, transcription_display],
                           [stream_state, transcription_display])

# Launch the Gradio app
demo.launch(server_name="0.0.0.0")
