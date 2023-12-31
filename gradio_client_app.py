import time

import gradio as gr
import librosa
import numpy as np
import pandas as pd
import requests

STEPS_IN_SEC: int = 1    # How frequently we should run the transcription
LENGHT_IN_SEC: int = 6    # We'll process this amount of audio data together maximum

TRANSCRIPTION_API_ENDPOINT = "http://localhost:8008/predict"


def send_audio_to_server(audio_data: np.ndarray) -> str:
    # This is how the server expects the data
    audio_data_bytes = audio_data.astype(np.int16).tobytes()

    response = requests.post(TRANSCRIPTION_API_ENDPOINT,
                             data=audio_data_bytes,
                             headers={
                                 "accept": "application/json",
                                 "Content-Type": "application/octet-stream"
                             })

    result = response.json()
    return result["prediction"]


def dummy_function(stream, new_chunk, max_length, latency_data, current_transcription, transcription_history):
    start_time = time.time()

    if latency_data is None:
        latency_data = {
            "total_resampling_latency": [],
            "total_transcription_latency": [],
            "total_latency": [],
        }

    sampling_rate, y = new_chunk
    y = y.astype(np.float32)

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y

    # Perform the transcription every second
    # TODO: can be problematic - as in theory we could get a chunk which is 0.5 sec long - but it's good enough for now
    # if len(stream) % sampling_rate != 0:
    #     return stream, transcription_display, information_table_outout, latency_data, current_transcription, transcription_history

    transcription = "ERROR"
    try:
        sampling_start_time = time.time()
        # We need to resample the audio chunk to 16kHz (without this we don't have any output) - gradio cannot handle this
        # (https://github.com/jonashaag/audio-resampling-in-python)
        # We need to resample the concatenated stream, because if we resample chunk by chunk, the audio will be distorted
        stream_resampled = librosa.resample(stream, orig_sr=sampling_rate, target_sr=16000)
        sampling_end_time = time.time()
        latency_data["total_resampling_latency"].append(sampling_end_time - sampling_start_time)

        transcription_start_time = time.time()
        transcription = send_audio_to_server(stream_resampled)
        current_transcription = f"{transcription}"
        # remove anything from the text which is between () or [] --> these are non-verbal background noises/music/etc.
        # transcription = re.sub(r"\[.*\]", "", transcription)
        # transcription = re.sub(r"\(.*\)", "", transcription)
        transcription_end_time = time.time()
        latency_data["total_transcription_latency"].append(transcription_end_time - transcription_start_time)

    except Exception as e:
        print("[*] There is an error with the transcription", e)

    end_time = time.time()
    latency_data["total_latency"].append(end_time - start_time)

    # Reset the stream if it's already exceeding the maximum length
    # This is required as for a longer audio the latency increases and we'd like to keep it as low as possible
    if len(stream) > sampling_rate * max_length:
        stream = None
        transcription_history.append(current_transcription)
        current_transcription = ""

    display_text = f"{current_transcription}\n\n"
    display_text += "\n\n".join(transcription_history[::-1])

    info_df = pd.DataFrame(latency_data)
    info_df = info_df.apply(lambda x: x * 1000)
    info_df = info_df.describe().loc[["min", "max", "mean"]]
    info_df = info_df.round(2)
    info_df = info_df.astype(str) + " ms"
    info_df = info_df.T

    return stream, display_text, info_df.to_markdown(), latency_data, current_transcription, transcription_history


custom_css = """
.transcription_display_container {max-height: 500px; overflow-y: scroll}
footer {visibility: hidden}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Live Transcription\n\n**Right now only English is supported**")

    # Stores the audio data that we'll process
    stream_state = gr.State(None)
    # Stores the latency data
    latency_data_state = gr.State(None)
    # Stores the transcription history that we can visualize
    transcription_history_state = gr.State([])
    # Stores the current transcription (as we process the audio in chunks, and we have a maximum lengtht of audio that we process together)
    current_transcription_state = gr.State("")

    with gr.Row():
        mic_audio_input = gr.Audio(sources=["microphone"], streaming=True)
        reset_button = gr.Button("Reset")
        max_length_input = gr.Slider(value=6, minimum=2, maximum=30, step=1, label="Max length of audio (sec)")

    gr.Markdown("## Transcription\n\n(audio is sent to the server each second)\n\n---------")
    transcription_display = gr.Markdown(elem_classes=["transcription_display_container"])

    gr.Markdown("## Statistics\n\n---------")

    information_table_outout = gr.Markdown("Info about latency will be shown here")

    # In gradio the default samplign rate is 48000 (https://github.com/gradio-app/gradio/issues/6526)
    # and the chunks size varies between 24000 and 48000 - so between 0.5sec and 1 sec
    mic_audio_input.stream(dummy_function, [
        stream_state, mic_audio_input, max_length_input, latency_data_state, current_transcription_state,
        transcription_history_state
    ], [
        stream_state, transcription_display, information_table_outout, latency_data_state, current_transcription_state,
        transcription_history_state
    ],
                           show_progress="hidden")

    def _reset_button_click(stream_state, transcription_display, information_table_outout, latency_data_state,
                            transcription_history_state, current_transcription_state):
        stream_state = None
        transcription_display = ""
        information_table_outout = ""
        latency_data_state = None
        transcription_history_state = []
        current_transcription_state = ""

        return stream_state, transcription_display, information_table_outout, latency_data_state, transcription_history_state, current_transcription_state

    reset_button.click(_reset_button_click, [
        stream_state, transcription_display, information_table_outout, latency_data_state, transcription_history_state,
        current_transcription_state
    ], [
        stream_state, transcription_display, information_table_outout, latency_data_state, transcription_history_state,
        current_transcription_state
    ])

demo.launch(server_name="0.0.0.0", server_port=18445)
