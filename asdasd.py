import random
import string

import gradio as gr


def update(audio_stream_input):
    length = random.randint(1, 300)
    rnd = list(''.join(random.choices(string.ascii_letters + string.digits, k=length)))
    num_breaks = random.randint(0, 5)
    for _ in range(num_breaks):
        pos = random.randint(0, len(rnd))
        rnd.insert(pos, '\n')
    return ''.join(rnd) + f"\n\n {audio_stream_input[1][:5]}"

with gr.Blocks() as demo:
    with gr.Row():
        mic_audio_input = gr.Audio(sources=["microphone"], streaming=True)
    with gr.Row():
        audio_stream_out = gr.Markdown()

    mic_audio_input.stream(update, [mic_audio_input], [audio_stream_out], show_progress="hidden")

demo.launch(share=True,server_name="0.0.0.0", debug=True)
