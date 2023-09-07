import asyncio

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, Request
from faster_whisper import WhisperModel

app = FastAPI()

NUM_WORKERS = 10


def create_whisper_model() -> WhisperModel:
    whisper = WhisperModel("base",
                           device="cpu",
                           compute_type="int8",
                           num_workers=NUM_WORKERS,
                           cpu_threads=1,
                           download_root="./models")
    print("Loaded model")
    return whisper


model = create_whisper_model()
print("Loaded model")


async def parse_body(request: Request):
    data: bytes = await request.body()
    return data


def execute_blocking_whisper_prediction(model: WhisperModel, audio_data_array) -> str:
    segments, _ = model.transcribe(audio_data_array,
                                   language="en",
                                   beam_size=5,
                                   vad_filter=False,
                                   vad_parameters=dict(min_silence_duration_ms=1000))
    segments = [s.text for s in segments]
    transcription = " ".join(segments)
    transcription = transcription.strip()
    return transcription


@app.post("/predict")
async def predict(audio_data: bytes = Depends(parse_body)):
    # Convert the audio bytes to a NumPy array
    audio_data_array: np.ndarray = np.frombuffer(audio_data, np.int16).astype(np.float32) / 255.0

    try:
        # Run the prediction on the audio data
        result = await asyncio.get_running_loop().run_in_executor(None, execute_blocking_whisper_prediction, model,
                                                                  audio_data_array)

    except Exception as e:
        print(e)
        result = "Error"

    return {"prediction": result}


if __name__ == "__main__":
    # Run the FastAPI app with multiple threads
    uvicorn.run(app, host="0.0.0.0", port=8008)
