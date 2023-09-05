import queue
import threading
from typing import Tuple

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, Request
from faster_whisper import WhisperModel

app = FastAPI()


def create_whisper_model() -> WhisperModel:
    whisper = WhisperModel("large-v2", device="cpu", compute_type="int8", cpu_threads=4, download_root="./models")
    print("Loaded model")
    return whisper


# Initialize a queue to hold the loaded models
model_lock = threading.Lock()
model_pool = queue.Queue()
NUM_MODELS = 10

# Load initial models into the pool
for i in range(NUM_MODELS):
    model_pool.put({"id": i, "model": create_whisper_model()})


async def parse_body(request: Request):
    data: bytes = await request.body()
    return data


def get_model_from_pool() -> Tuple[int, WhisperModel]:
    m = model_pool.get()
    model = m["model"]
    model_id = m["id"]
    print(f"Got model {model_id}")
    return model_id, model


def put_model_in_pool(model_id: int, model: WhisperModel) -> None:
    model_pool.put({"id": model_id, "model": model})
    print(f"Put model {model_id}")


@app.post("/predict")
async def predict(audio_data: bytes = Depends(parse_body)):
    # Convert the audio bytes to a NumPy array
    audio_data_array: np.ndarray = np.frombuffer(audio_data, np.int16).astype(np.float32) / 255.0

    with model_lock:
        model_id, model = get_model_from_pool()

    try:
        segments, _ = model.transcribe(audio_data_array,
                                       language="en",
                                       beam_size=5,
                                       vad_filter=True,
                                       vad_parameters=dict(min_silence_duration_ms=1000))
        segments = [s.text for s in segments]
        transcription = " ".join(segments)
    except Exception as e:
        print(e)
        transcription = "Error"
    finally:
        put_model_in_pool(model_id, model)

    return {"prediction": transcription}


if __name__ == "__main__":
    # Run the FastAPI app with multiple threads
    uvicorn.run(app, host="0.0.0.0", port=8008)
