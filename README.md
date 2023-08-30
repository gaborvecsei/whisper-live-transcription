# Live Transcription with Whisper

https://github.com/gaborvecsei/whisper-live-transcription/assets/18753533/99ad0808-f212-4f5e-9406-7ee5911b9f5c
(*🔈 sound on*)

# Setup

- `$ pip install -r requirements.txt`
- `$ mkdir models`
- Download `ggml` whisper model binaries: [how-to](https://github.com/ggerganov/whisper.cpp/tree/master/models) and place them in the models folder
    - [available ggml models](https://ggml.ggerganov.com/)
- Adjust parameters in the script

# Run

```shell
$ python ./live-transcribe.py
```

