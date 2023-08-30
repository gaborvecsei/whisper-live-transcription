# Live Transcription with Whisper

Sample with an `Macbook Pro (M1)` with the `tiny ggml` model:

https://github.com/gaborvecsei/whisper-live-transcription/assets/18753533/6364a160-4043-437d-be36-52e84a91fe60

(*🔈 sound on*, [audio source](https://www.youtube.com/watch?v=-WSrY-xH5pI))

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

