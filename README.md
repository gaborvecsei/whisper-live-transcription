# Live Transcription with Whisper

# Sample

Sample with an `Macbook Pro (M1)`

https://github.com/gaborvecsei/whisper-live-transcription/assets/18753533/6364a160-4043-437d-be36-52e84a91fe60

(_🔈 sound on_, [audio source](https://www.youtube.com/watch?v=-WSrY-xH5pI), `whisper-cpp-python` package, `tiny` model)

# Setup

- `$ pip install -r requirements.txt`
- `$ mkdir models`

## `whisper.cpp` based (`ggml`) ([`whisper-cpp-python`](https://github.com/carloscdias/whisper-cpp-python) and [`whispercpp`](https://github.com/aarnphm/whispercpp) packages)

- Download `ggml` whisper model binaries: [how-to](https://github.com/ggerganov/whisper.cpp/tree/master/models) and place them in the models folder
  - [available ggml models](https://ggml.ggerganov.com/)
- Adjust parameters in the script

## `faster-whisper` package

- Nothing to do

# Run

```
$ python ./live-transcribe.py
```
