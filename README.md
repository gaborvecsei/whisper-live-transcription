# Live Transcription with Whisper

# Sample (outdated)

Sample with an `Macbook Pro (M1)`

https://github.com/gaborvecsei/whisper-live-transcription/assets/18753533/3a4667ce-9af2-4dfe-aa68-8c9ad6307e74

(_🔈 sound on_, `faster-whisper` package, `base` model - latency was around 0.5s)

# Setup

- `$ pip install -r requirements.txt`
- `$ mkdir models`

# Run

- `$ python server.py`
- `$ python client.py`

There are a few parameters at each script that you can modify

# How it works?

I this beautiful art will explain it:

```
- step = 1
- length = 4

$t$ is the current tie (1 second of audio to be precise)

------------------------------------------
1st second: [t,   0,   0,   0] --> "Hi"
2nd second: [t-1, t,   0,   0] --> "Hi I am"
3rd second: [t-2, t-1, t,   0] --> "Hi I am the one"
4th second: [t-3, t-2, t-1, t] --> "Hi I am the one and only Gabor"
5th second: [t,   0,   0,   0] --> "How" --> Here we started the process again, and the output is in a new line
6th second: [t-1, t,   0,   0] --> "How are"
etc...
------------------------------------------

```

# Improvements

- Use a [`VAD`](https://github.com/snakers4/silero-vad) on the client side, and either send the audio for transcription when we detect a longer silence (e.g. 1sec) or if there is no silence we can fall back to the maximum length.
- Transcribe shorter timeframes to get more instant transcriptions and meanwhile, we can use larger timeframes to "correct" already transcribed parts (async correction)
