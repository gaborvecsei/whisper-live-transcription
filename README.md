# Live Transcription with Whisper

# Sample (outdated)

Sample with an `Macbook Pro (M1)`

https://github.com/gaborvecsei/whisper-live-transcription/assets/18753533/6364a160-4043-437d-be36-52e84a91fe60

(_🔈 sound on_, [audio source](https://www.youtube.com/watch?v=-WSrY-xH5pI), `whisper-cpp-python` package, `tiny` model)

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
