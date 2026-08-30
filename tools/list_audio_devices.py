try:
    import pyaudiowpatch as pyaudio
except ImportError as exc:
    raise SystemExit("Install PyAudioWPatch on Windows: pip install -r requirements.txt") from exc

audio = pyaudio.PyAudio()
try:
    loopback = {int(item["index"]) for item in audio.get_loopback_device_info_generator()}
    for index in range(audio.get_device_count()):
        item = audio.get_device_info_by_index(index)
        print(f"index={index:<3} loopback={index in loopback!s:<5} rate={item['defaultSampleRate']:<8.0f} "
              f"channels={int(item['maxInputChannels']):<2} name={item['name']}")
finally:
    audio.terminate()
