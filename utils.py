import simpleaudio as sa
def playback(path):
    wave_obj = sa.WaveObject.from_wave_file(path)
    play_obj = wave_obj.play()
    play_obj.wait_done()