import speech_recognition as sr

r = sr.Recognizer()
trial = sr.AudioFile("trial.wav")

with trial as source:
    r.adjust_for_ambient_noise(source, 1)
    audio = r.record(source)

r.recognize_google(audio)
