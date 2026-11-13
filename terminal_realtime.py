"""Live Khmer speech-to-text in the terminal with faster-whisper. Say "ឈប់" to stop.

    uv run --extra realtime terminal_realtime.py
"""

import io
import logging
import queue
import threading
import time

import pyaudio
import speech_recognition as sr
from faster_whisper import WhisperModel

STOP_WORD = "ឈប់"
_STOP = object()  # queue sentinel that ends the worker thread


class STT:
    """Records the default microphone in the background and transcribes each phrase.

    Call `listen()` to start, poll `get_last_transcription()` for new text, and
    `stop()` to end. Transcription runs on its own daemon thread.
    """

    def __init__(self, model_size: str = "medium.en", device: str = "cuda", compute_type: str = "float16",
                 language: str = "en", logging_level: str = None):
        if logging_level:
            logging.basicConfig(level=logging_level.upper())

        self.recorder = sr.Recognizer()
        self.data_queue = queue.Queue()
        self.transcription = []
        self.last_transcription = ""
        self.is_listening = True
        self.language = language
        self.lock = threading.Lock()
        self._stop_recording = None
        self.mic_index = find_input_device()

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            local_files_only=True,
            num_workers=2,
            cpu_threads=4,
        )

        self.thread = threading.Thread(target=self._transcribe_forever, daemon=True)
        self.thread.start()
        logging.info("Ready!")
        print("Ready!\n")

    def listen(self):
        """Start recording the microphone in the background."""
        with sr.Microphone(device_index=self.mic_index) as source:
            self.recorder.adjust_for_ambient_noise(source)
        self._stop_recording = self.recorder.listen_in_background(source, self._on_phrase)

    def stop(self):
        """Stop recording and end the transcription thread. Safe to call more than once."""
        if not self.is_listening:
            return
        logging.info("Stopping... Transcription:\n %s", self.transcription)
        self.is_listening = False
        if self._stop_recording:
            self._stop_recording(wait_for_stop=False)
        self.data_queue.put(_STOP)

    def get_last_transcription(self) -> str:
        """Text recognized since the previous call, or "" if nothing new."""
        with self.lock:
            text, self.last_transcription = self.last_transcription, ""
        return text

    def _on_phrase(self, _, audio_data):
        self.data_queue.put(io.BytesIO(audio_data.get_wav_data()))

    def _transcribe_forever(self):
        while (audio := self.data_queue.get()) is not _STOP:
            segments, _ = self.model.transcribe(
                audio, beam_size=5, language=self.language, vad_filter=False, multilingual=True)
            for segment in segments:
                text = segment.text.strip()
                logging.info("[%.2fs -> %.2fs] %s", segment.start, segment.end, text)
                with self.lock:
                    self.transcription.append(text)
                    self.last_transcription = text


def find_input_device() -> int:
    """Index of the default microphone, or of the first input device if there is no default."""
    audio = pyaudio.PyAudio()
    try:
        return audio.get_default_input_device_info()["index"]
    except (IOError, OSError):
        logging.error("Default input device not found. Input devices:")
        inputs = [i for i in range(audio.get_device_count())
                  if audio.get_device_info_by_index(i)['maxInputChannels'] > 0]
        for i in inputs:
            logging.info("Device index: %d, Device name: %s", i, audio.get_device_info_by_index(i)['name'])
        if not inputs:
            raise RuntimeError("No input devices found.")
        return inputs[0]
    finally:
        audio.terminate()


def main():
    stt = STT(model_size='whisper-tiny-khmer-ct2', device="cpu", compute_type='int8', language='km')
    stt.listen()
    try:
        while stt.is_listening:
            text = stt.get_last_transcription()
            if text:
                print("អ្នកនិយាយ :", text)
                if STOP_WORD in text:
                    stt.stop()
            time.sleep(1)
    except KeyboardInterrupt:
        stt.stop()


if __name__ == "__main__":
    main()
