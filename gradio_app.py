"""Web demo: transcribe Khmer speech from the microphone or an uploaded file.

    uv run --extra demo gradio_app.py

(Named gradio_app.py, not gradio.py: a file called gradio.py makes
`import gradio` import itself instead of the real package.)
"""

import gradio as gr
import torch
from transformers import pipeline

MODEL_ID = 'PhanithLIM/whisper-small-aug-28-april-lightning-v1'


def main():
    asr = pipeline(
        task='automatic-speech-recognition',
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        generate_kwargs={"language": "Khmer", "task": "transcribe", "num_beams": 5},
        return_timestamps=True,
    )

    gr.Interface(
        fn=lambda audio_path: asr(audio_path)['text'],
        inputs=gr.Audio(sources=['microphone', 'upload'], type='filepath'),
        outputs='text',
        title="🎙️ Khmer Speech-to-Text Demo",
        description=f"Transcribe spoken Khmer using the fine-tuned Whisper model ({MODEL_ID}). "
                    "Supports microphone and audio file input.",
    ).launch(share=True)


if __name__ == "__main__":
    main()
