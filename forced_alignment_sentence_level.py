"""Cut a long Khmer recording into one clip per transcript line (forced alignment).

Each non-blank line of the transcript is romanized, aligned against the audio
with torchaudio's MMS aligner, and written out as `<name>_<i>.wav` next to a
`<name>_<i>.txt` holding the original Khmer line.

    uv run --extra align forced_alignment_sentence_level.py data/audio.wav data/transcript.txt data/output
"""

import argparse
import os
from functools import cache
from typing import List

import librosa
import numpy as np
import soundfile as sf
import torch
from torchaudio.pipelines import MMS_FA as bundle

from khmer_asr.text import remove_zero_width, romanize


def align_audio_to_text(audio_path: str, transcript_path: str, output_dir: str) -> None:
    """Write one (wav, txt) pair to `output_dir` for every non-blank transcript line."""
    print(f"Processing: {audio_path} with transcript {transcript_path}")

    waveform = torch.tensor(librosa.load(audio_path, sr=bundle.sample_rate)[0]).unsqueeze(0)
    khmer_lines = read_transcript(transcript_path)
    num_frames, line_spans = _align(waveform, [romanize(line) for line in khmer_lines])

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    for i, (spans, line) in enumerate(zip(line_spans, khmer_lines)):
        out_path = os.path.join(output_dir, f"{base_name}_{i}")
        _save_segment(f"{out_path}.wav", waveform, spans, num_frames)
        with open(f"{out_path}.txt", 'w', encoding='utf-8') as f:
            f.write(line)

    print(f"✅ Finished alignment for {audio_path}")


def read_transcript(path: str) -> List[str]:
    """Non-blank lines of the transcript, stripped and without zero-width spaces."""
    with open(path, encoding='utf-8') as f:
        lines = (remove_zero_width(line.strip()) for line in f)
        return [line for line in lines if line]


@cache
def _mms():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device, bundle.get_model().to(device), bundle.get_tokenizer(), bundle.get_aligner()


def _align(waveform: torch.Tensor, romanized_lines: List[str]):
    """Return the number of emission frames and, per line, its aligned token spans."""
    device, model, tokenizer, aligner = _mms()
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        line_spans = aligner(emission[0], tokenizer(romanized_lines))
    return emission.size(1), line_spans


def _save_segment(path: str, waveform: torch.Tensor, spans, num_frames: int) -> None:
    """Write the audio between the first and last aligned token of one line."""
    samples_per_frame = waveform.size(1) / num_frames
    start = int(samples_per_frame * spans[0].start)
    end = int(samples_per_frame * spans[-1].end)
    sf.write(path, np.ravel(waveform[:, start:end].numpy()), bundle.sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", nargs="?", default="data/audio.wav")
    parser.add_argument("transcript", nargs="?", default="data/transcript.txt")
    parser.add_argument("output_dir", nargs="?", default="data/output")
    args = parser.parse_args()
    align_audio_to_text(args.audio, args.transcript, args.output_dir)
