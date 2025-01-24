'''Implementations of preparing a single channel speech separation dataset base on VCTK corpus.

Copyright (c) Kanami Imamura
All rights reserved.
'''

import os
import argparse
from pathlib import Path

import numpy as np
import pandas
import tqdm

import librosa
import soundfile as sf

SEED = 123
threshold = 20

def preprocess(audio):
    # cutting silent section
    silences = librosa.effects.split(audio, top_db=threshold)
    cutted_audio = audio[silences[0][0]:]
    cutted_audio = cutted_audio / max(cutted_audio) * 0.8
    return cutted_audio

def _pad_audio(audio, length):
    if audio.shape[0] == length:
        return audio
    elif audio.shape[0] > length:
        raise ValueError
    elif audio.shape[0] < length:
        paddings = np.zeros(length - audio.shape[0])
        return np.concatenate((audio, paddings))

def mixture_and_source(sep: str, mix_type: str, vctk_path: Path, metadata_path: Path, outdir_path: Path, sample_rate: int):
    outdir_path = outdir_path / mix_type / sep
    metadata_file = metadata_path / f"{sep}_mixture.txt"

    os.makedirs(outdir_path / "mix", exist_ok=True)
    os.makedirs(outdir_path / "s1", exist_ok=True)
    os.makedirs(outdir_path / "s2", exist_ok=True)
    with open(metadata_file, "r") as f:
        for line in f:
            line = line.rstrip()
            tmp = line.split(",")
            speaker0, speaker1, utterance0, utterance1, snr = tmp
            snr = float(snr)

            # utterance
            utterance0_path = vctk_path / speaker0 / utterance0
            utterance1_path = vctk_path / speaker1 / utterance1
            utterance0_audio, sr = sf.read(utterance0_path)
            utterance1_audio, sr = sf.read(utterance1_path)

            # Resampling
            utterance0_audio = librosa.core.resample(utterance0_audio, orig_sr=sr, target_sr=sample_rate)
            utterance1_audio = librosa.core.resample(utterance1_audio, orig_sr=sr, target_sr=sample_rate)
            
            # preprocessing
            utterance0_audio = preprocess(utterance0_audio)
            utterance1_audio = preprocess(utterance1_audio)

            # mix
            if mix_type == "min":
                audio_length = np.min([utterance0_audio.shape[0], utterance1_audio.shape[0]])
                mixture_audio = (utterance0_audio[:audio_length] + utterance1_audio[:audio_length] * (10 ** (snr/20)))
            elif mix_type == "max":
                audio_length = np.max([utterance0_audio.shape[0], utterance1_audio.shape[0]])
                mixture_audio = (_pad_audio(utterance0_audio, audio_length) + _pad_audio(utterance1_audio, audio_length) * (10 ** (snr/20)))
            mixture_audio = mixture_audio / np.max(mixture_audio) * 0.8

            # save the mixed audio
            mixture_file = utterance0_path.stem + "_" + utterance1_path.name
            mixture_file_path = outdir_path / "mix" / mixture_file
            speaker0_file_path = outdir_path / "s1" / mixture_file
            speaker1_file_path = outdir_path / "s2" / mixture_file
            sf.write(mixture_file_path, mixture_audio, sample_rate, format="WAV", subtype='PCM_16')
            sf.write(speaker0_file_path, utterance0_audio, sample_rate, format="WAV", subtype='PCM_16')
            sf.write(speaker1_file_path, utterance1_audio, sample_rate, format="WAV", subtype='PCM_16')

def process(mix_type: str, vctk_path: Path, metadata_path: Path, outdir_path: Path, sample_rate: int, sep: str):
    if sep == "train":
        sep_list = ["tr", "cv"]
    elif sep == "test":
        sep_list = ["tt"]
    for sep in sep_list:
        mixture_and_source(sep=sep, mix_type=mix_type, vctk_path=vctk_path, metadata_path=metadata_path, outdir_path=outdir_path, sample_rate=sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_vctk_path", required=True, type=str, help="Path to the vctk dataset.")
    parser.add_argument("--metadata_path", required=True, type=str, help="Path to the directory containing metadata files.")
    parser.add_argument("--outdir_path", required=True, type=str, help="Example: local/vctk_32k_2mix/2speakers/wav32k")
    parser.add_argument("--mix_type", type=str, default="min")
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--sep", type=str, choices=["train", "test"])
    args = parser.parse_args()  

    vctk_path = Path(args.original_vctk_path)
    metadata_path = Path(args.metadata_path)
    outdir_path = Path(args.outdir_path)
    
    process(mix_type=args.mix_type, vctk_path=vctk_path, metadata_path=metadata_path, outdir_path=outdir_path, sample_rate=args.sample_rate)