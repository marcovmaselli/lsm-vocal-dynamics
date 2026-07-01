"""Build the windowed source/filter feature dataset consumed by ``pretrain_lsm.py``.

For every subject's audio recording, the script slides a 2 s window (0.5 s hop)
over the signal, applies LPC-based source-filter decomposition and gammatone/ERB
encoding (Section IV.A of the paper), and stores the resulting tensors together
with a CSV log.

Speech-only filtering (ELAN third-party annotations + Silero VAD) matches the
inference/feature-extraction split described in Section IV; pass --all-windows
to instead keep every window unfiltered, as used for the unsupervised reservoir
pretraining split.

Raw audio and subject-level labels are not distributed with this repository
(clinical/privacy-restricted data, see the top-level README). To run this
script on your own cohort, provide:
  - one or more directories of ``<subject_id>.wav`` recordings (--audio-dirs)
  - a CSV with columns ``subject_id,label`` (--labels-csv)
  - for speech-only mode, ELAN ``.eaf`` files in a sibling ``Label/`` directory
    next to each audio directory, named ``<subject_id>.eaf``
"""
import argparse
import csv
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import (
    extract_windows_with_overlap,
    find_non_silent_vad_silero,
    get_annotations_from_file,
    linear_predictive_analysis,
    load_audio_torch,
    lowpass_filter_torch,
    lpc_filters_to_erb,
    remove_other_people_from_segments,
    residuals_to_gammatone_energy,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-dirs", nargs="+", required=True,
                    help="Directories of subject .wav recordings.")
    p.add_argument("--labels-csv", required=True,
                    help="CSV with columns subject_id,label (0=control, 1=clinical group).")
    p.add_argument("--output-dir", required=True,
                    help="Where to store the extracted tensors and dataset_log.csv.")
    p.add_argument("--all-windows", action="store_true",
                    help="Skip ELAN/VAD speech filtering and keep every window "
                         "(unsupervised reservoir pretraining split).")
    p.add_argument("--f-min", type=float, default=80.0)
    p.add_argument("--f-max", type=float, default=4000.0)
    p.add_argument("--n-channels", type=int, default=56)
    return p.parse_args()


def main():
    """Window every listed subject's audio, encode each window, and append to ``dataset_log.csv``.

    For each ``.wav`` file under ``--audio-dirs`` with a matching row in
    ``--labels-csv``, this: loads and lowpass-filters the audio; finds the
    windows to keep (all of them, or only VAD/ELAN speech-only windows);
    slices 2 s/0.5 s-hop windows over those regions; runs LPC source-filter
    decomposition + gammatone/ERB encoding on each window; and writes the
    resulting ``.pt`` tensors plus one manifest row per window. Already-encoded
    windows (matching output filenames) are skipped, so reruns are resumable.
    """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    labels = pd.read_csv(args.labels_csv).set_index("subject_id")["label"].to_dict()

    csv_path = os.path.join(args.output_dir, "dataset_log.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=[
            "subject_id", "data_path", "window_meta", "label",
            "tensor_gammatone_path", "tensor_erb_path",
        ])
        if write_header:
            writer.writeheader()

        for audio_dir in args.audio_dirs:
            subjects = [f.split(".")[0] for f in os.listdir(audio_dir) if f.endswith(".wav")]
            for subject in tqdm(subjects, desc=os.path.basename(audio_dir)):
                if subject not in labels:
                    continue
                audio_file = os.path.join(audio_dir, f"{subject}.wav")
                annotation_file = os.path.join(os.path.dirname(audio_dir), "Label", f"{subject}.eaf")
                try:
                    signal, duration = load_audio_torch(audio_file)
                    signal_filtered = lowpass_filter_torch(signal)
                    if args.all_windows:
                        vad_times = [[0.0, float(duration)]]
                        annotation_times = []
                    else:
                        annotation_times, _ = get_annotations_from_file(annotation_file)
                        vad_times, _ = find_non_silent_vad_silero(signal_filtered)
                    _, remaining_idxs = remove_other_people_from_segments(vad_times, annotation_times)
                    windows, windows_meta = extract_windows_with_overlap(signal, remaining_idxs)
                except Exception as exc:
                    print(f"[skip] subject {subject}: {exc}")
                    continue

                for w_idx, window in enumerate(windows, 1):
                    gam_path = os.path.join(args.output_dir, f"{subject}_window{w_idx}_gammatone.pt")
                    erb_path = os.path.join(args.output_dir, f"{subject}_window{w_idx}_erb.pt")
                    if os.path.exists(gam_path) and os.path.exists(erb_path):
                        continue
                    try:
                        window = window / torch.max(torch.abs(window))
                        lpc_coeffs, residual, _ = linear_predictive_analysis(window)
                        gammatone_source = residuals_to_gammatone_energy(
                            residual, n_filters=args.n_channels, f_max=args.f_max, f_min=args.f_min)
                        erb_filter = lpc_filters_to_erb(
                            lpc_coeffs, n_filters=args.n_channels, f_max=args.f_max, f_min=args.f_min)
                    except Exception:
                        continue

                    torch.save(gammatone_source, gam_path)
                    torch.save(erb_filter, erb_path)
                    writer.writerow({
                        "subject_id": subject,
                        "data_path": audio_dir,
                        "window_meta": windows_meta[w_idx - 1],
                        "label": labels[subject],
                        "tensor_gammatone_path": gam_path,
                        "tensor_erb_path": erb_path,
                    })


if __name__ == "__main__":
    main()
