"""Unsupervised STDP pretraining of the paired source/filter LSM reservoirs,
followed by frozen feature and spike extraction (Section IV.B-C of the paper).

Two dataset variants are used, matching the two-stage strategy described in
the paper: the reservoirs are pretrained on *all* windows (silence, noise,
other speakers included) via --train-data-dir, while feature/spike extraction
for the downstream classifier is restricted to the speech-filtered windows via
--extract-data-dir. Both directories are produced by ``build_dataset.py``
(the latter with --all-windows, the former without).
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lsm_pipeline import build_lsm_pair, extract_features_and_spikes, pretrain_lsm_pair
from src.analysis_utils import run_pca_kmeans
from src.logging import setup_logger


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extract-data-dir", required=True,
                    help="Speech-filtered dataset directory (build_dataset.py without --all-windows).")
    p.add_argument("--train-data-dir", required=True,
                    help="Unfiltered dataset directory (build_dataset.py with --all-windows).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-channels", type=int, default=56)
    p.add_argument("--n-epochs", type=int, default=3,
                    help="STDP pretraining epochs (3 in the paper).")
    p.add_argument("--exclude-subjects", nargs="*", default=[],
                    help="Subject IDs to drop (e.g. insufficient data or excluded diagnosis).")
    p.add_argument("--resume", action="store_true",
                    help="Resume pretraining from checkpoints already present in --output-dir.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    spikes_dir = os.path.join(args.output_dir, "spikes")
    os.makedirs(spikes_dir, exist_ok=True)
    setup_logger(args.output_dir, prefix="pretrain")

    df = pd.read_csv(os.path.join(args.extract_data_dir, "dataset_log.csv"), sep=",", quotechar='"')
    df_train = pd.read_csv(os.path.join(args.train_data_dir, "dataset_log.csv"), sep=",", quotechar='"')
    subjects = np.setdiff1d(df["subject_id"].unique(), args.exclude_subjects)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # STDP hyperparameters from Section IV.B of the paper: tau+ = 20 ms for both
    # reservoirs; tau- = 80/100 ms (source/filter); A+ = 0.012/0.004; A- = 0.015/0.008;
    # Vth = 12 (source) / 17 (filter). Selected empirically, not via grid search
    # (see paper limitations).
    stdp_params_source = dict(tau_plus=20, tau_minus=80, A_plus=0.012, A_minus=0.015)
    stdp_params_filter = dict(tau_plus=20, tau_minus=100, A_plus=0.004, A_minus=0.008)

    lsm_source, lsm_filter = build_lsm_pair(
        stdp_params_source=stdp_params_source,
        stdp_params_filter=stdp_params_filter,
        n_layers=args.n_channels,
        device=device,
        th_source=12,
        th_filter=17,
        inhibit=True,
    )

    source_ckpt = os.path.join(args.output_dir, "lsm_source.pth")
    filter_ckpt = os.path.join(args.output_dir, "lsm_filter.pth")
    if args.resume and os.path.exists(source_ckpt) and os.path.exists(filter_ckpt):
        print(f"Resuming from checkpoints in {args.output_dir}")
        lsm_source.load_state_dict(torch.load(source_ckpt, map_location=device))
        lsm_filter.load_state_dict(torch.load(filter_ckpt, map_location=device))

    for epoch in range(args.n_epochs):
        print(f"--- Pretraining epoch {epoch + 1}/{args.n_epochs} ---")
        pretrain_lsm_pair(
            lsm_source=lsm_source,
            lsm_filter=lsm_filter,
            df=df_train,
            subjects=subjects,
            project_path=args.output_dir,  # unused by pretrain_lsm_pair, kept for API compatibility
            data_path=args.train_data_dir,
            device=device,
            max_windows_per_chunk=1,
            clamp_min=-1.0,
            clamp_max=1.0,
        )
        torch.save(lsm_source.state_dict(), source_ckpt)
        torch.save(lsm_filter.state_dict(), filter_ckpt)
        print(f"[SAVE] checkpoints written to {args.output_dir} (epoch {epoch + 1})")

    # Freeze weights (STDP is disabled internally by extract_features_and_spikes)
    # and extract subject-level spikes/features from the speech-only windows.
    subject_features, subject_labels, subject_spikes_paths = extract_features_and_spikes(
        lsm_source=lsm_source,
        lsm_filter=lsm_filter,
        df=df,
        subjects=subjects,
        project_path=args.output_dir,  # unused by extract_features_and_spikes, kept for API compatibility
        data_path=args.extract_data_dir,
        device=device,
        max_windows_per_chunk=1,
        clamp_min=-1.0,
        clamp_max=1.0,
        save_spikes_dir=spikes_dir,
    )

    _, _, clusters, subjects_list = run_pca_kmeans(subject_features)
    out_pkl = os.path.join(args.output_dir, "subject_features.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "features": subject_features,
            "labels": subject_labels,
            "subjects": subjects_list,
            "clusters": clusters,
            "spike_files": subject_spikes_paths,
        }, f)
    print(f"Saved subject features to {out_pkl}")


if __name__ == "__main__":
    main()
