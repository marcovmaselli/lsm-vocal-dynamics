"""Public package exports for the neurodevelopmental audio pipeline.

The package groups audio I/O, preprocessing, gammatone feature extraction,
and the dual Liquid State Machine (LSM) reservoir pipeline used to reproduce
the paper's results.
"""

from .analysis_utils import run_pca_kmeans
from .config import CONFIG
from .features import (
	MinMaxNormalizer,
	compute_mel_spectrogram,
	erb2hz,
	hz2erb,
	lpc_filters_to_erb,
	make_erb_centers,
	residuals_to_gammatone_energy,
)
from .io_utils import (
	get_annotations_from_file,
	load_audio_torch,
	load_subject_features,
	load_subject_windows,
	save_chunks_with_csv_pd,
)
from .logging import Tee, setup_logger
from .lsm import (
	LSM,
	build_3d_positions,
	build_weight_in_layered,
	build_weight_in_layered_one_to_one,
	build_weight_lsm_probabilistic,
	pairwise_squared_distances,
)
from .lsm_pipeline import (
	build_lsm_pair,
	extract_features_and_spikes,
	pretrain_lsm_pair,
	stack_windows_segment,
)
from .preprocessing import (
	extract_windows_with_overlap,
	find_non_silent_vad_silero,
	levinson_durbin,
	linear_predictive_analysis,
	lowpass_filter_torch,
	remove_other_people_from_segments,
)
from .stdp import AsymmetricSTDP

__all__ = [
	"AsymmetricSTDP",
	"CONFIG",
	"LSM",
	"MinMaxNormalizer",
	"Tee",
	"build_3d_positions",
	"build_lsm_pair",
	"build_weight_in_layered",
	"build_weight_in_layered_one_to_one",
	"build_weight_lsm_probabilistic",
	"compute_mel_spectrogram",
	"erb2hz",
	"extract_features_and_spikes",
	"extract_windows_with_overlap",
	"find_non_silent_vad_silero",
	"get_annotations_from_file",
	"hz2erb",
	"linear_predictive_analysis",
	"load_audio_torch",
	"load_subject_features",
	"load_subject_windows",
	"lowpass_filter_torch",
	"lpc_filters_to_erb",
	"levinson_durbin",
	"make_erb_centers",
	"pairwise_squared_distances",
	"pretrain_lsm_pair",
	"remove_other_people_from_segments",
	"residuals_to_gammatone_energy",
	"run_pca_kmeans",
	"save_chunks_with_csv_pd",
	"setup_logger",
	"stack_windows_segment",
]