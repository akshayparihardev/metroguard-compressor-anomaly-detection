"""File names of the artifacts the pipeline steps pass to each other.

This is the single source of truth: a step that writes a file and a step that reads it both
import the name from here. Directories come from `config.yaml`; only the names live in this module.
"""

# data/processed: windowed inputs built in step 2
WINDOW_FEATURES_FILE = "window_features.parquet"
STANDARDISED_FEATURES_FILE = "window_standardised_features.parquet"
DISCRETE_STATES_FILE = "window_discrete_states.parquet"
RBM_VISIBLE_FILE = "window_rbm_visible.npz"

# outputs/models: fitted objects
STANDARDISER_FILE = "feature_standardiser.json"
BIN_EDGES_FILE = "discretisation_bin_edges.json"
MOTOR_CUT_POINTS_FILE = "motor_current_state_cut_points.json"
ZSCORE_REFERENCE_FILE = "zscore_reference.json"
MARKOV_MATRIX_FILE = "markov_transition_matrix.json"
VAE_WEIGHTS_FILE = "vae.pt"

# outputs/scores: per-window anomaly scores, one file per model family
ZSCORE_SCORES_FILE = "zscore_baseline_scores.parquet"
RBM_SCORES_FILE = "rbm_scores.parquet"
VAE_SCORES_FILE = "vae_scores.parquet"
MARKOV_SCORES_FILE = "markov_scores.parquet"
SCORE_FILES = (ZSCORE_SCORES_FILE, RBM_SCORES_FILE, VAE_SCORES_FILE, MARKOV_SCORES_FILE)

# reports/tables: human-readable results
THRESHOLDS_FILE = "thresholds.json"
METRICS_CSV_FILE = "metrics.csv"
METRICS_MD_FILE = "metrics.md"
RBM_HISTORY_FILE = "rbm_training_history.csv"
VAE_HISTORY_FILE = "vae_training_history.csv"
NETWORK_EDGES_FILE = "bayesian_network_edges.csv"
NETWORK_DEMONSTRATIONS_FILE = "bayesian_network_demonstrations.md"
NETWORK_ALERT_CONTEXT_FILE = "bayesian_network_alert_context.md"


def rbm_weights_file(k: int) -> str:
    """Name of the trained RBM weights for a given number of Gibbs steps."""
    return f"rbm_k{k}.pt"
