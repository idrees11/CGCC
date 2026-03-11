import os
from datetime import datetime
from pathlib import Path
from .hidden_labels_reader import read_hidden_labels
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

SUBMISSIONS_DIR = Path(__file__).resolve().parent.parent / "submissions"


def _team_name_from_path(path: Path) -> str:
    return path.stem.replace("_ideal", "").replace("_perturbed", "")


def calculate_scores(submission_path: Path):
    """
    Score a single submission file (ideal OR perturbed)
    """

    submission_path = Path(submission_path).resolve()

    labels_df = read_hidden_labels()
    if labels_df is None:
        raise FileNotFoundError("Hidden test labels not found.")

    submission_df = pd.read_csv(submission_path)

    prediction_col = "prediction" if "prediction" in submission_df.columns else "target"

    merged = labels_df.merge(
        submission_df[["filename", prediction_col]],
        on="filename",
        how="inner"
    )

    y_true = merged["target"]
    y_pred = merged[prediction_col]

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return accuracy, f1


def get_leaderboard_data():
    """
    Look for pairs:
        ideal_submission.csv
        perturbed_submission.csv
    """

    files = os.listdir(SUBMISSIONS_DIR)

    ideal_files = [f for f in files if "ideal" in f and f.endswith(".csv")]
    perturbed_files = [f for f in files if "perturbed" in f and f.endswith(".csv")]

    scores = []

    for ideal_file in ideal_files:

        team_name = ideal_file.replace("_ideal_submission.csv", "")

        perturbed_file = f"{team_name}_perturbed_submission.csv"

        ideal_path = SUBMISSIONS_DIR / ideal_file
        perturbed_path = SUBMISSIONS_DIR / perturbed_file

        if not perturbed_path.exists():
            print(f"Skipping {team_name}: perturbed file missing")
            continue

        try:

            acc_i, f1_i = calculate_scores(ideal_path)
            acc_p, f1_p = calculate_scores(perturbed_path)

            robustness_gap = abs(f1_i - f1_p)

            timestamp = datetime.fromtimestamp(
                ideal_path.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M:%S")

            scores.append({
                "team_name": team_name,
                "validation_f1_ideal": float(f1_i),
                "validation_f1_perturbed": float(f1_p),
                "robustness_gap": float(robustness_gap),
                "validation_accuracy_ideal": float(acc_i),
                "validation_accuracy_perturbed": float(acc_p),
                "timestamp": timestamp
            })

        except Exception as e:
            print(f"Skipping invalid submission {team_name}: {e}")
            continue

    # Ranking rule
    scores.sort(
        key=lambda x: (
            -x["validation_f1_perturbed"],   # highest perturbed first
            x["robustness_gap"]              # smallest gap next
        )
    )

    return scores
