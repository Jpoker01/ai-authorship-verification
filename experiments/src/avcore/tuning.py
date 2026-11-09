"""Seeded, resumable Optuna studies shared by all notebooks.

Every study lives in its own SQLite file, runs one trial at a time and stops at
N_TRIALS finished (complete + pruned) trials. A study killed in the middle of a
trial resumes with the same suggestions as an uninterrupted run.
"""

from optuna.trial import TrialState

N_TRIALS = 30
FINISHED_STATES = (TrialState.COMPLETE, TrialState.PRUNED)
NOT_FAILED_STATES = tuple(state for state in TrialState if state != TrialState.FAIL)
