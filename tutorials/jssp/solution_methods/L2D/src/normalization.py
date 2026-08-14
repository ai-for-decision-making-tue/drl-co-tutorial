"""DANIEL-style reward and operation-feature normalization for L2D."""

import numpy as np


NORMALIZATION_EPSILON = 1e-8


def build_operation_features(
        lower_bounds,
        finished_mark,
        normalize_reward_and_state,
        end_time_normalize_coefficient):
    """Build L2D's node features, optionally standardizing each feature.

    DANIEL standardizes every raw operation feature over the operation-node
    dimension whenever it constructs a state. L2D has two operation features:
    the completion-time lower bound and the finished flag. Applying the same
    rule here keeps the feature scale instance- and problem-size-independent.
    Constant features become zero because of the epsilon in the denominator.
    """
    features = np.concatenate(
        (
            np.asarray(lower_bounds, dtype=np.single).reshape(-1, 1),
            np.asarray(finished_mark, dtype=np.single).reshape(-1, 1),
        ),
        axis=1,
    )

    if normalize_reward_and_state:
        feature_mean = features.mean(axis=0, keepdims=True)
        feature_std = features.std(axis=0, keepdims=True)
        features = (features - feature_mean) / (
            feature_std + NORMALIZATION_EPSILON
        )
    else:
        features[:, 0] /= end_time_normalize_coefficient

    return features.astype(np.single, copy=False)


def get_reward_normalization_factor(processing_times, normalize_reward_and_state):
    """Return the processing-time scale used by DANIEL-style rewards.

    DANIEL normalizes the JSSP processing-time tensor using its maximum (the
    incompatible machine entries make the minimum zero). Since L2D stores only
    the compatible processing time of each operation, dividing its shaped
    rewards by the instance maximum is the equivalent transformation without
    changing the schedule calculations themselves.
    """
    if not normalize_reward_and_state:
        return 1.0

    processing_times = np.asarray(processing_times)
    if processing_times.size == 0 or not np.all(np.isfinite(processing_times)):
        raise ValueError("processing times must be a non-empty finite array")

    maximum_processing_time = float(np.max(processing_times))
    if maximum_processing_time <= 0:
        raise ValueError("processing times must be strictly positive")

    return maximum_processing_time
