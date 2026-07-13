import numpy as np
from sklearn.metrics import roc_auc_score


def SIM(prediction, target, eps=1e-12):
    prediction = prediction / (prediction.sum() + eps)
    target = target / (target.sum() + eps)
    return np.minimum(prediction, target).sum()


def compute_metrics(predictions, targets):
    """Compute the four metrics reported by MIFAG."""
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    mae = np.mean(np.abs(predictions - targets))
    sim = np.mean(
        [SIM(predictions[index], targets[index]) for index in range(len(targets))]
    )

    binary_targets = (targets >= 0.5).astype(np.int32)
    thresholds = np.linspace(0, 1, 20)
    auc_values = []
    aiou_values = []
    for prediction, target in zip(predictions, binary_targets):
        if target.sum() == 0:
            continue
        auc_values.append(roc_auc_score(target, prediction))
        sample_ious = []
        for threshold in thresholds:
            prediction_mask = (prediction >= threshold).astype(np.int32)
            intersection = np.sum(prediction_mask & target)
            union = np.sum(prediction_mask | target)
            sample_ious.append(intersection / union)
        aiou_values.append(np.mean(sample_ious))

    return {
        "AUC": float(np.mean(auc_values)),
        "aIOU": float(np.mean(aiou_values)),
        "SIM": float(sim),
        "MAE": float(mae),
    }
