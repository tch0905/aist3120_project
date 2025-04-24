import numpy as np
from seqeval.metrics import classification_report
from dataset import label_names

def compute_metrics(p):
    predictions, labels = p.predictions, p.label_ids


    # Handle CRF case where predictions might come as lists of lists
    if isinstance(predictions, list):
        # CRF returns tags directly
        true_predictions = predictions
    else:
        # Standard argmax for non-CRF
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

    true_labels = [
        [label_names[l] for l in label if l != -100]
        for label in labels
    ]

    # Get full classification report with float precision
    report = classification_report(
        true_labels, true_predictions, output_dict=True, digits=6
    )  # Increased decimal places

    # Calculate exact match accuracy with high precision
    correct = sum(
        1 for true, pred in zip(true_labels, true_predictions) if true == pred
    )
    total = len(true_labels)
    accuracy = correct / total if total > 0 else 0.0

    # Format all metrics to 6 decimal places
    metrics = {
        # Overall scores
        "accuracy": round(accuracy, 6),
        "overall_precision": round(report["micro avg"]["precision"], 6),
        "overall_recall": round(report["micro avg"]["recall"], 6),
        "overall_f1": round(report["micro avg"]["f1-score"], 6),
        # Macro averages
        "macro_precision": round(report["macro avg"]["precision"], 6),
        "macro_recall": round(report["macro avg"]["recall"], 6),
        "macro_f1": round(report["macro avg"]["f1-score"], 6),
    }

    # Add per-class F1 scores with high precision
    for label_name in label_names:
        if (
                label_name in report
                and label_name != "macro avg"
                and label_name != "micro avg"
        ):
            metrics.update(
                {
                    f"{label_name}_precision": round(
                        report[label_name]["precision"], 6
                    ),
                    f"{label_name}_recall": round(report[label_name]["recall"], 6),
                    f"{label_name}_f1": round(report[label_name]["f1-score"], 6),
                }
            )

    # Print beautifully formatted report
    print("\nDetailed Classification Report (6 decimal places):")
    print(classification_report(true_labels, true_predictions, digits=6))

    return metrics