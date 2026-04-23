from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)

from utils.modeling import (
    build_cnn,
    build_resnet,
    build_efficientnet,
    build_lstm,
    build_discriminator,
    build_vit,
    build_xception,
    FocalLoss,
)

# All model names used across the system
MODEL_NAMES = ["xception", "cnn", "resnet", "efficientnet", "lstm", "discriminator", "vit"]


def _make_ds(data_dir: str, image_size=(224, 224), batch_size=32, shuffle=True):
    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode="binary",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _normalize(ds):
    return ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)


_AUGMENTATION = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.GaussianNoise(0.02),
])


def _augment(ds):
    return ds.map(
        lambda x, y: (_AUGMENTATION(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


# ---------------------------------------------------------------------------
# Temperature Scaling (confidence calibration)
# ---------------------------------------------------------------------------
def _calibrate_temperature(model, val_ds, save_path: Path) -> float:
    """Learn optimal temperature T for confidence calibration on validation set.
    
    Temperature scaling is a post-hoc calibration method:
      calibrated = sigmoid(logit / T)
    A single scalar T is learned to minimize NLL on the validation set.
    """
    # Collect logits and labels
    logits_list = []
    labels_list = []
    for x_batch, y_batch in val_ds:
        pred = model.predict(x_batch, verbose=0).reshape(-1)
        # Convert sigmoid output back to logit
        pred_clipped = np.clip(pred, 1e-7, 1 - 1e-7)
        logit = np.log(pred_clipped / (1 - pred_clipped))
        logits_list.extend(logit.tolist())
        labels_list.extend(y_batch.numpy().reshape(-1).tolist())

    logits = np.array(logits_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)

    # Grid search for best temperature
    best_t = 1.0
    best_nll = float("inf")
    for t in np.arange(0.1, 5.0, 0.05):
        scaled = 1.0 / (1.0 + np.exp(-logits / t))
        scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
        nll = -np.mean(labels * np.log(scaled) + (1 - labels) * np.log(1 - scaled))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    # Save temperature
    with save_path.open("w") as f:
        json.dump({"temperature": best_t, "nll": best_nll}, f, indent=2)
    
    print(f"  Calibration temperature: {best_t:.2f} (NLL: {best_nll:.4f})")
    return best_t


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    processed_root: str = "data/processed",
    model_path: str = "models/deepfake_cnn_model.h5",
    epochs: int = 10,
    batch_size: int = 32,
) -> Dict[str, float]:
    train_dir = Path(processed_root) / "train"
    val_dir = Path(processed_root) / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Processed dataset missing. Run preprocessing first.")

    train_ds = _augment(_normalize(_make_ds(str(train_dir), batch_size=batch_size, shuffle=True)))
    val_ds = _normalize(_make_ds(str(val_dir), batch_size=batch_size, shuffle=False))

    model_builders = {
        "xception": build_xception,
        "cnn": build_cnn,
        "resnet": build_resnet,
        "efficientnet": build_efficientnet,
        "lstm": build_lstm,
        "discriminator": build_discriminator,
        "vit": build_vit,
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(model_path).parent
    models_dir.mkdir(parents=True, exist_ok=True)

    final_metrics = {}

    for name, builder in model_builders.items():
        print(f"\n{'='*60}")
        print(f"Training {name} model...")
        print(f"{'='*60}")
        model = builder()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=3, factor=0.5, min_lr=1e-7
            ),
        ]
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks
        )

        out_path = models_dir / f"deepfake_{name}_model.h5"
        model.save(out_path)
        print(f"  Saved: {out_path}")

        # Save training history
        history_path = reports_dir / f"training_history_{name}.json"
        serializable_history = {}
        for k, v in history.history.items():
            serializable_history[k] = [float(val) for val in v]
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(serializable_history, f, indent=2)

        _plot_history(history.history, reports_dir / f"accuracy_loss_{name}.png")

        # Calibrate temperature for this model
        print(f"  Calibrating temperature for {name}...")
        temp_path = models_dir / f"temperature_{name}.json"
        _calibrate_temperature(model, val_ds, temp_path)

        val_acc = float(history.history["val_accuracy"][-1])
        final_metrics[f"{name}_val_accuracy"] = val_acc
        print(f"  {name} val_accuracy: {val_acc:.4f}")

    # Save ensemble weights (based on validation accuracy)
    _save_ensemble_weights(final_metrics, models_dir / "ensemble_weights.json")

    return final_metrics


def _save_ensemble_weights(metrics: Dict[str, float], save_path: Path) -> None:
    """Compute ensemble weights proportional to validation accuracy."""
    accs = {}
    for key, val in metrics.items():
        name = key.replace("_val_accuracy", "")
        accs[name] = val

    total = sum(accs.values())
    weights = {name: round(acc / total, 4) for name, acc in accs.items()}

    with save_path.open("w") as f:
        json.dump({"weights": weights, "raw_accuracies": accs}, f, indent=2)
    print(f"\nEnsemble weights: {weights}")


# ---------------------------------------------------------------------------
# Evaluation (with AUC-ROC, F1, Precision-Recall)
# ---------------------------------------------------------------------------
def evaluate_model(
    processed_root: str = "data/processed",
    model_path: str = "models/deepfake_cnn_model.h5",
    batch_size: int = 32,
) -> Dict[str, object]:
    test_dir = Path(processed_root) / "test"
    if not test_dir.exists():
        raise FileNotFoundError("Test dataset missing. Build processed dataset first.")

    models_dir = Path(model_path).parent
    loaded_models = []
    model_names_loaded = []
    temperatures = []

    for name in MODEL_NAMES:
        path = models_dir / f"deepfake_{name}_model.h5"
        if path.exists():
            loaded_models.append(tf.keras.models.load_model(
                path, custom_objects={"FocalLoss": FocalLoss}
            ))
            model_names_loaded.append(name)
            # Load temperature
            temp_path = models_dir / f"temperature_{name}.json"
            if temp_path.exists():
                with temp_path.open() as f:
                    temperatures.append(json.load(f).get("temperature", 1.0))
            else:
                temperatures.append(1.0)

    if not loaded_models:
        loaded_models.append(tf.keras.models.load_model(model_path))
        model_names_loaded.append("fallback")
        temperatures.append(1.0)

    # Load ensemble weights
    weights_path = models_dir / "ensemble_weights.json"
    if weights_path.exists():
        with weights_path.open() as f:
            weight_data = json.load(f)
        weights = [weight_data["weights"].get(n, 1.0 / len(loaded_models)) for n in model_names_loaded]
    else:
        weights = [1.0 / len(loaded_models)] * len(loaded_models)

    # Normalize weights
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    test_ds_raw = _make_ds(str(test_dir), batch_size=batch_size, shuffle=False)
    test_ds = _normalize(test_ds_raw)

    y_true = []
    y_score = []
    for x_batch, y_batch in test_ds:
        batch_preds = []
        for i, model in enumerate(loaded_models):
            pred = model.predict(x_batch, verbose=0).reshape(-1)
            # Apply temperature scaling
            pred_clipped = np.clip(pred, 1e-7, 1 - 1e-7)
            logit = np.log(pred_clipped / (1 - pred_clipped))
            calibrated = 1.0 / (1.0 + np.exp(-logit / temperatures[i]))
            batch_preds.append(calibrated * weights[i])
        
        avg_pred = np.sum(batch_preds, axis=0)
        y_score.extend(avg_pred.tolist())
        y_true.extend(y_batch.numpy().reshape(-1).astype(int).tolist())

    y_pred = [1 if p >= 0.5 else 0 for p in y_score]
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred, target_names=["real", "fake"], output_dict=True
    )

    # Advanced metrics
    auc_roc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else 0.0
    f1 = float(f1_score(y_true, y_pred))
    avg_precision = float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else 0.0

    results = {
        "confusion_matrix": cm,
        "classification_report": report,
        "auc_roc": auc_roc,
        "f1_score": f1,
        "average_precision": avg_precision,
        "models_used": model_names_loaded,
        "ensemble_weights": weights,
        "temperatures": temperatures,
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot precision-recall curve
    if len(set(y_true)) > 1:
        _plot_precision_recall(y_true, y_score, reports_dir / "precision_recall.png")

    print(f"\nEvaluation Results:")
    print(f"  AUC-ROC:           {auc_roc:.4f}")
    print(f"  F1-Score:          {f1:.4f}")
    print(f"  Avg Precision:     {avg_precision:.4f}")
    print(f"  Models used:       {model_names_loaded}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_history(history: Dict[str, list], output_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.get("accuracy", []), label="train")
    plt.plot(history.get("val_accuracy", []), label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.get("loss", []), label="train")
    plt.plot(history.get("val_loss", []), label="val")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_precision_recall(y_true, y_score, output_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
