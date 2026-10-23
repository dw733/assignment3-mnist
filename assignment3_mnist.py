# Assignment 3 MNIST experiments


# install
import os
import sys
import struct
import shutil
from pathlib import Path
import psutil

if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise RuntimeError("Python 3.10–3.12 required")
if struct.calcsize("P") != 8:
    raise RuntimeError("64-bit Python required")

available_gb = psutil.virtual_memory().available / 1024**3
free_disk_gb = shutil.disk_usage(Path.cwd()).free / 1024**3
if available_gb < 3.0:
    raise MemoryError(f"Available RAM: {available_gb:.2f} GB; minimum: 3.0 GB")
if free_disk_gb < 1.0:
    raise OSError(f"Free disk space: {free_disk_gb:.2f} GB; minimum: 1.0 GB")
print("Python executable:", sys.executable)
print("Python version:", sys.version.split()[0])
print("Available RAM (GB):", round(available_gb, 2))
ENVIRONMENT_CHECKED = True
print("Environment checks passed")


# imports
import os 
import sys

if not globals().get("ENVIRONMENT_CHECKED", False):
    raise RuntimeError("Environment preflight has not completed")
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise RuntimeError("This environment requires Python 3.10–3.12.")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MLFLOW_ENABLE_ASYNC_LOGGING"] = "false"

import json
import gc
import hashlib
import psutil
import math
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

SEED = 42
TRACKING_MODE = "dagshub"
REPO_OWNER = "dw733"
REPO_NAME = "assignment3-mnist-mlflow"
EXPERIMENT_NAME = "Assignment 3 MNIST"
DAGSHUB_URL = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}"
DAGSHUB_TRACKING_URI = DAGSHUB_URL + ".mlflow"
OUTPUT_DIR = Path("assignment3_results") / f"windows_cpu_{TRACKING_MODE}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = OUTPUT_DIR / "session_state.json"
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
RESULTS = {}
MODEL_PATHS = {}
HISTORIES = {}

CONFIGS = [
    {"name": "baseline_128_adam_b64_e50", "nodes": [128], "optimizer": "Adam",
     "learning_rate": 0.001, "momentum": 0.0, "batch_size": 64, "epochs": 50},
    {"name": "wider_256_adam_b64_e10", "nodes": [256], "optimizer": "Adam",
     "learning_rate": 0.001, "momentum": 0.0, "batch_size": 64, "epochs": 10},
    {"name": "deeper_128_64_adam_b64_e10", "nodes": [128, 64], "optimizer": "Adam",
     "learning_rate": 0.001, "momentum": 0.0, "batch_size": 64, "epochs": 10},
    {"name": "sgd_128_b64_e10", "nodes": [128], "optimizer": "SGD",
     "learning_rate": 0.01, "momentum": 0.9, "batch_size": 64, "epochs": 10},
]

for getter, setter in [
    (tf.config.threading.get_intra_op_parallelism_threads, tf.config.threading.set_intra_op_parallelism_threads),
    (tf.config.threading.get_inter_op_parallelism_threads, tf.config.threading.set_inter_op_parallelism_threads),
]:
    if getter() != 1:
        setter(1)
if tf.config.list_physical_devices("GPU"):
    tf.config.set_visible_devices([], "GPU")

SESSION_FINGERPRINT = hashlib.sha256(json.dumps({
    "configs": CONFIGS, "seed": SEED, "tracking_mode": TRACKING_MODE,
    "repo_owner": REPO_OWNER, "repo_name": REPO_NAME,
    "tensorflow_version": tf.__version__, "pipeline_version": 2,
}, sort_keys=True).encode("utf-8")).hexdigest()
if STATE_PATH.is_file():
    previous = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if previous["fingerprint"] != SESSION_FINGERPRINT:
        raise RuntimeError("Saved session settings do not match current settings")
    SESSION_ID = previous["session_id"]
    RESULTS = previous["results"]
    for name in RESULTS:
        MODEL_PATHS[name] = OUTPUT_DIR / name / "final_model.keras"
        history_path = OUTPUT_DIR / name / "history.json"
        if not MODEL_PATHS[name].is_file() or not history_path.is_file():
            raise RuntimeError(f"Incomplete saved model: {name}")
        HISTORIES[name] = json.loads(history_path.read_text(encoding="utf-8"))

tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
print("TensorFlow:", tf.__version__)
print("MLflow:", mlflow.__version__)
print("Python:", sys.version.split()[0])
print("Results folder:", OUTPUT_DIR.resolve())
print("CPU execution; TensorFlow thread limits: 1 intra-op, 1 inter-op")
print("Previously completed models:", len(RESULTS))
print("Setup complete")


# connect
if TRACKING_MODE == "dagshub":
    import dagshub
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    TRACKING_URI = DAGSHUB_TRACKING_URI
elif TRACKING_MODE == "local":
    TRACKING_URI = "sqlite:///" + str((OUTPUT_DIR / "mlflow.db").resolve())
else:
    raise ValueError("Tracking mode must be local or dagshub")

mlflow.set_tracking_uri(TRACKING_URI)
tracking_client = MlflowClient()
if TRACKING_MODE == "local" and tracking_client.get_experiment_by_name(EXPERIMENT_NAME) is None:
    tracking_client.create_experiment(
        EXPERIMENT_NAME, artifact_location=(OUTPUT_DIR / "artifacts").resolve().as_uri()
    )
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.tensorflow.autolog(
    log_models=True, log_datasets=False, log_model_signatures=False,
    checkpoint=False, log_every_epoch=True,
)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise RuntimeError("Experiment not found on the tracking server")
print("Tracking mode:", TRACKING_MODE)
print("MLflow Tracking URI:", TRACKING_URI)
print("Experiment:", experiment.name)
print("Configured DagsHub repository:", DAGSHUB_URL)
print("Configured DagsHub Tracking URI:", DAGSHUB_TRACKING_URI)


# data
(images, labels), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
train_indices, val_indices = train_test_split(
    np.arange(len(labels)), test_size=0.20, random_state=SEED, stratify=labels
)
x_train = images[train_indices].astype("float32")
x_train /= np.float32(255.0)
x_val = images[val_indices].astype("float32")
x_val /= np.float32(255.0)
y_train, y_val = labels[train_indices], labels[val_indices]
x_test = x_test.astype("float32")
x_test /= np.float32(255.0)
del images
gc.collect()
np.savez_compressed(OUTPUT_DIR / "split_indices.npz", train=train_indices, validation=val_indices)
print("Training:", x_train.shape, "Validation:", x_val.shape, "Test:", x_test.shape)
print("Pixel range:", float(x_train.min()), "to", float(x_train.max()))


# functions
class MemoryGuard(tf.keras.callbacks.Callback): 
    def on_train_batch_end(self, batch, logs=None):
        if batch % 50 == 0 and psutil.virtual_memory().available < 768 * 1024**2:
            raise MemoryError("Available RAM fell below 768 MB")


def save_session_state():
    state = {"fingerprint": SESSION_FINGERPRINT, "session_id": SESSION_ID, "results": RESULTS}
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(STATE_PATH)


def make_dataset(x, y, batch_size, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        dataset = dataset.shuffle(min(len(y), 4096), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(1)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    options.experimental_deterministic = True
    return dataset.with_options(options)


def build_model(config):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(name="flatten"),
    ])
    for index, nodes in enumerate(config["nodes"], start=1):
        model.add(tf.keras.layers.Dense(nodes, activation="relu", name=f"hidden_{index}"))
    model.add(tf.keras.layers.Dense(10, activation="softmax", name="digit_probabilities"))
    if config["optimizer"] == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    elif config["optimizer"] == "SGD":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=config["learning_rate"], momentum=config["momentum"]
        )
    else:
        raise ValueError("Unsupported optimizer")
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_learning_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["loss"]) + 1)
    for ax, metric, label in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        ax.plot(epochs, history[metric], label="Training", color="gray")
        ax.plot(epochs, history["val_" + metric], label="Validation", color="#17648d")
        ax.set(xlabel="Epoch", ylabel=label)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_one(config):
    name = config["name"]
    if name in RESULTS:
        print("Already completed in this session:", name)
        return RESULTS[name]
    tf.keras.backend.clear_session()
    gc.collect()
    available_gb = psutil.virtual_memory().available / 1024**3
    if available_gb < 1.5:
        raise MemoryError(f"Available RAM before training: {available_gb:.2f} GB; minimum: 1.5 GB")
    tf.keras.utils.set_random_seed(SEED)
    folder = OUTPUT_DIR / name
    folder.mkdir(exist_ok=True)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"dataset": "MNIST", "assignment": "3", "session_id": SESSION_ID,
                         "summary_metrics": "final_epoch"})
        params = {
            "number_of_hidden_layers": len(config["nodes"]),
            "nodes_in_each_hidden_layer": json.dumps(config["nodes"]),
            "optimizer": config["optimizer"], "batch_size": config["batch_size"],
            "epochs": config["epochs"], "learning_rate": config["learning_rate"],
            "momentum": config["momentum"], "random_seed": SEED,
            "training_examples": len(y_train), "validation_examples": len(y_val),
            "normalization": "float32 divided by 255", "loss_function": "sparse_categorical_crossentropy",
        }
        for index, nodes in enumerate(config["nodes"], start=1):
            params[f"hidden_layer_{index}_nodes"] = nodes
        mlflow.log_params(params)
        model = build_model(config)
        model.summary()
        summary_lines = []
        model.summary(print_fn=lambda line, **kwargs: summary_lines.append(line))
        (folder / "architecture.txt").write_text("\n".join(summary_lines), encoding="utf-8")
        history = model.fit(
            make_dataset(x_train, y_train, config["batch_size"], training=True),
            validation_data=make_dataset(x_val, y_val, config["batch_size"]),
            batch_size=config["batch_size"], epochs=config["epochs"], shuffle=False,
            callbacks=[MemoryGuard()], verbose=2,
        ).history
        history = {key: [float(value) for value in values] for key, values in history.items()}
        if len(history["loss"]) != config["epochs"]:
            raise RuntimeError("Training stopped before the requested number of epochs.")
        metrics = {
            "training_accuracy": history["accuracy"][-1],
            "validation_accuracy": history["val_accuracy"][-1],
            "training_loss": history["loss"][-1],
            "validation_loss": history["val_loss"][-1],
        }
        mlflow.log_metrics(metrics, step=config["epochs"])
        (folder / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (folder / "hyperparameters.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        pd.DataFrame(history).rename_axis("epoch_index").to_csv(folder / "history.csv")
        save_learning_curves(history, folder / "learning_curves.png")
        model_path = folder / "final_model.keras"
        model.save(model_path)
        mlflow.log_artifacts(str(folder), artifact_path="details")
        result = {
            "run_name": name, "run_id": run.info.run_id,
            "hidden_layers": len(config["nodes"]), "hidden_nodes": json.dumps(config["nodes"]),
            "optimizer": config["optimizer"], "learning_rate": config["learning_rate"],
            "momentum": config["momentum"], "batch_size": config["batch_size"],
            "epochs": config["epochs"], **metrics,
        }
    RESULTS[name] = result
    MODEL_PATHS[name] = model_path
    HISTORIES[name] = history
    save_session_state()
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    pd.DataFrame(RESULTS.values()).to_csv(OUTPUT_DIR / "experiment_comparison.csv", index=False)
    print("Finished:", name, "| Validation accuracy:", f"{metrics['validation_accuracy']:.2%}")
    return result


# baseline
baseline_result = train_one(CONFIGS[0]) 
print(json.dumps(baseline_result, indent=2))


# baseline_links
print("Tracking mode:", TRACKING_MODE) 
print("MLflow Tracking URI:", TRACKING_URI)
print("Configured DagsHub repository:", DAGSHUB_URL)
print("Configured DagsHub Tracking URI:", DAGSHUB_TRACKING_URI)
print("Baseline run ID:", baseline_result["run_id"])


# additional
for config in CONFIGS[1:]: 
    train_one(config)
print("All four model runs are complete.")


# comparison
if len(RESULTS) != len(CONFIGS): 
    raise RuntimeError("Model selection requires four completed experiments.")
comparison = pd.DataFrame(RESULTS.values()).sort_values(
    ["validation_accuracy", "validation_loss"], ascending=[False, True], kind="stable"
).reset_index(drop=True)
comparison.to_csv(OUTPUT_DIR / "experiment_comparison.csv", index=False)
print(comparison.to_string(index=False))
best = comparison.iloc[0].to_dict()
gap = 100 * (best["training_accuracy"] - best["validation_accuracy"])
summary = (
    f"I selected {best['run_name']}, which has {best['hidden_layers']} hidden layer(s) "
    f"with nodes {best['hidden_nodes']}, optimizer {best['optimizer']}, learning rate "
    f"{best['learning_rate']}, batch size {best['batch_size']}, and {best['epochs']} epochs. "
    f"Its final validation accuracy is {best['validation_accuracy']:.2%}, with validation "
    f"loss {best['validation_loss']:.6f}. I selected the highest final validation accuracy, "
    f"using lower validation loss to break an exact accuracy tie. Its final training "
    f"accuracy is {best['training_accuracy']:.2%}, with training loss "
    f"{best['training_loss']:.6f}. The training-minus-validation accuracy gap is "
    f"{gap:.2f} percentage points. All four runs use the same held-out validation examples. "
    "The baseline trained for 50 epochs and the three additional configurations trained "
    "for 10 epochs each to limit computation. This compares the configurations with their "
    "tested training budgets; differences cannot be attributed to architecture or optimizer alone. "
    "The choice is based on validation performance; training accuracy and test accuracy "
    "were not used to select the model. Results from one split and one seed do not "
    "establish that a small difference will persist on other data."
)
(OUTPUT_DIR / "best_model_explanation.txt").write_text(summary + "\n", encoding="utf-8")
print("\nMODEL SELECTION\n" + summary)

fig, ax = plt.subplots(figsize=(10, 4.5))
names = comparison["run_name"].str.replace("_b64", "", regex=False)
values = 100 * comparison["validation_accuracy"]
ax.barh(names, values, color=["#17648d"] + ["#aaaaaa"] * (len(values) - 1))
ax.invert_yaxis()
ax.set(xlabel="Final validation accuracy (%)", xlim=(0, 104), title="MNIST validation results")
for index, value in enumerate(values):
    ax.text(value + 0.4, index, f"{value:.2f}%", va="center")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "validation_comparison.png", dpi=160)
plt.show()
plt.close(fig)

with mlflow.start_run(run_id=best["run_id"]):
    for filename in ["experiment_comparison.csv", "best_model_explanation.txt", "validation_comparison.png"]:
        mlflow.log_artifact(str(OUTPUT_DIR / filename), artifact_path="comparison")


# test
selected_model = tf.keras.models.load_model(MODEL_PATHS[best["run_name"]]) 
with mlflow.start_run(run_id=best["run_id"]):
    test_metrics = selected_model.evaluate(make_dataset(x_test, y_test, 64), verbose=0, return_dict=True)
    mlflow.log_metrics({"test_accuracy": float(test_metrics["accuracy"]),
                        "test_loss": float(test_metrics["loss"])})
print("Selected model test accuracy:", f"{test_metrics['accuracy']:.2%}")
print("Selected model test loss:", f"{test_metrics['loss']:.6f}")


# verification
client = MlflowClient() 
for row in RESULTS.values():
    tracked = client.get_run(row["run_id"])
    if tracked.info.status != "FINISHED":
        raise RuntimeError(f"Run is not finished: {row['run_name']}")
    for metric in ["training_accuracy", "validation_accuracy", "training_loss", "validation_loss"]:
        if not math.isclose(tracked.data.metrics[metric], row[metric], rel_tol=1e-6, abs_tol=1e-8):
            raise RuntimeError(f"Tracked value mismatch: {row['run_name']} / {metric}")
    expected_params = {
        "number_of_hidden_layers": str(row["hidden_layers"]),
        "nodes_in_each_hidden_layer": row["hidden_nodes"],
        "optimizer": row["optimizer"], "batch_size": str(row["batch_size"]),
    }
    for key, value in expected_params.items():
        if tracked.data.params.get(key) != value:
            raise RuntimeError(f"Tracked parameter mismatch: {row['run_name']} / {key}")
    for metric in ["accuracy", "val_accuracy", "loss", "val_loss"]:
        if len(client.get_metric_history(row["run_id"], metric)) != row["epochs"]:
            raise RuntimeError(f"Missing autologged epochs: {row['run_name']} / {metric}")
    if not any(item.path == "model" for item in client.list_artifacts(row["run_id"])):
        raise RuntimeError(f"Autologged model artifact missing: {row['run_name']}")
    print("Verified:", row["run_name"])

manifest = {
    "dagshub_url": DAGSHUB_URL, "dagshub_tracking_uri": DAGSHUB_TRACKING_URI,
    "tracking_mode": TRACKING_MODE, "tracking_uri": TRACKING_URI,
    "experiment_name": EXPERIMENT_NAME, "session_id": SESSION_ID,
    "runs": list(RESULTS.values()), "selected_run": best["run_name"],
    "selected_run_id": best["run_id"], "tensorflow_version": tf.__version__,
    "mlflow_version": mlflow.__version__, "python_version": sys.version.split()[0],
}
(OUTPUT_DIR / "session_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("\nAll four tracked runs verified.")
print("MLflow Tracking URI:", TRACKING_URI)
