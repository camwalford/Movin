import numpy as np
import pandas as pd
import tqdm
from glob import glob
import keras
from keras import layers, Sequential, Input
from keras import ops
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from src.utils.logger import setup_logger
import os

logger = setup_logger(__name__, "./logs/keypoint_classifier", "INFO")
REQUIRED_COLUMNS = ["movement_label"] + [f"{coord}{i}" for i in range(33) for coord in ("x", "y", "z")] # 33 keypoints


def load_data(file_path, data_format="parquet"):
    file = os.path.join(file_path, "labels." + data_format)
    match data_format:
        case "parquet":
            df = pd.read_parquet(file)
        case "csv":
            df = pd.read_csv(file)
        case _:
            raise ValueError(f"Invalid data format: {data_format}")

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        logger.warning(f"Missing columns in {file_path}: {missing_columns}")
    df = df[[col for col in REQUIRED_COLUMNS if col in df.columns]]
    return df


def load_most_recent_data(base_path, movements):
    """
    Load the most recent data for the specified movements, keeping only the required columns.

    Args:
        base_path (str): Base directory containing movement data.
        movements (List[str]): List of movement names.

    Returns:
        pd.DataFrame: Concatenated DataFrame with data from all movements.
    """
    dfs = []
    for movement in movements:
        movement_path = os.path.join(base_path, movement)
        most_recent_dir = max(glob(movement_path + "/*"), key=os.path.getctime)
        logger.info(f"Loading data from {most_recent_dir}")

        df = load_data(most_recent_dir)
        dfs.append(df)

    # Concatenate all movement DataFrames
    concatenated_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded and concatenated data for {len(movements)} movements.")

    return concatenated_df


def preprocess_data(df, label_encoder=None):
    """
    Preprocess the data for training or testing.

    Args:
        df (pd.DataFrame): Input DataFrame.
        label_encoder (LabelEncoder, optional): Pre-fitted LabelEncoder. Defaults to None.

    Returns:
        np.ndarray: Feature matrix (X).
        np.ndarray: Label array (y).
        LabelEncoder: The fitted LabelEncoder.
    """
    logger.info(f"Available columns in DataFrame: {list(df.columns)}")
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing columns: {missing_columns}")
        raise KeyError(f"Missing required columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS]

    X = df.drop(columns=["movement_label"]).values
    y = df["movement_label"].values

    # Encode labels as integers
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    else:
        y = label_encoder.transform(y)

    return X, y, label_encoder


def build_model(input_shape, num_classes):
    """
    Build a simple neural network model.

    Args:
        input_shape (int): Number of input features.
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: Compiled TensorFlow model.
    """
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model





def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train the model on the dataset.

    Args:
        model: Compiled Keras model.
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation feature matrix.
        y_val (np.ndarray): Validation labels.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        history: Training history object.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    return history


def main():
    movements = ["jumping_jacks", "squat", "right_lunge", "left_lunge"]

    # Load and preprocess the most recent training data
    train_data = load_most_recent_data("../labeller/labeller_output/train", movements)
    X, y, label_encoder = preprocess_data(train_data)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build the model
    num_classes = len(label_encoder.classes_)
    input_shape = X_train.shape[1]
    model = build_model(input_shape, num_classes)

    # Train the model
    logger.info("Starting model training...")
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Save the model and label encoder to timestamped directories
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_filepath = f"./keypoint_classifier_output/{timestamp}"
    model_filepath = os.path.join(output_filepath, "movement_classifier_model.keras")
    label_encoder_filepath = os.path.join(output_filepath, "label_encoder_classes.npy")

    os.makedirs(output_filepath, exist_ok=True)
    os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(label_encoder_filepath), exist_ok=True)

    model.save(model_filepath)
    np.save(label_encoder_filepath, label_encoder.classes_)
    logger.info("Model and label encoder saved successfully.")

    # Unseen test data for final evaluation
    test_data = load_most_recent_data("../labeller/labeller_output/test", movements)
    X_test, y_test, _ = preprocess_data(test_data, label_encoder=label_encoder)
    logger.debug(f"Test feature matrix shape: {X_test.shape}")
    logger.debug(f"Test label array shape: {y_test.shape}")

    # Evaluate the model on the test data to avoid overfitting
    test_val_loss, test_val_accuracy = model.evaluate(X_test, y_test, verbose=1)
    logger.info(f"Test Set Loss: {test_val_loss:.4f}, Test Set Accuracy: {test_val_accuracy:.4f}")

    # Save the test set evaluation results to the output directory
    test_results_filepath = os.path.join(output_filepath, "test_results.txt")
    with open(test_results_filepath, "w") as f:
        f.write(f"Test Set Loss: {test_val_loss:.4f}\n")
        f.write(f"Test Set Accuracy: {test_val_accuracy:.4f}\n")
    logger.info(f"Test set evaluation results saved to {test_results_filepath}")

if __name__ == '__main__':
    main()