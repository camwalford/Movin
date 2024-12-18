import numpy as np
import pandas as pd
from glob import glob
from keras import Sequential, Input
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils.custom_logger import setup_logger
import os

logger = setup_logger(__name__, "logs/keypoint_classifier", "INFO")
REQUIRED_COLUMNS = ["movement_label"] + [f"{coord}{i}" for i in range(33) for coord in ("x", "y", "z")]  # 33 keypoints

def load_data(file_path, data_format="parquet"):
    file = os.path.join(file_path, "labels." + data_format)
    if data_format == "parquet":
        df = pd.read_parquet(file)
    elif data_format == "csv":
        df = pd.read_csv(file)
    else:
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
        logger.info(f"Loading data for {movement} from {movement_path}")

        # Get all matching files/directories
        paths = glob(movement_path + "/*")
        if not paths:
            logger.warning(f"No data found for {movement} in {movement_path}")
            continue

        most_recent_dir = max(paths, key=os.path.getctime)
        logger.info(f"Loading data from {most_recent_dir}")
        try:
            df = load_data(most_recent_dir)
            dfs.append(df)
        except FileNotFoundError:
            logger.warning(f"No data file found for {movement} in {most_recent_dir}")

    if not dfs:
        raise ValueError("No valid data found for any of the specified movements.")

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
    early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.20,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    return history

def main():
    movements = ["jumping_jacks", "squat", "right_lunge", "left_lunge", "idle", "left_oblique"]

    # Load and preprocess the most recent training data
    train_data = load_most_recent_data("src/data_labelling/labeller_output/train", movements)
    X, y, label_encoder = preprocess_data(train_data)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build the model
    num_classes = len(label_encoder.classes_)
    input_shape = X_train.shape[1]
    model = build_model(input_shape, num_classes)
    model.summary()

    # Train the model
    logger.info("Starting model training...")
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=50)

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Save the model and label encoder to timestamped directories
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_filepath = f"src/keypoint_classifier_output/{timestamp}"
    model_filepath = os.path.join(output_filepath, "model/model.h5")
    label_encoder_filepath = os.path.join(output_filepath, "model/label_encoder.npy")

    os.makedirs(output_filepath, exist_ok=True)

    model.save(model_filepath)
    np.save(label_encoder_filepath, label_encoder.classes_)
    logger.info("Model and label encoder saved successfully.")

    # Unseen test data for final evaluation
    test_data = load_most_recent_data("src/data_labelling/labeller_output/test", movements)
    X_test, y_test, _ = preprocess_data(test_data, label_encoder=label_encoder)
    logger.debug(f"Test feature matrix shape: {X_test.shape}")
    logger.debug(f"Test label array shape: {y_test.shape}")

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    logger.info(f"Test Set Loss: {test_loss:.4f}, Test Set Accuracy: {test_accuracy:.4f}")

    # Generate predictions
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # Generate classification report
    class_report_dict = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_, output_dict=True)
    class_report_str = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)
    logger.info(f"Classification Report:\n{class_report_str}")

    test_results_filepath = os.path.join(output_filepath, "test_results")
    os.makedirs(test_results_filepath, exist_ok=True)
    # Save classification report to file
    class_report_filepath = os.path.join(test_results_filepath, "classification_report.txt")
    with open(class_report_filepath, "w") as f:
        f.write(class_report_str)
    logger.info(f"Classification report saved to {class_report_filepath}")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    confusion_matrix_filepath = os.path.join(test_results_filepath, "confusion_matrix.png")
    plt.savefig(confusion_matrix_filepath)
    plt.close()
    logger.info(f"Confusion matrix saved to {confusion_matrix_filepath}")

    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    accuracy_plot_filepath = os.path.join(test_results_filepath, "accuracy_plot.png")
    plt.savefig(accuracy_plot_filepath)
    plt.close()
    logger.info(f"Accuracy plot saved to {accuracy_plot_filepath}")

    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    loss_plot_filepath = os.path.join(test_results_filepath, "loss_plot.png")
    plt.savefig(loss_plot_filepath)
    plt.close()
    logger.info(f"Loss plot saved to {loss_plot_filepath}")

    # Plot precision, recall, and F1-score for each class
    classes = label_encoder.classes_
    precision = []
    recall = []
    f1_score = []

    for cls in classes:
        cls_report = class_report_dict[cls]
        precision.append(cls_report['precision'])
        recall.append(cls_report['recall'])
        f1_score.append(cls_report['f1-score'])

    # Plot Precision per class
    plt.figure()
    plt.bar(classes, precision, color='skyblue')
    plt.title('Precision per Class')
    plt.xlabel('Class')
    plt.ylabel('Precision')
    plt.xticks(rotation=45)
    plt.ylim([0, 1])
    plt.tight_layout()
    precision_plot_filepath = os.path.join(test_results_filepath, "precision_per_class.png")
    plt.savefig(precision_plot_filepath)
    plt.close()
    logger.info(f"Precision per class plot saved to {precision_plot_filepath}")

    # Plot Recall per class
    plt.figure()
    plt.bar(classes, recall, color='lightgreen')
    plt.title('Recall per Class')
    plt.xlabel('Class')
    plt.ylabel('Recall')
    plt.xticks(rotation=45)
    plt.ylim([0, 1])
    plt.tight_layout()
    recall_plot_filepath = os.path.join(test_results_filepath, "recall_per_class.png")
    plt.savefig(recall_plot_filepath)
    plt.close()
    logger.info(f"Recall per class plot saved to {recall_plot_filepath}")

    # Plot F1-score per class
    plt.figure()
    plt.bar(classes, f1_score, color='salmon')
    plt.title('F1 Score per Class')
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.ylim([0, 1])
    plt.tight_layout()
    f1_score_plot_filepath = os.path.join(test_results_filepath, "f1_score_per_class.png")
    plt.savefig(f1_score_plot_filepath)
    plt.close()
    logger.info(f"F1 Score per class plot saved to {f1_score_plot_filepath}")

if __name__ == '__main__':
    main()
