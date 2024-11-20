import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import cv2
import glob
from sklearn.model_selection import train_test_split
from datetime import datetime


class MuscleActivationModel:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32, output_dir='./classifier_output'):
        """
        Initializes the MuscleActivationModel with data directory and parameters.

        Args:
            data_dir (str): Path to the data directory.
            img_size (tuple): Target size for images.
            batch_size (int): Batch size for training and validation.
            output_dir (str): Path to the output directory for saving models.
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.model = None
        self.train_generator = None
        self.validation_generator = None
        self.num_classes = None
        self.class_indices = None
        self.inverse_class_indices = None
        self.labels = None
        self.saved_model_path = None  # To store the path of the saved model

    def load_data(self):
        """
        Loads and preprocesses data from the specified directory.
        Assumes that labels are stored in 'labels.csv' within data_dir.
        """
        # Load labels.csv
        labels_csv_path = os.path.join(self.data_dir, 'labels.csv')
        if not os.path.exists(labels_csv_path):
            raise FileNotFoundError(f"Labels file not found at {labels_csv_path}")

        labels_df = pd.read_csv(labels_csv_path)

        # Ensure necessary columns exist
        if 'filename' not in labels_df.columns or 'labels' not in labels_df.columns:
            raise ValueError("labels.csv must contain 'filename' and 'labels' columns.")

        # Adjust the filename paths if needed
        labels_df['filename'] = labels_df['filename'].apply(lambda x: os.path.join('frames', x))

        # Convert labels to list
        labels_df['labels_list'] = labels_df['labels'].apply(lambda x: x.split(',') if pd.notna(x) and x != '' else [])

        # Get all unique classes
        all_labels = [label.strip() for sublist in labels_df['labels_list'] for label in sublist if label.strip()]
        unique_classes = sorted(list(set(all_labels)))
        self.num_classes = len(unique_classes)
        self.labels = unique_classes

        # Create a mapping from class names to indices
        self.class_indices = {label: idx for idx, label in enumerate(unique_classes)}
        self.inverse_class_indices = {idx: label for label, idx in self.class_indices.items()}

        # Create binary columns for each class
        for label in unique_classes:
            labels_df[label] = labels_df['labels_list'].apply(lambda x: 1 if label in x else 0)

        # Split data into training and validation sets
        train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)

        # Data augmentation and preprocessing
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1
        )

        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        # Flow from dataframe generators
        self.train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=self.data_dir,
            x_col='filename',
            y_col=unique_classes,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='raw',
            shuffle=True
        )

        self.validation_generator = val_datagen.flow_from_dataframe(
            dataframe=val_df,
            directory=self.data_dir,
            x_col='filename',
            y_col=unique_classes,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='raw',
            shuffle=False
        )

        print("Data loaded successfully.")

    def build_model(self):
        """
        Builds the Keras model using transfer learning from ResNet50.
        """
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.img_size + (3,))
        base_model.trainable = False  # Freeze the base model

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        output = Dense(self.num_classes, activation='sigmoid')(x)

        self.model = Model(inputs=base_model.input, outputs=output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        print("Model built successfully.")

    def train(self, epochs=10):
        """
        Trains the model using the data generators.

        Args:
            epochs (int): Number of epochs to train the model.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() before training.")

        self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator
        )

        # Save the trained model
        self._save_model()

        print("Model trained and saved successfully.")

    def _save_model(self):
        """
        Saves the trained model to the output directory with a timestamp.
        """
        models_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(models_dir, f'model_{timestamp}.h5')
        self.model.save(model_path)
        self.saved_model_path = model_path  # Store the saved model path
        print(f"Model saved to {model_path}")

    def evaluate(self):
        """
        Evaluates the model on the validation set.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() before evaluation.")

        loss, accuracy = self.model.evaluate(self.validation_generator)
        print(f"Validation Loss: {loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")

    def predict_on_test_data(self, test_data_dir='./input/test'):
        """
        Runs the model on test images and prints the predictions.

        Args:
            test_data_dir (str): Path to the test data directory.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() before prediction.")

        # Load test data
        labels_csv_path = os.path.join(test_data_dir, 'labels.csv')
        if not os.path.exists(labels_csv_path):
            raise FileNotFoundError(f"Labels file not found at {labels_csv_path}")

        labels_df = pd.read_csv(labels_csv_path)

        # Ensure necessary columns exist
        if 'filename' not in labels_df.columns or 'labels' not in labels_df.columns:
            raise ValueError("labels.csv must contain 'filename' and 'labels' columns.")

        # Adjust the filename paths if needed
        labels_df['filename'] = labels_df['filename'].apply(lambda x: os.path.join('frames', x))

        # Convert labels to list
        labels_df['labels_list'] = labels_df['labels'].apply(lambda x: x.split(',') if pd.notna(x) and x != '' else [])

        # Create binary columns for each class using the model's labels
        for label in self.labels:
            labels_df[label] = labels_df['labels_list'].apply(lambda x: 1 if label in x else 0)

        # Data preprocessing
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        test_generator = test_datagen.flow_from_dataframe(
            dataframe=labels_df,
            directory=test_data_dir,
            x_col='filename',
            y_col=self.labels,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='raw',
            shuffle=False
        )

        # Evaluate on test data
        loss, accuracy = self.model.evaluate(test_generator)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        # Get predictions
        predictions = self.model.predict(test_generator)

        # Convert predictions to labels
        predicted_labels = []
        threshold = 0.5  # You can adjust this threshold
        for pred in predictions:
            labels = [self.labels[i] for i, p in enumerate(pred) if p >= threshold]
            predicted_labels.append(labels)

        # Print predictions
        for idx, (filename, true_labels) in enumerate(zip(labels_df['filename'], labels_df['labels_list'])):
            print(f"Image: {filename}")
            print(f"True Labels: {true_labels}")
            print(f"Predicted Labels: {predicted_labels[idx]}")
            print("-" * 40)

    def apply_grad_cam(self, img_path, muscle_label=None):
        """
        Applies Grad-CAM to an image to visualize areas important for muscle contraction prediction.

        Args:
            img_path (str): Path to the input image.
            muscle_label (str): Specific muscle label to visualize. If None, uses the top predicted class.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() before applying Grad-CAM.")

        img_array = self._preprocess_image(img_path)

        # Get model predictions
        preds = self.model.predict(img_array)
        if muscle_label:
            if muscle_label not in self.class_indices:
                raise ValueError(f"Muscle label '{muscle_label}' not found in class indices.")
            class_idx = self.class_indices[muscle_label]
        else:
            # Use the top predicted class
            class_idx = np.argmax(preds[0])

        # Get the last convolutional layer
        last_conv_layer = self._find_last_conv_layer()

        # Create a model that maps the input image to the activations of the last conv layer and predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(last_conv_layer).output, self.model.output]
        )

        # Compute the gradient of the class output value with respect to the feature map
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)[0]
        output = conv_outputs[0]

        # Compute the guided gradients
        guided_grads = grads * tf.cast(output > 0, "float32") * tf.cast(grads > 0, "float32")

        # Compute the weights
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))

        # Create the Grad-CAM heatmap
        cam = np.dot(output.numpy(), weights.numpy())
        cam = cv2.resize(cam, self.img_size)
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min()) if cam.max() != cam.min() else cam

        # Load the original image
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create an RGB heatmap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Superimpose the heatmap on original image
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        # Display the results
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Grad-CAM Heatmap')
        plt.imshow(heatmap)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Superimposed Image')
        plt.imshow(superimposed_img)
        plt.axis('off')

        plt.suptitle(f"Grad-CAM for '{self.inverse_class_indices[class_idx]}'")
        plt.show()

    def _preprocess_image(self, img_path):
        """
        Loads and preprocesses an image for the model.

        Args:
            img_path (str): Path to the image file.

        Returns:
            np.array: Preprocessed image array.
        """
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def _find_last_conv_layer(self):
        """
        Finds the name of the last convolutional layer in the model.

        Returns:
            str: Name of the last convolutional layer.
        """
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("No convolutional layer found in the model.")

    def predict_on_video(self, video_path, display=True, output_path=None):
        """
        Processes a video frame by frame, makes predictions using the trained model,
        and displays the video with predictions overlaid in real-time.

        Args:
            video_path (str): Path to the input video file.
            display (bool): If True, displays the video with predictions in real-time.
            output_path (str): If provided, saves the output video with predictions to this path.
        """
        if self.model is None:
            raise ValueError("Model has not been loaded or built.")

        # Initialize video capture and get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video file: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Video writer to save the output video if output_path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Preprocess the frame for prediction
            img_array = self._preprocess_frame_for_prediction(frame)

            # Make prediction
            preds = self.model.predict(img_array)
            pred_labels = self._get_labels_from_predictions(preds[0])

            # Overlay predictions on the frame
            annotated_frame = self._overlay_predictions_on_frame(frame, pred_labels)

            # Display the frame
            if display:
                cv2.imshow('Muscle Activation Predictions', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # Exit if 'q' is pressed

            # Write the frame to output video
            if output_path:
                out.write(annotated_frame)

        # Release resources
        cap.release()
        if display:
            cv2.destroyAllWindows()
        if output_path:
            out.release()
            print(f"Output video saved to {output_path}")

    def _preprocess_frame_for_prediction(self, frame):
        """
        Preprocesses a video frame for prediction.

        Args:
            frame (np.array): The video frame.

        Returns:
            np.array: Preprocessed frame suitable for model prediction.
        """
        img = cv2.resize(frame, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def _get_labels_from_predictions(self, preds, threshold=0.5):
        """
        Converts model predictions to label names based on a threshold.

        Args:
            preds (np.array): The prediction probabilities.
            threshold (float): The threshold for classifying a label as present.

        Returns:
            list: List of predicted label names.
        """
        labels = [self.labels[i] for i, p in enumerate(preds) if p >= threshold]
        return labels

    def _overlay_predictions_on_frame(self, frame, pred_labels):
        """
        Overlays the predicted labels on the video frame.

        Args:
            frame (np.array): The original video frame.
            pred_labels (list): List of predicted label names.

        Returns:
            np.array: Annotated frame with predictions overlaid.
        """
        annotated_frame = frame.copy()
        text = ', '.join(pred_labels) if pred_labels else 'No Activation Detected'

        # Choose position and styling for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 2
        color = (0, 255, 0)  # Green color in BGR
        x, y = 10, 30  # Position to place the text

        # Put text on the frame
        cv2.putText(annotated_frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        return annotated_frame



def main():
    use_trained_model = False  # Set to True to load a pre-trained model, False to train a new one

    # Get the latest timestamped training data directory
    train_data_base_dir = 'classifier_input/train'
    train_data_dirs = [d for d in glob.glob(os.path.join(train_data_base_dir, '*')) if os.path.isdir(d)]
    if not train_data_dirs:
        raise FileNotFoundError("No training data directories found.")
    train_data_dir = max(train_data_dirs, key=os.path.getmtime)  # Use the most recent directory

    # Initialize the model instance
    model = MuscleActivationModel(data_dir=train_data_dir, output_dir='classifier_output')

    # Load and preprocess the data (necessary for both training and loading to initialize labels)
    model.load_data()

    if use_trained_model:
        # Get the latest saved model
        models_dir = os.path.join(model.output_dir, 'models')
        model_files = [f for f in glob.glob(os.path.join(models_dir, '*.h5')) if os.path.isfile(f)]
        if not model_files:
            raise FileNotFoundError("No saved model files found.")
        latest_model_path = max(model_files, key=os.path.getmtime)  # Use the most recent model
        # Load the saved model
        loaded_model = load_model(latest_model_path)
        print(f"Loaded model from {latest_model_path}")
        # Update the model in the class instance
        model.model = loaded_model
    else:
        # Build the model architecture
        model.build_model()
        # Train the model
        model.train(epochs=10)
        # Evaluate the model on the validation set
        model.evaluate()

    # # Apply Grad-CAM to an image
    # # Use an actual image from your dataset
    # sample_frames_dir = os.path.join(train_data_dir, 'frames')
    # sample_images = [f for f in glob.glob(os.path.join(sample_frames_dir, '*.jpg')) if os.path.isfile(f)]
    # if not sample_images:
    #     raise FileNotFoundError("No sample images found in frames directory.")
    # test_image_path = sample_images[0]  # Use the first image as a sample
    # model.apply_grad_cam(img_path=test_image_path, muscle_label='left_biceps')

    # Predict on test data
    # Get the latest timestamped test data directory
    # test_data_base_dir = './output/test'
    # test_data_dirs = [d for d in glob.glob(os.path.join(test_data_base_dir, '*')) if os.path.isdir(d)]
    # if not test_data_dirs:
    #     raise FileNotFoundError("No test data directories found.")
    # test_data_dir = max(test_data_dirs, key=os.path.getmtime)  # Use the most recent directory

    # model.predict_on_test_data(test_data_dir=test_data_dir)

    input_video_path = '/classifier_input/test/testplayback.mp4'
    output_video_path = './classifier_output/test/test_predictions.mp4'  # Optional
    model.predict_on_video(
        video_path=input_video_path,
        display=True,
        output_path=output_video_path
    )

if __name__ == "__main__":
    main()
