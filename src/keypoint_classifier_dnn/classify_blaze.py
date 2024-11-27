import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from dnn import DNN

path = "classifier_input/blaze"
cols_to_remove = [
    'SHOULDER_LEFT', 'SHOULDER_RIGHT',
    'HIP_LEFT', 'HIP_RIGHT',
    'KNEE_LEFT', 'KNEE_RIGHT',
]

# grab full data
full_data = pd.DataFrame()
for entry in os.listdir(path):
    if os.path.isfile(os.path.join(path, entry)):
        if entry.endswith('.parquet'):
            data = pd.read_parquet(path + "/" + entry)
            data.drop(columns=['frame_filename'], inplace=True)
            data.drop(
                columns=[col for col in cols_to_remove if col in data.columns],
                inplace=True
            )
            full_data = pd.concat([full_data, data], ignore_index=True)


# print("\nFirst few rows of data:")
# print(full_data.head())

# Split into data and label
x = full_data.drop(["movement_label"], axis=1)
y = full_data.movement_label.values

# Encode string labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into train and test data
x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.25, shuffle=True
)
print('Shape of train data is : ', x_train.shape)
print('Shape of label data is : ', y_train.shape)

# Convert labels into categorical features (one-hot encoding)
y_train = to_categorical(y_train, num_classes=len(label_encoder.classes_))
y_test = to_categorical(y_test, num_classes=len(label_encoder.classes_))

# Train and predict model
bdnn = DNN()
bdnn.train_model(x_train, y_train, "nn_blaze.keras")
bdnn.predict_model(x_test)
