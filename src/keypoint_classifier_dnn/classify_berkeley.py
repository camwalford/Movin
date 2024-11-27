import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from dnn import DNN

path = './classifier_input/berkeley/'

full_data = pd.DataFrame()

# grab full data
for entry in sorted(os.listdir(path)):
    print(entry)
    if os.path.isfile(os.path.join(path, entry)):
        if entry.endswith('.txt'):
            data = pd.read_csv(path + entry, sep=' ', header=None)
            data.drop([129, 130], inplace=True, axis=1)
            data['classs'] = entry[-10:-8]
            full_data = pd.concat([full_data, data], ignore_index=True)

# Split into data and label
x = full_data.drop(["classs"], axis=1)
y = full_data.classs.values

# Label t-pose as class type 12
y = pd.DataFrame(y)
y.iloc[:, 0] = y.iloc[:, 0].str.replace('t', '1')
y.iloc[:, 0] = y.iloc[:, 0].str.replace('-', '2')
y.astype('int32')

# Split into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                    shuffle=True)
print('Shape of train data is : ', x_train.shape)
print('Shape of label data is : ', y_train.shape)

# Convert labels into categorical features
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Train and predict model
bdnn = DNN()
bdnn.train_model(x_train, y_train, "nn_blaze.keras")
bdnn.predict_model(x_test)