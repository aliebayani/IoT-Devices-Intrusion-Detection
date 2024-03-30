import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import csv

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from keras import regularizers

df = pd.read_csv('./DNN-EdgeIIoT-dataset.csv', low_memory=False)

from sklearn.utils import shuffle
drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4","arp.dst.proto_ipv4",
                "http.file_data","http.request.full_uri","icmp.transmit_timestamp",
                "http.request.uri.query", "tcp.options","tcp.payload","tcp.srcport",
                "tcp.dstport", "udp.port", "mqtt.msg"]

df.drop(drop_columns, axis=1, inplace=True)
df.dropna(axis=0, how='any', inplace=True)
df.drop_duplicates(subset=None, keep="first", inplace=True)
df = shuffle(df)
df.isna().sum()
print(df['Attack_type'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

encode_text_dummy(df,'http.request.method')
encode_text_dummy(df,'http.referer')
encode_text_dummy(df,"http.request.version")
encode_text_dummy(df,"dns.qry.name.len")
encode_text_dummy(df,"mqtt.conack.flags")
encode_text_dummy(df,"mqtt.protoname")
encode_text_dummy(df,"mqtt.topic")

# Apply Median Filtering to all features
window_size_median = 3  # Window size for median filtering
for feature in df.columns:
    if feature != 'Attack_type':  # Exclude the label column
        df[feature + '_median_filtered'] = df[feature].rolling(window=window_size_median, center=True).median()

# Apply Standard Deviation-based Filtering to all features
threshold_std = 3  # Threshold for standard deviation-based filtering
for feature in df.columns:
    if feature != 'Attack_type':  # Exclude the label column
        mean = df[feature].mean()
        std = df[feature].std()
        upper_bound = mean + threshold_std * std
        lower_bound = mean - threshold_std * std
        df[feature + '_std_filtered'] = df[feature].apply(lambda x: x if (lower_bound <= x <= upper_bound) else None)

# Drop original features that were filtered
df.drop(df.columns[df.columns.str.endswith('_filtered')], axis=1, inplace=True)

df.to_csv('preprocessed_DNN.csv', encoding='utf-8', index=False)

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
df = pd.read_csv('./preprocessed_DNN.csv', low_memory=False)
df

df['Attack_type'].value_counts()

df.info()

feat_cols = list(df.columns)
label_col = "Attack_type"

feat_cols.remove(label_col)
#feat_cols

empty_cols = [col for col in df.columns if df[col].isnull().all()]
empty_cols

skip_list = ["icmp.unused", "http.tls_port", "dns.qry.type", "mqtt.msg_decoded_as"]

df[skip_list[3]].value_counts()

fig, (ax1, ax2)  = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
explode = list((np.array(list(df[label_col].dropna().value_counts()))/sum(list(df[label_col].dropna().value_counts())))[::-1])[:]
labels = list(df[label_col].dropna().unique())[:]
sizes = df[label_col].value_counts()[:]

ax2.pie(sizes,  explode=explode, startangle=60, labels=labels, autopct='%1.0f%%', pctdistance=0.8)
ax2.add_artist(plt.Circle((0,0),0.4,fc='white'))
sns.countplot(y=label_col, data=df, ax=ax1)
ax1.set_title("Count of each Attack type")
ax2.set_title("Percentage of each Attack type")
plt.show()

X = df.drop([label_col], axis=1)
y = df[label_col]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

del X
del y

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train =  label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

label_encoder.classes_

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
X_train =  min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

# Verify column names
print(df.columns)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(X_train.shape)
print(X_test.shape)

input_shape = X_train.shape[1:]

print(X_train.shape, X_test.shape)
print(input_shape)

num_classes = len(np.unique(y_train))
num_classes

from  tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

print(y_train.shape, y_test.shape)

import numpy as np
from keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Load your saved model
model = load_model('model.h5')  # Adjust the filename as per your saved model

# Step 1: Prediction
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Step 2: Convert Predictions and True Labels
y_true_classes = np.argmax(y_test, axis=1)

# Step 3: Calculate Metrics
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Print Metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", conf_matrix)