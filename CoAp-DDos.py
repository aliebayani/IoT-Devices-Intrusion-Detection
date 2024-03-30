import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

# Define Median Filtering function
def apply_median_filter(arr, window_size):
    filtered_arr = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            filtered_arr.append(median_filter(arr[i][j], size=window_size))
    return np.array(filtered_arr).reshape(arr.shape)

# Define Standard Deviation-based Filtering function
def apply_std_dev_filter(arr, window_size, num_std):
    filtered_arr = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            mean = np.mean(arr[i][j])
            std_dev = np.std(arr[i][j])
            filtered_col = []
            for val in arr[i][j]:
                if abs(val - mean) <= num_std * std_dev:
                    filtered_col.append(val)
                else:
                    filtered_col.append(mean)  # Replace outliers with mean
            filtered_arr.append(filtered_col)
    return np.array(filtered_arr).reshape(arr.shape)

# Function to split 3D array and labels
def split_3d_and_label(arr, time, amount):
    mylist = []
    labels = []
    sum1 = 0
    sum2 = 0
    for i in range(0, len(arr)):
        sublist = []
        low = i
        high = len(arr) - 1
        old_high = high
        
        while low <= high:
            middle = low + ((high - low) // 2)
            if int(abs(arr[low][1] - arr[middle][1])) < time:
                golden = 0
                for x in range(middle, old_high + 1):
                    if int(abs(arr[low][1] - arr[x][1])) < time:
                        golden = x + 1
                    else:
                        golden = x
                        break
                
                sum1 = sum(x.count('192.168.1.12') for x in arr[low:golden])
                sum2 = sum(x.count('192.168.1.5') for x in arr[low:golden])
                mylist.append(np.array(arr[low:golden]))
                
                if len(arr[low:golden]) >= amount and sum1 + sum2 >= amount:
                    labels.append(1)
                else:
                    labels.append(0)

                low = high + 1
            else:
                old_high = high + 1
                high = middle - 1
        print(i+1, " of ", len(arr), " Percent Done: ", ((i+1)/len(arr)*100), end="\r")
    return mylist, labels

# Function to pad 3D numpy array
def padded_numpy(arr):
    max_len = max(len(ele) for ele in arr)
    arbitrary_pad_for_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    padded_list = []
    for i in range(len(arr)):
        if arr[i].shape[0] == max_len:
            padded_list.append(arr[i].tolist())

        if arr[i].shape[0] < max_len:
            lists_to_add = max_len - arr[i].shape[0]
            leng_of_index_arr = len(arr[i]) - 1
            last_time = arr[i][leng_of_index_arr][1]

            li = arr[i].tolist()

            for j in range(lists_to_add):
                arbitrary_pad_for_list[1] = last_time + 1
                li.append(arbitrary_pad_for_list)
            padded_list.append(li)

    return padded_list

# Function to normalize 2D matrix
def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix

# Function to prepare array for normalization
def prepare_arr_for_norm(arr):
    for i in range(len(arr)):
        normalized_matrix = normalize_2d(arr[i])
        arr[i] = normalized_matrix
    return arr

# Read Tokenized .csv file into dataframe to convert to 2D list
df = pd.read_csv(r'./CoAPDDoSTokenized.csv')
train_data = df.values.tolist()

# Uses only 5000 packets for example purposes
train_data = train_data[5000:10000]

# To get every packet flow of length 10 seconds
time = 10

# Split training data into packet flows of length 'time' 
# where all flows with 350 or more packets from malicious ip's are labeled malicious
train_data, train_labels = split_3d_and_label(train_data, time, 350)  

train_data = np.array(train_data, dtype="object")

# Remove src IP address from all packets
for i in range(len(train_data)):
    train_data[i] = np.delete(train_data[i], 0, axis=1)
    train_data[i] = np.array(train_data[i]).astype(np.float32)

# Pad 3D array
train_data = padded_numpy(train_data)
train_data = np.array(train_data)

# Apply Median Filtering
window_size_median = 3
train_data = apply_median_filter(train_data, window_size_median)

# Apply Standard Deviation-based Filtering
window_size_std = 5
num_std = 2
train_data = apply_std_dev_filter(train_data, window_size_std, num_std)

# Normalize array
train_data = prepare_arr_for_norm(train_data)

# Count all 0's and 1's
count = np.count_nonzero(train_labels == 0)
count1s = np.count_nonzero(train_labels == 1)
print("\n0's: ", count, " 1's: ", count1s)

# Convert train_labels to float64
#train_labels = train_labels.astype(np.float64)
# Convert train_labels to a numpy array of float64
train_labels = np.array(train_labels).astype(np.float64)


# Split training data into training and test
train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.33, random_state=42)

# Shuffle training and Test data
train_labels, train_data = shuffle(train_labels, train_data)
test_labels, test_data = shuffle(test_labels, test_data)

# Define and compile the model
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(None, 16)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(16, return_sequences=True),
    LSTM(8),
    Dense(4, activation='tanh'),
    Dense(2, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0003),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model
#model.fit(x=train_data, y=train_labels, validation_split=0.1, batch_size=40, epochs=25, verbose=1)
history = model.fit(x=train_data, y=train_labels, validation_split=0.1, batch_size=40, epochs= 20, verbose=1)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x=test_data, y=test_labels)

print("Test Accuracy:", test_accuracy)

# Save the trained model
model.save('model.h5')
# Evaluate the model
predictions = model.predict(x=test_data, batch_size=100, verbose=1)
rounded_predictions = np.argmax(predictions, axis=-1)

# Confusion matrix
cm = confusion_matrix(test_labels, rounded_predictions)
plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.size': 22})
sn.heatmap(cm/np.sum(cm), annot=True, fmt=".2%", cmap='Blues')
plt.xlabel("Predictions")
plt.ylabel("Actual")
plt.show()

# Calculate precision, recall, and F1-score for validation data
val_predictions = model.predict(x=test_data, batch_size=100, verbose=1)
rounded_val_predictions = np.argmax(val_predictions, axis=-1)

# Compute precision, recall, and f1-score
precision = precision_score(test_labels, rounded_val_predictions, average='macro')
recall = recall_score(test_labels, rounded_val_predictions, average='macro')
f1 = f1_score(test_labels, rounded_val_predictions, average='macro')

print("Validation Precision:", precision)
print("Validation Recall:", recall)
print("Validation F1-score:", f1)

# Plot precision, recall, and f1-score
plt.figure(figsize=(8, 6))
plt.plot([precision, recall, f1], marker='o', color='b')
plt.title('Validation Metrics')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.xticks(range(3), ['Precision', 'Recall', 'F1-score'])
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot training history and confusion matrix in a single figure
plt.figure(figsize=(14, 10))

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy', linestyle='-', marker='o', color='b')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='-', marker='o', color='r')

# Plot loss
plt.plot(history.history['loss'], label='Training Loss', linestyle='-', marker='o', color='g')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='-', marker='o', color='m')

plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sn.heatmap(cm/np.sum(cm), annot=True, fmt=".2%", cmap='Blues')
plt.xlabel("Predictions")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
