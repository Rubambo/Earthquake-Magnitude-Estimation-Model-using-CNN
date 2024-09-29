import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from google.colab import files

uploaded = files.upload()
file_path = 'data_test_1.npz'
data = np.load(file_path, mmap_mode='r')

waveforms = data['data']
print("Original shape of waveforms:", waveforms.shape)

waveforms_flattened = waveforms.reshape((waveforms.shape[0], -1))
print("Flattened shape for KMeans:", waveforms_flattened.shape)

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
waveform_clusters = kmeans.fit_predict(waveforms_flattened)

labels = np.where(waveform_clusters == 0, 0, 1)
print(f"Cluster labels: {labels}")

np.savez('labeled_seismic_data.npz', waveforms=waveforms, labels=labels)
data.close()

scaler = MinMaxScaler()
waveforms_normalized = scaler.fit_transform(waveforms_flattened)

waveforms_reshaped = waveforms_normalized.reshape((waveforms.shape[0], waveforms.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(waveforms_reshaped, labels, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(waveforms_reshaped.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)

for i in range(5):
    plt.figure(figsize=(10, 4))
    plt.plot(X_test[i].reshape(-1))
    plt.title(f"Predicted: {'Earthquake' if predictions[i][0] == 1 else 'Not Earthquake'}, True: {'Earthquake' if y_test[i] == 1 else 'Not Earthquake'}")
    plt.grid(True)
    plt.show()

