import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your dataset
data = pd.read_csv('Customer Purchasing Behaviors.csv')

# Print the first few rows of the dataset
print(data.head())

# Print basic statistics of the dataset
print(data.describe(include='all'))  # include='all' to describe categorical columns as well

# Check for missing values
print(data.isnull().mean())

# Create an example DataFrame for Label Encoding demonstration
df = pd.DataFrame({'region': ['North', 'South', 'East', 'West']})

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'region' column in df
df['category_encoded'] = label_encoder.fit_transform(df['region'])
print(df)

# Apply Label Encoding to the 'region' column in the original dataset
# Make sure to use the correct column name from your actual dataset
data['region_encoded'] = label_encoder.fit_transform(data['region'])

# Print the updated dataset with encoded 'region' column
print(data.head())
data = data.drop(columns=['region'])
print(data.head())









import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Data Loading and Preprocessing

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the images to the range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape data to fit the model (28x28 images with 1 color channel)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# 2. Model Building

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Model Training

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# 4. Model Evaluation

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# 5. Model Saving and Loading

# Save the model
model.save('mnist_cnn_model.h5')

# Load the model
loaded_model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Predict using the loaded model
predictions = loaded_model.predict(X_test)
predicted_classes = tf.argmax(predictions, axis=1)

# Display the first 5 predictions and the corresponding images
for i in range(5):
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_classes[i]}, Actual: {y_test[i]}")
    plt.show()
