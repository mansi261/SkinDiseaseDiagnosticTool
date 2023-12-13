import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
df = pd.read_csv('//Users/mansivyas/RA-MultiModel-ReducedSpecifciation/Categorization/nine_partition_embeddings_40.csv')

# Process text data
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['label'])
sequences = tokenizer.texts_to_sequences(df['label'])
text_data = pad_sequences(sequences, maxlen=max_len)

# Process CLIP image embeddings
# Exclude text and label columns to get only embeddings
clip_data = df.drop(columns=['label', 'nine_partition_label']).values

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['nine_partition_label'])

# Split the dataset
X_train_text, X_test_text, X_train_clip, X_test_clip, y_train, y_test = train_test_split(
    text_data, clip_data, labels, test_size=0.2, random_state=42
)

# Text model
text_model = models.Sequential([
    layers.Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    layers.LSTM(64)
])

# CLIP image model (assuming CLIP embeddings are preprocessed)
clip_input = layers.Input(shape=(X_train_clip.shape[1],))
clip_model = layers.Dense(64, activation='relu')(clip_input)

# Combine models
combined = layers.concatenate([text_model.output, clip_model])

# Classifier
z = layers.Dense(64, activation='relu')(combined)
z = layers.Dense(len(np.unique(labels)), activation='softmax')(z)

# Final model
model = models.Model(inputs=[text_model.input, clip_input], outputs=z)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X_train_text, X_train_clip], y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate([X_test_text, X_test_clip], y_test, verbose=2)
print(f"Test accuracy: {test_acc}")
