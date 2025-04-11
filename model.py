from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(input_dim=len(le.classes_), output_dim=64, input_length=50),
    LSTM(64),
    Dense(10, activation='softmax')  # assuming 10 intent classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(padded_seqs, labels, epochs=10, batch_size=32)
