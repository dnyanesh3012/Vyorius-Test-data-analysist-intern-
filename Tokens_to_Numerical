from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences

# Encode tokens to integers
le = LabelEncoder()
flat_tokens = [token for seq in sequences for token in seq]
le.fit(flat_tokens)

encoded_sequences = [[le.transform([token])[0] for token in seq] for seq in sequences]

# Pad sequences
padded_seqs = pad_sequences(encoded_sequences, maxlen=50, padding='post')
