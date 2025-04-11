def predict_user_intent(new_sequence):
    encoded = le.transform(new_sequence)
    padded = pad_sequences([encoded], maxlen=50, padding='post')
    pred = model.predict(padded)
    return pred.argmax()

# Example usage
predict_user_intent(['Ctrl+C', 'Ctrl+V', 'Click_Chart'])
