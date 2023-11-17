import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('ai_model/Skin Cancer1.h5')

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('your_model.tflite', 'wb') as f:
    f.write(tflite_model)
