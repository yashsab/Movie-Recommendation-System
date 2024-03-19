import keras
import tensorflow as tf
# Load the Keras model using keras.models.load_model()
loaded_model = keras.models.load_model('recommendation_model.h5')
# You can then access the model's properties and methods as usual
print("hello yash")
print(loaded_model.summary())  # Example: Print model summary
