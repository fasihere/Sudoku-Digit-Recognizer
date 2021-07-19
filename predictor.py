from keras.models import Sequential, load_model

model_path = r'keras_mnist.h5'

def predict(cells):
    # load the model and create predictions on the test set
    mnist_model = load_model(model_path)
    predicted_classes = mnist_model.predict_classes(cells)
    return predicted_classes
    