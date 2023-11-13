from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
import os 
import scipy
import tensorflow as tf
import xml.etree.ElementTree as ET

# images_directory = os.path.join('facedetection', 'Images')
images_directory = 'Images'

# Define your data paths and parameters
# train_data_directory = 'Images\TraningData'
# test_data_dir = 'Images\TestingData'
# validation_data_directory = 'Images\ValidateData'

# Define paths to training, validation, and testing datasets within the Images directory
train_data_directory = os.path.join(images_directory, 'TraningData')
validation_data_directory = os.path.join(images_directory, 'ValidateData')
test_data_dir = os.path.join(images_directory, 'TestingData')

num_classes = len(os.listdir(train_data_directory))
num_epochs = 18  # Set the number of training epochss

# Data augmentation and generators
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
    
)

train_generator = datagen.flow_from_directory(
    train_data_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(
    validation_data_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Create and compile your model

# model = keras.Sequential([
#     # ... Define your model layers here ...
# ])

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 3)))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
model = keras.Sequential([
    # Add convolutional layers and pooling layers here
    # Example: Conv2D, MaxPooling2D
    # ...
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# model.build(input_shape=(None, 150, 150, 3)) 
# model.summary()
# model.add(tf.keras.layers.InputLayer(input_shape=(150, 150, 3)))
# num_classes = len(os.listdir(train_data_directory))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train your model
# model.fit(train_generator, epochs=num_epochs, steps_per_epoch=len(train_generator), verbose=1)
model.fit(train_generator, epochs=18, validation_data=validation_generator)

# Evaluate the model on the test dataset
test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

test_loss, test_accuracy = model.evaluate(test_generator)

# evaluation = model.evaluate(validation_generator, steps=len(validation_generator))
evaluation = model.evaluate(validation_generator)

print("Validation Loss:", test_loss)
print("Validation Accuracy:", test_accuracy)


# Save your model
# model.save('my_image_model.h5')
# model.save('my_model.keras')
# model.save('my_image_model.xml')

# Create an XML-like structure to store the model architecture
model_xml = ET.Element("model")

# Create an element for each layer in the model
for layer in model.layers:
    layer_xml = ET.SubElement(model_xml, "layer")
    layer_xml.set("name", layer.name)
    layer_xml.set("type", type(layer).__name__)

    # Extract layer configuration (hyperparameters)
    layer_config = layer.get_config()
    for key, value in layer_config.items():
        param_xml = ET.SubElement(layer_xml, "param")
        param_xml.set("name", str(key))
        param_xml.text = str(value)

# Create an XML-like structure to store the model weights
weights_xml = ET.Element("weights")

# Export weights for each layer
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        layer_weights_xml = ET.SubElement(weights_xml, "layer_weights")
        layer_weights_xml.set("name", layer.name)

        for weight in weights:
            weight_xml = ET.SubElement(layer_weights_xml, "weight")
            weight_xml.text = ",".join(map(str, weight.flatten()))

# Combine the model architecture and weights into a single XML tree
model_xml.append(weights_xml)

# Save the XML-like structure to a file
tree = ET.ElementTree(model_xml)
tree.write("my_model.xml")



# Load your model (uncomment when needed)
# loaded_model = keras.models.load_model('my_image_model.h5')