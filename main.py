from data_loader import *
from model_builder import *
from trainer import train_and_evaluate
from utils import *
from tensorflow.keras.applications import VGG19 # type: ignore
from tensorflow.keras.applications.vgg19 import preprocess_input # type: ignore
from tensorflow.keras.layers import Dense, Flatten # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.datasets import cifar100 # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

def main():
    # MNIST: Standard 28x28 handwritten digits. 60,000 training samples, 10,000 test samples
    dataset_name = 'cifar10'  # 'mnist', 'fashion_mnist', 'cifar10', 'cifar100
    epochs = 50
    use_augmentation = True
    model_name = dataset_name + ('_augmented' if use_augmentation else '') + '_complex_cnn_dropout_test_epochs' + str(epochs) # Will be used to save the model
    batch_size = 32
    
    (x_train, y_train), (x_test, y_test) = load_data(dataset_name)
    input_shape = x_train.shape[1:] # = (28, 28, 1) for MNIST
    num_classes = y_train.shape[1] # = 10 for MNIST
    
    # Optionally use a smaller subset of the training data to induce overfitting
    # Range of subset_size: 1 to 60,000 for the standard MNIST
    # Also consider how many sample to use for the vaildation set
    subset_size = 40000  # For example, use 50,000 samples for training
    validation_size = 10000  # For example, use 10,000 samples for validation
    x_train_subset = x_train[:subset_size]
    y_train_subset = y_train[:subset_size]
    x_validation = x_train[subset_size:subset_size + validation_size]
    y_validation = y_train[subset_size:subset_size + validation_size]

    model = build_complex_cnn_with_dropout(input_shape, num_classes)
    # model, history = load_model_and_history('cifar100_augmented_vgg19_dropout_BN_plateauLR_epochs25')
    history, val_loss, val_acc = train_and_evaluate(model,
                                                    ((x_train_subset, y_train_subset),
                                                     (x_validation, y_validation)),
                                                     epochs,
                                                     batch_size,
                                                     use_augmentation
                                                     )
    
    save_model_and_history(model, history, model_name)
    print(f"Model: {model_name}, Validation accuracy: {val_acc}, Validation loss: {val_loss}")


if __name__ == '__main__':
    main()


    # dataset_name = 'cifar10'
    # (x_train, y_train), (x_test, y_test) = load_data(dataset_name)
    # model, history = load_model_and_history('cifar10_augmented_complex_cnn_test_epochs50')
    # test_loss, test_acc = model.evaluate(x_test, y_test)
    # print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
    
'''
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import load_model
    from tensorflow.keras.datasets import cifar100
    from tensorflow.keras.utils import to_categorical

    # Load the trained model
    model_path = 'results/cifar100_augmented_complex_dropout_cnn_BN_plateauLR_epochs50.keras'
    model = load_model(model_path)

    # Load the CIFAR-100 dataset
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # Normalize the images
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode the labels
    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)

    # Create a validation set
    validation_split = 0.1  # 10% of training data for validation
    val_size = int(len(x_train) * validation_split)
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    # Select an image from the validation set
    index = 2002  # You can change this index to view different images
    image = x_val[index]
    true_label = np.argmax(y_val[index])

    # Make a prediction
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_label = np.argmax(prediction, axis=1)[0]

    # CIFAR-100 class names
    class_names = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm"
    ]

    # Visualize the image and the prediction
    plt.imshow(image)
    plt.title(f"True Label: {class_names[true_label]}, Predicted: {class_names[predicted_label]}")
    plt.show()
'''