from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100 # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

def load_data(dataset_name):
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset_name == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        raise ValueError("Dataset not supported: {}".format(dataset_name))
    
    # Normalize the images
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    
    # Reshape the data to include the channel dimension if necessary
    if dataset_name in ['mnist', 'fashion_mnist']:
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
    elif dataset_name in ['cifar10', 'cifar100']:
        # CIFAR datasets already have the channel dimension
        pass

    # One-hot encode the labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    return (x_train, y_train), (x_test, y_test)
