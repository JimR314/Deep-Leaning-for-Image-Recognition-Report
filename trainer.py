from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping  # type: ignore

def train_and_evaluate(model, data, epochs, batch_size, use_augmentation):
    (x_train, y_train), (x_val, y_val) = data
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=5,
                                  min_lr=0.0001)  # Learning rate scheduler
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    if use_augmentation:
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        datagen.fit(x_train)
        history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                            epochs=epochs, validation_data=(x_val, y_val), 
                            callbacks=[reduce_lr, early_stopping])  # Include the scheduler here
    else:
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
                            validation_data=(x_val, y_val), 
                            callbacks=[reduce_lr, early_stopping])  # Include the scheduler here
    
    val_loss, val_acc = model.evaluate(x_val, y_val)
    return history, val_loss, val_acc
