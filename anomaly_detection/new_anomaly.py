import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    return img

def load_images(image_paths):
    images = [load_and_preprocess_image(path) for path in image_paths]
    return np.array(images)

real_mona_lisa_image_paths = ["real2.jpg"]
real_images = load_images(real_mona_lisa_image_paths)

input_img = tf.keras.layers.Input(shape=(224, 224, 3))


x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)


x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = tf.keras.models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(real_images, real_images, epochs=100, batch_size=4, shuffle=True)

def calculate_reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2, axis=-1)

def show_anomaly(image_path):
    original_img = load_and_preprocess_image(image_path)
    original_img_batch = np.expand_dims(original_img, axis=0)
    reconstructed_img = autoencoder.predict(original_img_batch)[0]
    
    error_map = calculate_reconstruction_error(original_img, reconstructed_img)
    mean_error = np.mean(error_map)
    
    threshold = 1.0 * np.mean(calculate_reconstruction_error(real_images, autoencoder.predict(real_images)))
    
    if mean_error > threshold:
        result = "Altered Mona Lisa"
    else:
        result = "Real Mona Lisa"
    
    plt.figure(figsize=(20, 5))  # Adjusted figsize for the new subplot
    
    plt.subplot(1, 5, 1)  # Original Image
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor((original_img * 255).astype('uint8'), cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 5, 2)  # Reconstructed Image
    plt.title('Reconstructed Image')
    plt.imshow(cv2.cvtColor((reconstructed_img * 255).astype('uint8'), cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 5, 3)  # Reconstruction Error
    plt.title('Reconstruction Error')
    plt.imshow(error_map, cmap='inferno')  # Use 'inferno' colormap for brighter colors
    plt.colorbar()
    
    plt.subplot(1, 5, 4)  # Result
    plt.title(result)
    plt.text(0.5, 0.5, result, fontsize=15, ha='center')
    plt.axis('off')
    
    plt.show()

test_image_path = "real1.jpg"
show_anomaly(test_image_path)

