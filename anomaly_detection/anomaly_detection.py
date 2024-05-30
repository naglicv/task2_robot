import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


#NOTE: Code was not tested in simulation. It may not work perfectly on robot 
#      so some things might have to be changed like threshold below. 
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    return img

def load_images(image_paths):
    images = [load_and_preprocess_image(path) for path in image_paths]
    return np.array(images)

real_mona_lisa_image_paths = ["real1.jpg", "real2.jpg"]
real_images = load_images(real_mona_lisa_image_paths)

input_img = tf.keras.layers.Input(shape=(224, 224, 3))

x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = tf.keras.models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(real_images, real_images, epochs=50, batch_size=4, shuffle=True)

def calculate_reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2, axis=-1)

def show_anomaly(image_path):
    original_img = load_and_preprocess_image(image_path)
    original_img_batch = np.expand_dims(original_img, axis=0)
    reconstructed_img = autoencoder.predict(original_img_batch)[0]
    
    error_map = calculate_reconstruction_error(original_img, reconstructed_img)
    mean_error = np.mean(error_map)
    
    threshold = 0.01
    if mean_error > threshold:
        result = "Altered Mona Lisa"
    else:
        result = "Real Mona Lisa"
    

    #NOTE: Code below is for creating a heatmap.
    #      Anomalies are show as brighter colors (they are glowing).
    #      Couldn't find better solution -> If you can do better do it. GL
    #      The darker the part of img it means that that part is the same as in real Mona Lisa!
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor((original_img * 255).astype('uint8'), cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 4, 2)
    plt.title('Reconstructed Image')
    plt.imshow(cv2.cvtColor((reconstructed_img * 255).astype('uint8'), cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 4, 3)
    plt.title('Reconstruction Error')
    plt.imshow(error_map, cmap='hot')
    plt.colorbar()
    
    plt.subplot(1, 4, 4)
    plt.title(result)
    plt.text(0.5, 0.5, result, fontsize=15, ha='center')
    plt.axis('off')
    
    plt.show()
#NOTE: Use test_.jpg to test if the model is working and if it's detecting anomalies!
#      It's not working very good with test7.jpg.
#      That's the picture I created with small anomaly. IDK how to fix it :(
test_image_path = "test7.jpg"
show_anomaly(test_image_path)
