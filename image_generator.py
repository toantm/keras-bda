from da_lenet5_mnist import build_generator, build_discriminator
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

latent_size = 100
generator = build_generator(latent_size)

# load the weights from the last epoch
generator.load_weights(sorted(glob('params_generator*'))[-1])

np.random.seed(31337)

noise = np.random.normal(loc=0.0, scale=1, size=(100, latent_size))

sampled_labels = np.array([
                              [i] * 10 for i in range(10)
                              ]).reshape(-1, 1)

# get a batch to display
generated_images = generator.predict(
    [noise, sampled_labels], verbose=0)

# arrange them into a grid
img = (np.concatenate([r.reshape(-1, 28)
                       for r in np.split(generated_images, 10)
                       ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

plt.imshow(img, cmap='gray')
_ = plt.axis('off')

plt.show()
