from da_lenet5_mnist import build_generator, build_discriminator
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

latent_size = 100
g = build_generator(latent_size)

# load the weights from the last epoch
g.load_weights(sorted(glob('params_generator*'))[-1])

np.random.seed(31337)

noise = np.tile(np.random.uniform(-1, 1, (10, latent_size)), (10, 1))
sampled_labels = np.array([
    [i] * 10 for i in range(10)
]).reshape(-1, 1)

# get a batch to display
generated_images = g.predict(
    [noise, sampled_labels], verbose=0)

# arrange them into a grid
img = (np.concatenate([r.reshape(-1, 28)
                       for r in np.split(generated_images, 10)
                       ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

plt.imshow(img, cmap='gray')
_ = plt.axis('off')

plt.show()


