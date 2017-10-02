from da_lenet5_mnist import build_generator, build_discriminator
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

latent_size = 100
g = build_generator(latent_size)
nb_miss = 100

# load the weights from the last epoch
# g.load_weights(sorted(glob('params_generator*'))[-1])
g.load_weights('params_generator_epoch_068.hdf5')

np.random.seed(31337)


def make_digit(digit=None):
    noise = np.random.normal(loc=0.0, scale=1.0, size=(1, latent_size))

    sampled_label = np.array([
            digit if digit is not None else np.random.randint(0, 10, 1)
        ]).reshape(-1, 1)

    generated_image = g.predict(
        [noise, sampled_label], verbose=0)

    return np.squeeze((generated_image * 127.5 + 127.5).astype(np.uint8))

plt.imshow(make_digit(digit=8), cmap='gray_r', interpolation='nearest')
_ = plt.axis('off')

plt.show()


