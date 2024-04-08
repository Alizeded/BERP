import numpy as np
import matplotlib.pyplot as plt


def plot_spectrogram(stft, title="Gammatonegram", xlim=None):
    magnitude = stft.numpy()
    spectrogram = 20 * np.log10(np.abs(magnitude) + 1e-8)
    figure, axis = plt.subplots(1, 1)
    img = axis.imshow(
        spectrogram, cmap="viridis", vmin=-100, vmax=0, origin="lower", aspect="auto"
    )
    figure.suptitle(title)
    plt.colorbar(img, ax=axis)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.show()
