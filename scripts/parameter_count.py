"""Count trainable parameters of models."""
from torchsummary import summary

from src.learn_direct_train_classifier import get_model
from src.models import compute_parameter_total
from src.ptwt_continuous_transform import get_diff_wavelet


def main() -> None:
    """Define Wavelet, count parameters."""
    wavelet = get_diff_wavelet("cmor4.6-0.87")

    models = ["learndeepnet", "onednet", "learnnet"]
    sample_rate = 16000
    window_size = 11025

    f_sizes = [21888, 5440, 39168]
    totals = []
    for i in range(len(models)):
        model = get_model(
            wavelet, models[i], sample_rate=sample_rate, flattend_size=f_sizes[i]
        )

        totals.append(compute_parameter_total(model))

        summary(model, (1, window_size))
        print("")

    print("")
    for i in range(len(models)):
        print(f"{models[i]}: {totals[i]}")


if __name__ == "__main__":
    main()
