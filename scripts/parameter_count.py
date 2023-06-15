"""Count trainable parameters of models."""
from src.models import compute_parameter_total
from src.ptwt_continuous_transform import get_diff_wavelet
from src.train_classifier import get_model


def main() -> None:
    """Define wavelet, count parameters."""
    wavelet = get_diff_wavelet("sym8")

    models = ["lcnn"]
    sample_rate = 22050
    num_of_scales = [256, 512]
    totals = []
    for _i in range(len(models)):
        model = get_model(
            wavelet=wavelet,
            model_name="lcnn",
            nclasses=2,
            batch_size=128,
            f_min=1,
            f_max=11025,
            sample_rate=sample_rate,
            num_of_scales=num_of_scales,
            features="none",
            hop_length=100,
            in_channels=1,
            channels=num_of_scales,
        )

        totals.append(compute_parameter_total(model))

        print("")

    print("")
    for i in range(len(models)):
        print(f"{models[i]}: {totals[i]}")


if __name__ == "__main__":
    main()
