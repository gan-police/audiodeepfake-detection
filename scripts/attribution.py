"""Plot attribution using saved attribution means."""
from src.audiofakedetect.integrated_gradients import plot_attribution

if __name__ == "__main__":
    transformations = ["packets"]
    wavelets = ["sym5"]
    cross_sources = [
        "melgan-lmelgan-mbmelgan-pwg-waveglow-avocodo-hifigan-conformer-jsutmbmelgan-jsutpwg-lbigvgan-bigvgan",
    ]

    seconds = 1
    sample_rate = 22050
    num_of_scales = 256

    plot_attribution(
        transformations=transformations,
        wavelets=wavelets,
        cross_sources=cross_sources,
        plot_path="./plots",
        seconds=seconds,
        sample_rate=sample_rate,
        num_of_scales=num_of_scales
    )
