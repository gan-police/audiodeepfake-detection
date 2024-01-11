"""Prepare custom dataset for asvspoof dataset 2019 and 2021."""
from src.audiofakedetect.data_loader import get_costum_dataset

if __name__ == "__main__":
    save_path = "./data/run2"
    data_path = "./data/asv"
    limit_train = (7472, 7672, 21320)
    seconds = 2

    only_use = ["asv2019real", "asv2019fake"]
    train_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="train",
        only_use=only_use,
        save_path=save_path,
        limit=limit_train[0],
        asvspoof_name="LA_T",
        train_ratio=1.0,
        val_ratio=0.0,
        file_type="flac",
        resample_rate=16000,
        seconds=seconds,
    )
    val_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="val",
        only_use=only_use,
        save_path=save_path,
        limit=limit_train[1],
        asvspoof_name="LA_D",
        train_ratio=0.0,
        val_ratio=1.0,
        file_type="flac",
        resample_rate=16000,
        seconds=seconds,
    )
    test_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="test",
        only_use=only_use,
        save_path=save_path,
        limit=limit_train[2],
        asvspoof_name="LA_E",
        train_ratio=0.0,
        val_ratio=0.0,
        file_type="flac",
        resample_rate=16000,
        seconds=seconds,
    )

    limit_train = (44368, 6336, 12672)

    only_use = ["asv2021real", "asv2021fake"]
    train_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="train",
        only_use=only_use,
        save_path=save_path,
        limit=limit_train[0],
        asvspoof_name="DF_E",
        file_type="flac",
        resample_rate=16000,
        seconds=seconds,
    )
    val_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="val",
        only_use=only_use,
        save_path=save_path,
        limit=limit_train[1],
        asvspoof_name="DF_E",
        file_type="flac",
        resample_rate=16000,
        seconds=seconds,
    )
    test_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="test",
        only_use=only_use,
        save_path=save_path,
        limit=limit_train[2],
        asvspoof_name="DF_E",
        file_type="flac",
        resample_rate=16000,
        seconds=seconds,
    )
