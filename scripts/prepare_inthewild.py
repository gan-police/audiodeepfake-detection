from src.data_loader import get_costum_dataset

if __name__ == "__main__":
    save_path = "/p/home/jusers/gasenzer1/juwels/project_drive/kgasenzer/audiodeepfakes/data/run2"
    data_path = "/p/home/jusers/gasenzer1/juwels/project_drive/kgasenzer/audiodeepfakes/data/inthewild/set"
    limit_train = (38968, 5568, 11136)
    seconds = 4

    only_use = ["inthewildReal", "inthewildFake"]
    train_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="train",
        only_use=only_use,
        save_path=save_path,
        limit=limit_train[0],
        resample_rate=16000,
        seconds=seconds,
    )
    val_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="val",
        only_use=only_use,
        save_path=save_path,
        limit=limit_train[1],
        resample_rate=16000,
        seconds=seconds,
    )
    test_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="test",
        only_use=only_use,
        save_path=save_path,
        limit=limit_train[2],
        resample_rate=16000,
        seconds=seconds,
    )
