from src.data_loader import get_costum_dataset

if __name__ == "__main__":
    save_path = "/home/s6kogase/data/data/run3"
    data_path = "/home/s6kogase/data/data/cross_test"
    limit_train = (55504, 7504, 15504)

    only_use = ["ljspeech", "fbmelgan"]
    train_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="train",
        only_use=only_use,
        save_path=save_path,
        limit=limit_train[0],
    )
    val_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="val",
        only_use=only_use,
        save_path=save_path,
        limit=limit_train[1],
    )
    test_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="test",
        only_use=only_use,
        save_path=save_path,
        limit=limit_train[2],
    )

    only_test_folders = ["conformer", "jsutmbmelgan", "jsutpwg"]
    limit_cross = (55500, 7304, 14600)

    cross_set_test = get_costum_dataset(
        data_path=data_path,
        ds_type="test",
        only_test_folders=only_test_folders,
        only_use=[
            "ljspeech",
            "melgan",
            "lmelgan",
            "mbmelgan",
            "pwg",
            "waveglow",
            "hifigan",
            "conformer",
            "jsutmbmelgan",
            "jsutpwg",
        ],
        save_path=save_path,
        limit=limit_cross[2],
    )
    cross_set_val = get_costum_dataset(
        data_path=data_path,
        ds_type="val",
        only_test_folders=only_test_folders,
        only_use=[
            "ljspeech",
            "melgan",
            "lmelgan",
            "mbmelgan",
            "pwg",
            "waveglow",
            "hifigan",
            "conformer",
            "jsutmbmelgan",
            "jsutpwg",
        ],
        save_path=save_path,
        limit=limit_cross[1],
    )
    cross_set_test = get_costum_dataset(
        data_path=data_path,
        ds_type="test",
        only_test_folders=only_test_folders,
        only_use=[
            "ljspeech",
            "melgan",
            "lmelgan",
            "mbmelgan",
            "pwg",
            "waveglow",
            "hifigan",
            "conformer",
            "jsutmbmelgan",
            "jsutpwg",
            "avocodo",
            "bigvgan",
            "bigvganl",
        ],
        save_path=save_path,
        limit=limit_cross[2],
    )
    cross_set_val = get_costum_dataset(
        data_path=data_path,
        ds_type="val",
        only_test_folders=only_test_folders,
        only_use=[
            "ljspeech",
            "melgan",
            "lmelgan",
            "mbmelgan",
            "pwg",
            "waveglow",
            "hifigan",
            "conformer",
            "jsutmbmelgan",
            "jsutpwg",
            "avocodo",
            "bigvgan",
            "bigvganl",
        ],
        save_path=save_path,
        limit=limit_cross[1],
    )
    cross_set_val = get_costum_dataset(
        data_path=data_path,
        ds_type="test",
        only_test_folders=only_test_folders,
        only_use=["ljspeech", "lbigvgan", "bigvgan"],
        save_path=save_path,
        limit=limit_cross[2],
    )
    cross_set_val = get_costum_dataset(
        data_path=data_path,
        ds_type="val",
        only_test_folders=only_test_folders,
        only_use=["ljspeech", "lbigvgan", "bigvgan"],
        save_path=save_path,
        limit=limit_cross[1],
    )
    cross_set_val = get_costum_dataset(
        data_path=data_path,
        ds_type="test",
        only_test_folders=only_test_folders,
        only_use=[
            "ljspeech",
            "avocodo",
        ],
        save_path=save_path,
        limit=limit_cross[2],
    )
    cross_set_val = get_costum_dataset(
        data_path=data_path,
        ds_type="val",
        only_test_folders=only_test_folders,
        only_use=[
            "ljspeech",
            "avocodo",
        ],
        save_path=save_path,
        limit=limit_cross[0],
    )
