from src.data_loader import get_costum_dataset

if __name__ == "__main__":
    save_path = "/home/s6kogase/data/data/run2"
    data_path = "/home/s6kogase/data/data/cross_test"
    limit_train = (55504, 7504, 15504)
    train_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="train",
        only_use=["ljspeech", "fbmelgan"],
        save_path=save_path,
        limit=limit_train,
    )
    val_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="val",
        only_use=["ljspeech", "fbmelgan"],
        save_path=save_path,
        limit=limit_train,
    )
    test_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="test",
        only_use=["ljspeech", "fbmelgan"],
        save_path=save_path,
        limit=limit_train,
    )
    cross_set_test = get_costum_dataset(
        data_path="/home/s6kogase/data/data/cross_test",
        ds_type="test",
        only_test_folders=["conformer", "jsutmbmelgan", "jsutpwg"],
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
        save_path="/home/s6kogase/data/data/run2",
        limit=(55500, 7304, 14600),
    )
    cross_set_val = get_costum_dataset(
        data_path="/home/s6kogase/data/data/cross_test",
        ds_type="val",
        only_test_folders=["conformer", "jsutmbmelgan", "jsutpwg"],
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
        save_path="/home/s6kogase/data/data/run2",
        limit=(55500, 7304, 14600),
    )
    cross_set_test = get_costum_dataset(
        data_path="/home/s6kogase/data/data/cross_test",
        ds_type="test",
        only_test_folders=["conformer", "jsutmbmelgan", "jsutpwg"],
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
            "bigvganl"
        ],
        save_path="/home/s6kogase/data/data/run2",
        limit=(55500, 7304, 14600),
    )
    cross_set_val = get_costum_dataset(
        data_path="/home/s6kogase/data/data/cross_test",
        ds_type="val",
        only_test_folders=["conformer", "jsutmbmelgan", "jsutpwg"],
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
            "bigvganl"
        ],
        save_path="/home/s6kogase/data/data/run2",
        limit=(55500, 7304, 14600),
    )
    cross_set_val = get_costum_dataset(
        data_path="/home/s6kogase/data/data/cross_test",
        ds_type="val",
        only_test_folders=["conformer", "jsutmbmelgan", "jsutpwg"],
        only_use=["ljspeech", "avocodo", "lbigvgan", "bigvgan"],
        save_path="/home/s6kogase/data/data/run2",
        limit=(55500, 7304, 14600),
    )
    cross_set_val = get_costum_dataset(
        data_path="/home/s6kogase/data/data/cross_test",
        ds_type="val",
        only_test_folders=["conformer", "jsutmbmelgan", "jsutpwg"],
        only_use=["ljspeech", "avocodo", "lbigvgan", "bigvgan"],
        save_path="/home/s6kogase/data/data/run2",
        limit=(55500, 7304, 14600),
    )
