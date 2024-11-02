from torch.utils.data import DataLoader

from src.dataset import WeatherDataset, Split


def get_train_dataloader_weather_dataset(path: str, batch_size: int) -> DataLoader:
    dset = WeatherDataset(path=path, split=Split.TRAIN)

    return DataLoader(dset, batch_size=batch_size, shuffle=True)

def get_eval_dataloader_weather_dataset(path: str, batch_size: int) -> DataLoader:
    dset = WeatherDataset(path=path, split=Split.EVAL)

    return DataLoader(dset, batch_size=batch_size, shuffle=False)

def get_test_dataloader_weather_dataset(path: str, batch_size: int) -> DataLoader:
    dset = WeatherDataset(path=path, split=Split.TEST)

    return DataLoader(dset, batch_size=batch_size, shuffle=False)
