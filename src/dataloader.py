from torch.utils.data import DataLoader
from dataset import WeatherDataset, Split, TransformationPipeline
from data_preprocessing import ReplaceNaNs, ReplaceCityName, TokenizeUnits, AssembleCustomOverview, ReduceKeys

def create_weather_dataloader(path: str, batch_size: int, split: Split, cached: bool, shuffle: bool) -> DataLoader:
    """
    Creates a DataLoader for the WeatherDataset.
    :param path: Path to the dataset.
    :param batch_size: Batch size for the DataLoader.
    :param split: Dataset split (TRAIN, EVAL, TEST).
    :param cached: Whether to cache the data in memory.
    :param shuffle: Whether to shuffle the dataset.
    :return: DataLoader instance.
    """
    transformations = TransformationPipeline([
        ReplaceNaNs(),
        ReplaceCityName(),
        TokenizeUnits(),
        AssembleCustomOverview(),
        ReduceKeys()
    ])
    dataset = WeatherDataset(path=path, split=split, transformations=transformations, cached=cached)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)    

def get_train_dataloader_weather_dataset(path: str, batch_size: int, cached: bool) -> DataLoader:
    return create_weather_dataloader(path, batch_size, Split.TRAIN, cached, shuffle=True)


def get_eval_dataloader_weather_dataset(path: str, batch_size: int, cached: bool) -> DataLoader:
    return create_weather_dataloader(path, batch_size, Split.EVAL, cached, shuffle=False)


def get_test_dataloader_weather_dataset(path: str, batch_size: int, cached: bool) -> DataLoader:
    return create_weather_dataloader(path, batch_size, Split.TEST, cached, shuffle=False)