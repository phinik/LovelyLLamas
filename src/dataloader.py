from torch.utils.data import DataLoader

from src.dataset import WeatherDataset, Split, TransformationPipeline
from src.data_preprocessing import *

def get_train_dataloader_weather_dataset(path: str, batch_size: int, cached: bool) -> DataLoader:
    dset = WeatherDataset(
        path=path, 
        split=Split.TRAIN, 
        transformations=TransformationPipeline([
            ReplaceNaNs(),
            ReplaceCityName(),
            TokenizeUnits(),
            AssembleCustomOverview(),
            ReduceKeys()
        ]),
        cached=cached
    )

    return DataLoader(dset, batch_size=batch_size, shuffle=True)

def get_eval_dataloader_weather_dataset(path: str, batch_size: int, cached: bool) -> DataLoader:
    dset = WeatherDataset(
        path=path, 
        split=Split.EVAL,
        transformations=TransformationPipeline([
            ReplaceNaNs(),
            ReplaceCityName(),
            TokenizeUnits(),
            AssembleCustomOverview(),
            ReduceKeys()
        ]),
        cached=cached
    )

    return DataLoader(dset, batch_size=batch_size, shuffle=False)

def get_test_dataloader_weather_dataset(path: str, batch_size: int, cached: bool) -> DataLoader:
    dset = WeatherDataset(
        path=path, 
        split=Split.TEST,
        transformations=TransformationPipeline([
            ReplaceNaNs(),
            ReplaceCityName(),
            TokenizeUnits(),
            AssembleCustomOverview(),
            ReduceKeys()
        ]),
        cached=cached
    )

    return DataLoader(dset, batch_size=batch_size, shuffle=False)
