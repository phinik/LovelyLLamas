from torch.utils.data import DataLoader


from src.dataset import WeatherDataset, Split, TransformationPipeline
from src.data_preprocessing import *


def get_train_dataloader_weather_dataset(
        path: str, 
        batch_size: int, 
        num_workers: int, 
        cached: bool, 
        n_samples: int = -1,
        overview: str = "full"
    ) -> DataLoader:
    dset = WeatherDataset(
        path=path, 
        split=Split.TRAIN, 
        transformations=TransformationPipeline([
            ReplaceNaNs(),
            ReplaceCityName(),
            TokenizeUnits(),
            OverviewFactory.get(overview),
            ReduceKeys()
        ]),
        cached=cached,
        n_samples=n_samples
    )

    return DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def get_eval_dataloader_weather_dataset(
        path: str, 
        batch_size: int, 
        num_workers: int, 
        cached: bool, 
        overview: str = "full"
    ) -> DataLoader:
    dset = WeatherDataset(
        path=path, 
        split=Split.EVAL,
        transformations=TransformationPipeline([
            ReplaceNaNs(),
            ReplaceCityName(),
            TokenizeUnits(),
            OverviewFactory.get(overview),
            ReduceKeys()
        ]),
        cached=cached
    )

    return DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def get_test_dataloader_weather_dataset(path: str, batch_size: int, cached: bool, overview: str = "full") -> DataLoader:
    dset = WeatherDataset(
        path=path, 
        split=Split.TEST,
        transformations=TransformationPipeline([
            ReplaceNaNs(),
            ReplaceCityName(),
            TokenizeUnits(),
            OverviewFactory.get(overview),
            ReduceKeys()
        ]),
        cached=cached
    )

    return DataLoader(dset, batch_size=batch_size, shuffle=True)


def get_demo_weather_dataset(path: str, overview: str = "full") -> WeatherDataset:
    return WeatherDataset(
        path=path, 
        split=Split.TEST,
        transformations=TransformationPipeline([
            ReplaceNaNs(),
            ReplaceCityName(),
            TokenizeUnits(),
            OverviewFactory.get(overview),
            ReduceKeys()
        ]),
        cached=True
    )
