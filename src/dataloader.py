from torch.utils.data import DataLoader


from src.dataset import WeatherDataset, WeatherDatasetClassifier, Split, TransformationPipeline
from src.data_preprocessing import *


def get_train_dataloader_weather_dataset(
        path: str, 
        batch_size: int, 
        num_workers: int, 
        cached: bool, 
        n_samples: int = -1,
    ) -> DataLoader:
    dset = WeatherDataset(
        path=path, 
        split=Split.TRAIN, 
        transformations=TransformationPipeline([
            ReplaceNaNs(),
            ReplaceCityName(),
            TokenizeUnits(),
            AssembleFullOverview(),
            AssembleOverviewCTPC(),
            AssembleOverviewCTC(),
            AssembleOverviewCT(),
            AssembleOverviewTPWC(),
            ReduceKeys()
        ]),
        cached=cached,
        n_samples=n_samples
    )

    return DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def get_eval_dataloader_weather_dataset(path: str, batch_size: int, num_workers: int, cached: bool) -> DataLoader:
    dset = WeatherDataset(
        path=path, 
        split=Split.EVAL,
        transformations=TransformationPipeline([
            ReplaceNaNs(),
            ReplaceCityName(),
            TokenizeUnits(),
            AssembleFullOverview(),
            AssembleOverviewCTPC(),
            AssembleOverviewCTC(),
            AssembleOverviewCT(),
            AssembleOverviewTPWC(),
            ReduceKeys()
        ]),
        cached=cached
    )

    return DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_test_dataloader_weather_dataset(path: str, batch_size: int, cached: bool) -> DataLoader:
    dset = WeatherDataset(
        path=path, 
        split=Split.TEST,
        transformations=TransformationPipeline([
            ReplaceNaNs(),
            ReplaceCityName(),
            TokenizeUnits(),
            AssembleFullOverview(),
            AssembleOverviewCTPC(),
            AssembleOverviewCTC(),
            AssembleOverviewCT(),
            AssembleOverviewTPWC(),
            ReduceKeys()
        ]),
        cached=cached
    )

    return DataLoader(dset, batch_size=batch_size, shuffle=True)


def get_train_dataloader_weather_dataset_classifier(path: str, batch_size: int, num_workers: int) -> DataLoader:
    dset = WeatherDatasetClassifier(
        path=path, 
        split=Split.TRAIN, 
        transformations=TransformationPipeline([
            ReplaceNaNs(),
            ReplaceCityName(),
            TokenizeUnits(),
            AssembleFullOverview(),
            AssembleOverviewCTPC(),
            AssembleOverviewCTC(),
            AssembleOverviewCT(),
            AssembleOverviewTPWC(),
            ReduceKeys()
        ])
    )

    return DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def get_eval_dataloader_weather_dataset_classifier(path: str, batch_size: int, num_workers: int) -> DataLoader:
    dset = WeatherDatasetClassifier(
        path=path, 
        split=Split.EVAL,
        transformations=TransformationPipeline([
            ReplaceNaNs(),
            ReplaceCityName(),
            TokenizeUnits(),
            AssembleFullOverview(),
            AssembleOverviewCTPC(),
            AssembleOverviewCTC(),
            AssembleOverviewCT(),
            AssembleOverviewTPWC(),
            ReduceKeys()
        ])
    )

    return DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_demo_weather_dataset(path: str) -> WeatherDataset:
    return WeatherDataset(
        path=path, 
        split=Split.TEST,
        transformations=TransformationPipeline([
            ReplaceNaNs(),
            ReplaceCityName(),
            TokenizeUnits(),
            AssembleFullOverview(),
            AssembleOverviewCTPC(),
            AssembleOverviewCTC(),
            AssembleOverviewCT(),
            AssembleOverviewTPWC(),
            ReduceKeys()
        ]),
        cached=True
    )