from sksurv.datasets import load_flchain, load_whas500
import pandas as pd

def load_dataset(dataset_name: str) -> pd.DataFrame:
    '''
    Loads a Pandas DataFrame for the specified dataset containing the input variables and the output variable.

    The `censorship` column is a boolean that is False if censored (e.g., by end of study) and True if not (e.g., event occurred).
    The `event_time` column is a float representing the time until the event.

    Args:
        dataset_name (str): The name of a known dataset, or a filepath to a .csv file. If specifying a .csv file, the censorship and event_time column must exist.

    Returns:
        pd.DataFrame: The loaded dataset obeying the conventions of survival analysis (see function summary).

    Raises:
        NotImplementedError: Dataset name was unknown and the .csv file was not found.
    '''
    dataset = None
    if dataset_name=="flchain":
        dataset = pd.concat((load_flchain()[0],pd.DataFrame(load_flchain()[1])),axis=1)
        dataset.rename(columns={'death': 'censorship', 'futime': 'event_time'}, inplace=True)
    elif dataset_name=="whas500":
        dataset = pd.concat((load_whas500()[0],pd.DataFrame(load_whas500()[1])),axis=1)
        dataset.rename(columns={'fstat': 'censorship', 'lenfol': 'event_time'}, inplace=True)
    else:
        try:
            dataset = pd.read_csv(dataset_name)
        except:
            raise NotImplementedError(f"Dataset not found: {dataset_name}")
    return dataset

def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    '''
    Preprocesses a loaded survival analysis dataset

    Args:
        dataset (pd.DataFrame): The dataset to be preprocessed.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    '''
    return dataset
    raise NotImplementedError()