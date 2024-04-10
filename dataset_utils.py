import pandas as pd
from sksurv.datasets import load_flchain, load_whas500
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_dataset(dataset_name: str) -> pd.DataFrame:
    '''
    Loads a Pandas DataFrame for the specified dataset containing the input variables and the output variable.

    The `event_indicator` column is a boolean that is False if censored (e.g., by end of study) and True if not (e.g., event occurred).
    The `event_time` column is a float representing the time until the event.

    Args:
        dataset_name (str): The name of a known dataset, or a filepath to a .csv file. If specifying a .csv file, the "event_indicator" and "event_time" column must exist.

    Returns:
        pd.DataFrame: The loaded dataset obeying the conventions of survival analysis (see function summary).

    Raises:
        NotImplementedError: Dataset name was unknown and the .csv file was not found.
    '''
    dataset = None
    # Load built-in dataset
    if dataset_name=="flchain":
        dataset = pd.concat((load_flchain()[0],pd.DataFrame(load_flchain()[1])),axis=1)
        dataset.rename(columns={'death': 'event_indicator', 'futime': 'event_time'}, inplace=True)
    elif dataset_name=="whas500":
        dataset = pd.concat((load_whas500()[0],pd.DataFrame(load_whas500()[1])),axis=1)
        dataset.rename(columns={'fstat': 'event_indicator', 'lenfol': 'event_time'}, inplace=True)
    # Load user-specified dataset
    else:
        try:
            dataset = pd.read_csv(dataset_name)
        except:
            raise NotImplementedError(f"Dataset not found: {dataset_name}")
        if "event_indicator" not in dataset.columns or "event_time" not in dataset.columns:
            raise ValueError("User-specified .csv must contain a 'event_indicator' and 'event_time' column.")
    return dataset

def preprocess_dataset(dataset: pd.DataFrame, ) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    '''
    Preprocesses a loaded survival analysis dataset

    Args:
        dataset (pd.DataFrame): The dataset to be preprocessed.

    Returns:
        X_train, X_test, Y_train, Y_test: The preprocessed datasets and their splits.
    '''
    # Split X and Y
    X = dataset[dataset.columns.difference(['event_indicator', 'event_time'])]
    Y = dataset[["event_indicator","event_time"]]

    # Train Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=226,shuffle=True)

    # Some Default Preprocessing (missing imputation and either onehot or scaling)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X.select_dtypes(include=['number']).columns
    transformer = Pipeline([
        ('preprocess', ColumnTransformer([
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)),
                ('onehot', OneHotEncoder(handle_unknown='ignore',sparse_output=False))
            ]), categorical_cols),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean', add_indicator=True)),
                ('scaler', StandardScaler())
            ]), numeric_cols)
        ])),
    ]).set_output(transform="pandas").fit(X_train)
    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)

    return X_train, X_test, Y_train, Y_test
    