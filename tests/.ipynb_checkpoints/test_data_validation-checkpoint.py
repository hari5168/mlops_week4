import pandas as pd

def test_data_shape():
    df = pd.read_csv("data/iris.csv")
    # Expecting 150 rows and 5 columns
    assert df.shape == (150, 5)

def test_no_missing_values():
    df = pd.read_csv("data/iris.csv")
    assert not df.isnull().values.any(), "Dataset contains missing values"

def test_valid_species():
    df = pd.read_csv("data/iris.csv")
    valid_species = {"setosa", "versicolor", "virginica"}
    assert set(df["species"].unique()) <= valid_species
