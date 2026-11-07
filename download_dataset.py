from pathlib import Path
from urllib.request import urlretrieve

URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
DESTINATION = Path(__file__).resolve().parent / "data" / "raw" / "insurance.csv"
DESTINATION.parent.mkdir(parents=True, exist_ok=True)

if DESTINATION.exists():
    print(f"Dataset already exists: {DESTINATION}")
else:
    print("Downloading the standard medical insurance charges dataset...")
    urlretrieve(URL, DESTINATION)
    print(f"Downloaded successfully: {DESTINATION}")
