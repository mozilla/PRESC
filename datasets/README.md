# Classification datasets

File | Description | Source link (with details) | Preprocessing applied | Label column
---|---|---|---|---
`generated.csv` | Automatically-generated dataset containing data samples separated into very well-delineated categories. This can be considered a "best-case scenario" test case. | | | `label`
`defaults.csv` | Defaults on credit card payments | [UCI](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#) | Minor (column name reformatting) | `defaulted`
`winequality.csv` | Quality ratings of Portuguese white wines | [UCI](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) | Added binarized label column `recommend` indicating `quality >= 7` | `recommend`
`vehicles.csv` | Recognizing vehicle type from its silhouette | [OpenML](https://www.openml.org/d/54) | None | `Class`
`eeg.csv` | EEG eye state measurements | [OpenML](https://www.openml.org/d/1471) | Dropped a few outlier rows | `Class`
`kick_starter.csv` | Kick stater project state | [Kaggle](https://www.kaggle.com/kemical/kickstarter-projects) | Dropped unnamed columns; Minor column name reformatting; Calculated duration of the project and dropped start and end dates; Dropped some rows with wrong input type; Dropped *main category* column and kept *category* column; randomply sampled 30% of the data; Filled NA with 0 for numeric values | `state`
`mushrooms.csv` | Classification mushrooms edibility based on physical features | [UCI](https://archive.ics.uci.edu/ml/datasets/Mushroom) |Renamed the column `class` to `edibility` for descriptiveness| `edibility` 
`Surgical-deepnet.csv`| Surgical cases related to complication |  [Kaggle](https://www.kaggle.com/omnamahshivai/surgical-dataset-binary-classification) | None | `complication`
`gender_classification.csv`| use hobbies to guess gender |  [Kaggle](https://www.kaggle.com/hb20007/gender-classification) | None | `Gender`


These can all be loaded using Pandas:

```python
import pandas as pd
dataset = pd.read_csv("file.csv")
```
