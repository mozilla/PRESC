# Classification datasets

File | Description | Source link (with details) | Preprocessing applied | Label column
---|---|---|---|---
`generated.csv` | Automatically-generated dataset containing data samples separated into very well-delineated categories. This can be considered a "best-case scenario" test case. | | | `label`
`defaults.csv` | Defaults on credit card payments | [UCI](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#) | Minor (column name reformatting) | `defaulted`
`winequality.csv` | Quality ratings of Portuguese white wines | [UCI](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) | Added binarized label column `recommend` indicating `quality >= 7` | `recommend`
`vehicles.csv` | Recognizing vehicle type from its silhouette | [OpenML](https://www.openml.org/d/54) | None | `Class`
`eeg.csv` | EEG eye state measurements | [OpenML](https://www.openml.org/d/1471) | Dropped a few outlier rows | `Class`

These can all be loaded using Pandas:

```python
import pandas as pd
dataset = pd.read_csv("file.csv")
```
