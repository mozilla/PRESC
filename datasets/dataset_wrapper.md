
# Dataset Wrapper API #

## Qeustions for David ##

1. Do we just keep everything in the wrapper like how my implementation is now? Or just write some sort of getter functions to return them?
2. Am I on the right track?
3. I noticed that the "quality" column was dropped for the dataset used for misclassification example, do we need to drop more of the data columns for the datasets in the datasets/ folder? Or are they ready to go?

4. For each dataset, do I need to implement a feature processor/transformer for each individual dataset?


>**For convenience we should wrap test datasets in an object that offers a common API.** -- David

## Example functionalities ##

- [ ] reading and writing from files
- [ ] accessing feature columns
- [ ] label column accessing pre-split train/test sets
- [ ] TBD

 > **That way we don't need to pass around information about column names or indexing separately. </br> There should also be methods for reading and writing from files.** -- David

## Requirements for Methods ##

1. reading and writing from files:
    - [ ] download them from their source location and apply any preprocessing
