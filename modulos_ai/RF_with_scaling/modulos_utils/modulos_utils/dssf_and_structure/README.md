# DSSF / Structure

## Use

The goal of this code is to read the dataset, determine if it is valid and
save it to the internal dataset.

## Structure

There are three class (families) doing the job:

- **DatasetSaver**: Point of entry (same as the SaveStructure class in the old implementation), uses DSSFileInterpreter to validate the `dataset_structure.json` and to receive an Component object for each component. It uses this to load and save the data in batches.
- **DSSFileInterpreter**: Validates the `dataset_structure.json`, fills missing values with default values and initiates the component class objects.
- **Component**: Family of classes for the different component types. They check the optional keys, read the data and yield / saves it.
