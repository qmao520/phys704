# Dataset Handler

## Outline

- [Overview](#Overview)
- [Dataset](#Dataset)
  - [Basic dataset structure](#Basic-dataset-structure)
  - [Dataset Creation](#Dataset-creation)
  - [Reading from a Dataset](#Reading-from-a-Dataset)
  - [Split a Dataset](#Split-a-Dataset)

## Overview

A **dataset** is a unified, standardized internal representation of the data. Models, feature extractors and all other components that need to interact with the data directly can leverage a common interface to perform read & write operations. This allows to write faster and more reliable code.

To represent a dataset we use **[H5py](https://www.h5py.org/)**, which is python interface to the HDF5 binary data format.
HDF5 is a Hierarchical Data Format and is designed to save and organize data.
___

## Dataset

### Hierarchical Dataset Structure

Every .hdf5 file defines a hierarchial structure. In our case a **dataset** has the following structure:

```txt
.
└── dataset                # data set information about the dataset as
│   │                      # attribute:
│   │                      # n_features, n_samples, valid
│   │
│   ├── sample_ids         # numpy string array containing the sample ids
│   │
│   ├── node_1(feature_1)  # numpy array of node_1(feature_1)
│   │
│   ├── ...                # additional features
│   │
│   └── node_n(feature_n)  # numpy array of node_n(feature_n)
│
└── metadata               # group containing the nodewise metadata
    │
    ├── node_1(feature_1)  # group containing the metadata of feature_1
    │   │
    │   ├── data_type      # metadata info (either single value or list)
    │   │
    │   └── ...            # additional metadata
    │
    └── ...                # additional metadata node groups
```

- **dataset:** (name is fixed) contains all features and sample_ids of the dataset. Additional it contains information about the dataset itself. The following information is available:
  - n_features (number of features present in the dataset)
  - n_samples (number of samples present in the dataset)
  - valid (boolean varaiable to indicate if the dataset is valid or not)

- **sample_ids:** (name is fixed) numpy string array that contains all the sample ids of the dataset.

    ```NOTE: The features(nodes) have to have the same order as the sample ids.```

- **feature_n:** (name can be set) numpy array that contains the data for feature n.

- **metadata**: hdf5 group containing node-wise the metadata.

___

### Dataset Creation

1) Creating an empty dataset:

    ```python
    import data_handler as dh

    with dh.get_Dataset_writer("path_to_hdf5_file.hdf5", 10) as dataset:
        pass
    ```

    These lines of code will create an empty dataset with length 10. And the new created hdf5 file only contains an empty dataset and metadata folder. The data set information associated with the dataset folder is:

    ```python
    dataset_info:{
        n_features: 0
        n_samples: 0
        valid: False
    }
    ```

2) Populating the dataset feature wise:

    ```python
    import data_handler as dh

    # Create a new hdf5 dataset using the DatasetWriter.
    with dh.get_Dataset_writer("path_to_hdf5_file.hdf5", 10) as dataset:

        # Populate the dataset with csv like data.
        csv_data = np.zeros((10), float)
        dataset.add_data_to_node("csv_data", csv_data)

        # Populate the dataset with image data (batch_size, image_shape).
        image_data = np.ones((10, 8, 8, 1), int)
        dataset.add_data_to_node("image_data", image_data)

        # Add metadata to the image node.
        metadata = {"data_shape": [8, 8, 1]}
        dataset.add_metadata_to_node("image_data", metadata)

        # Add sample_ids to the dataset.
        sample_ids = np.arange(1, 11).astype(str)
        dataset.add_data_to_node(dh.SAMPLE_IDS, sample_ids)
    ```

    These lines of code will populate the dataset with two features called `csv_data` and `image_data` as well as add a metadata information folder `image_data` containing `data_shape`. Additionally the mandatory `sample_ids` column is added at the end. The corresponding hdf5 file has the following structure:

    ```txt
    .
    └── dataset
    │   ├── sample_ids
    │   ├── csv_data
    │   └── image_data
    └── metadata
        └── image_data
            └── data_shape
    ```

3) Populating the dataset sample wise

    ```python
    import data_handler as dh

    # Create a new hdf5 dataset using the DatasetWriter.
    with dh.get_Dataset_writer("path_to_hdf5_file.hdf5", 10) as dataset:

        # Add a sample to the dataset.
        sample_data = {dh.SAMPLE_IDS: "sample_0",
                    "csv_data": 0.0,
                    "image_data": np.zeros((1, 8, 8, 1))}
        dataset.add_samples(sample_data)
    ```

    These lines of code will populate the dataset with a new sample.

4) Validating a dataset

    ```python
    import data_handler as dh

    # Create a new hdf5 dataset using the DatasetWriter.
    with dh.get_Dataset_writer("path_to_hdf5_file.hdf5", 10) as dataset:
        pass
        # The dataset gets validated automatically at the end of the with statement.
    # To check if the dataset is valid, use the following function:
    valid, msg = self.DatasetReader("path_to_hdf5_file.hdf5").get_validation_info()
    ```

    ```Note: Only a valid dataset can be used in the splitting functions.```

### Reading from a Dataset

1) Read data from a dataset

    ```python
    import data_handler as dh

    # Create a new hdf5 dataset using the DatasetReader.
    >>> dataset = dh.DatasetReader("path_to_hdf5_file.hdf5")

    # Get data set information.
    >>> dataset.get_data_info()
    {"n_features": 3,
     "n_samples": 10,
     "valid": True,}

    # Get all metadata.
    >>> dataset.get_all_metadata()
    {"image_data":{"data_shape": [8, 8, 1]}}

    # Get metadata of the `image_data` node.
    >>> dataset.get_metadata_of_node("image_data")
    {"data_shape": [8, 8, 1]}

    # Get a generator of all features/nodes with 1 samples in one batch.
    >>> data_generator = dataset.get_data_in_batches(batch=1)
    >>> list(data_generator)[0]
    {"__sample_ids__": "1",
     "csv_data": 0.0,
     "image_data": np.ones((1, 8, 8, 1))}

    # Get all data of all features/nodes.
    data = dataset.get_data_all()

    # Get a generator of one feature/nodes with 1 samples in one batch.
    >>> data_generator = dataset.get_data_of_node_in_batches("image_data",
                                                              batch=1)
    >>> list(data_generator)[0]
    np.ones((1, 8, 8, 1))

    # Get all data of one feature/nodes.
    data = dataset.get_data_of_node_all("node_name")

    # Get data for sample id.
    >>> dataset.get_data_for_sample_id("1")
    {"__sample_ids__": "1",
     "csv_data": 0.0,
     "image_data": np.ones((1, 8, 8, 1))}

    # Get the names of the nodes.
    >>> dataset.get_node_names()
    ["__sample_ids__", "csv_data", "image_data"]

    ```

### Split a Dataset

1) Split a dataset

    ```python

    # Create a new hdf5 dataset using the DatasetSplitter.
    dataset = dh.DatasetSplitter("path_to_hdf5_file.hdf5")

    # Split a dataset by node names.
    # csv_node is part of ds_1 all remaining nodes are part of ds_2.
    dataset.split_dataset_by_nodes(ds_path_1,
                                   ds_path_2,
                                   ["csv_data"])

    # Split a dataset by sample names.
    # ["0", "1"] is part of ds_1 all remaining nodes are part of ds_2.
    dataset.split_sample_wise(ds_path_1,
                              ds_path_2,
                              ["0", "1"])
    ```
