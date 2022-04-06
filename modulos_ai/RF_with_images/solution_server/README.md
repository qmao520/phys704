# Modulos Solution Server

 The `solution-server` is a python package that wraps a flask app around Modulos AutoML solutions. The solution server performs inference (on new data samples) with trained weights and can be accessed through a REST API. This package contains the definition of the solution REST API and the flask app that serves the API.

It is important to note that the server package is independent of the model and the feature extractor that is used in
the solution. Also, it is independent of the dataset that was used to generate the solution. Therefore it does not have to be re-installed if the user runs a different solution (assuming the requirements of all modules have been installed initially). However, when the server is started, it relies on the two input parameters `--src` and `--config` which are indeed dependent on the model, the feature extractor, and the dataset. Therefore, to test `k` different solutions, one needs to start the server `k` times with different arguments `--src` and `--config`.

## Directory structure
```
.
├── solution_server
│   ├── solution_server
│   ├── README.md
│   ├── setup.py
│   └── requirements.txt
|   └── doc
│       └── rest_api.yml
```
## Install the server
Note: It is in general not necessary to manually install the solution server. Please check out the `README.html` of the Modulos AutoML solution for the installation options for the solution. The solution server should always be installed using
the main installation script.

## Start the server
Before starting the server, make sure you have downloaded and untarred the Modulos AutoML solution that you want to run with the server. Every solution downloaded from the platform, contains a config file (`solution_server_config.yml`). The server parses this file such that it knows what the input data look like.

The server can be started with the following command:

```
solution-server start --src <path to src dir> --config <path to config file>
```
The src dir can be found in each solution directory and it contains the weights of the model and the feature extractor
that are used in the respective solution.

Alternatively, the arguments can also be omitted. In this case, the server assumes that the config file and the weights are in the same folder.

```
solution-server start
```

## Make a request
The server can be accessed through the REST API with standard clients. For more information, please check out the `rest_api.yml` in the folder `solution_server/doc`, or the `README.html` of any Modulos AutoML solution.



