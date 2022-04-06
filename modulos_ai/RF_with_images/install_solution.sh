#!/bin/bash
# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
# This script updates and installs the requirements in the solution.
SERVER_REQUIREMENTS_PATH="solution_server/requirements.txt"
IS_TIME_SERIES_SOLUTION=false
# Check whether the solution is a time series solution.
if test -f "$SERVER_REQUIREMENTS_PATH"; then
    IS_TIME_SERIES_SOLUTION=false
else
    IS_TIME_SERIES_SOLUTION=true
fi
MODULES_REQUIREMENTS_PATH="src/modules/requirements.txt"
MODULES_REQUIREMENTS_MODIFIED_PATH="src/modules/requirements-modified.txt"
ALL_MODULES_REQUIREMENTS_PATH="src/modules/requirements-all-modules.txt"
ALL_MODULES_REQUIREMENTS_MODIFIED_PATH="src/modules/requirements-all-modules-modified.txt"
ARG0=$(basename "$0" .sh)
INSTALL_ALL_MODULES=false
INSTALL_SERVER=false
INSTALL_GPU=false
INSTALL_MODULOS_UTILS=false
INSTALL_NECESSARY_MODULES=false
OVERRIDE_WARNING="
Warning: The flag '-n' is overridden by the flag '-a'.
"

# Define the help message for both the time series case and the standard case.
if [ $IS_TIME_SERIES_SOLUTION = true ] ; then
    USAGE_MESSAGE="This script installs the requirements of the solution. Per default,
    it installs the following things:

    - The modulos_utils library.
    - The modules that are used by the solution.

    If you want more control, you can specify exactly what to install with
    the following arguments: "
else
    USAGE_MESSAGE="This script installs the requirements of the solution. Per default,
    it installs the following things:

    - The modulos_utils library.
    - The solution_server library.
    - The modules that are used by the solution.

    If you want more control, you can specify exactly what to install with
    the following arguments: "
fi


help()
{
    echo
    echo "$USAGE_MESSAGE"
    echo
    echo "  -a      Install all modules that are available by the AutoML platform.
                    Note that this flag overrides the flag '-n'."
    echo "  -n      Install only the modules that are used by this solution.
                    Note that this flag is overridden by the flag '-a'."
    echo "  -m      Install the modulos_utils library."
    echo "  -g      Install the gpu version of pytorch. Note that on MacOS we
                    currently do not support GPUS, hence this flag is ignored."
    if [ $IS_TIME_SERIES_SOLUTION = false ] ; then
        echo "  -s      Install the solution_server library."
    fi
    echo "  -h      Print this help message and exit."
    echo
    exit 0
}



while getopts anmsgh opt; do
    case "$opt" in
        a) INSTALL_ALL_MODULES=true ;;
        n) INSTALL_NECESSARY_MODULES=true ;;
        m) INSTALL_MODULOS_UTILS=true ;;
        s) INSTALL_SERVER=true ;;
        g) INSTALL_GPU=true ;;
        h) help;;
        *) echo "Error in parsing options."
           exit 1
    esac
done

# Error in the case where -s is used in a time series solution.
if [ $INSTALL_SERVER = true ] && [ $IS_TIME_SERIES_SOLUTION = true ] ; then
    echo "install_solution.sh: illegal option -- s"
    echo "Error in parsing options."
    exit
fi

# Define the default behaviour for the case where no flag is used.
if [ $INSTALL_ALL_MODULES = false ] && [ $INSTALL_NECESSARY_MODULES = false ] && [ $INSTALL_MODULOS_UTILS = false ] && [ $INSTALL_SERVER = false ] ; then
    INSTALL_NECESSARY_MODULES=true
    INSTALL_MODULOS_UTILS=true
    INSTALL_SERVER=true
fi

# Make sure INSTALL_SERVER is not set to true in the time series case.
if [ $IS_TIME_SERIES_SOLUTION = true ] ; then
    INSTALL_SERVER=false
fi


function update_req_file {
    local req_file="$1"
    if [[ "$OSTYPE" == "darwin"* ]] ; then
        echo "Updating requirements for MacOS from: $req_file"

        # Remove the package index options related to torch.
        sed -i "" "/^--find-links.*torch.*/d" "$req_file"

        # Regex matching versions like:
        #   > torch==99.1
        #   > torch_anything  >= 1.0.100
        sed -E -i "" "s/(^torch.*[ ]*[=|>|<]=[ ]*[0-9]+\.[0-9]+[\.]*[0-9]*).*/\1/" "$req_file"
    elif [[ $INSTALL_GPU = false ]] ; then
        echo "Removing CUDA from requirements at: $req_file"
        # Regex matching versions like:
        #   > torch==99.1
        #   > torch_anything  >= 1.0.100
        sed -E -i "s/(^torch.*[ ]*[=|>|<]=[ ]*[0-9]+\.[0-9]+[\.]*[0-9]*).*/\1+cpu/" "$req_file"
    fi
}

# Raise a warning, if -a and -n are used.
if [ $INSTALL_ALL_MODULES = true ] && [ $INSTALL_NECESSARY_MODULES = true ] ; then
    echo "$OVERRIDE_WARNING"
fi

# Install the requirements.
printf "\nInstalling pip and setuptools ...\n"
python3 -m pip install -U pip
python3 -m pip install -U setuptools


if [ $INSTALL_ALL_MODULES = true ] ; then
    printf "\nInstalling all modules ...\n"
    cp $ALL_MODULES_REQUIREMENTS_PATH $ALL_MODULES_REQUIREMENTS_MODIFIED_PATH
    update_req_file $ALL_MODULES_REQUIREMENTS_MODIFIED_PATH
    python3 -m pip install -r $ALL_MODULES_REQUIREMENTS_MODIFIED_PATH
else
    if [ $INSTALL_NECESSARY_MODULES = true ] ; then
        printf "\nInstalling the necessary modules ...\n"
        cp $MODULES_REQUIREMENTS_PATH $MODULES_REQUIREMENTS_MODIFIED_PATH
        update_req_file $MODULES_REQUIREMENTS_MODIFIED_PATH
        python3 -m pip install -r $MODULES_REQUIREMENTS_MODIFIED_PATH
    fi
fi

if [ $INSTALL_MODULOS_UTILS = true ] || [ $INSTALL_SERVER = true ] ; then
    printf "\nInstalling modulos_utils ...\n"
    python3 -m pip install -e ./modulos_utils
fi

if [ $INSTALL_SERVER = true ] ; then
    printf "\nInstalling the solution server ...\n"
    python3 -m pip install -e ./solution_server
fi

printf "\nInstallation successful.\n"
