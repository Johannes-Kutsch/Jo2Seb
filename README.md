## Project Description

A team project focused on forecasting daily PM10 particulate matter levels at Hamburg Harbour. We combined air quality and weather data from public APIs (UBA, DWD, Copernicus) into a forecasting pipeline with experiment tracking via MLflow. Our best model significantly outperformed the baseline (R² 0.21 → 0.40).

Presentation: https://docs.google.com/presentation/d/1do9ZqPXh45yNRQ26nqWb1ITjbVk9BWNX99ijikiTROc/edit?usp=sharing

## Set up your Environment

Please make sure you have forked the repo and set up a new virtual environment. For this purpose you can use the following commands:

The added [requirements file](requirements.txt) contains all libraries and dependencies we need to execute the Gradient Descent notebooks.

*Note: If there are errors during environment setup, try removing the versions from the failing packages in the requirements file. M1 shizzle.*

### **`macOS`** type the following commands : 

- Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-Bash` CLI :
    ```
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

## GIT LFS

- We use git lfs to store data and models

### Setup

    For `Git-Bash` CLI :
    ```
    git lfs install
    git lfs version

    git lfs track "files/models/**"
    git lfs track "files/pipelines/**"
    git lfs track "files/data/lfs/**"
    ```

### Use

    For `Git-Bash` CLI :
    ```
    git lfs install
    git lfs version

    git pull
    git lfs pull  
    ```
