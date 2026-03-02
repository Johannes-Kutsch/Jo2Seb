## Project Description

This is the first Capstone Project for the AI-Engineering Class mateflow-24.11.25 from neue fische.

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

## Save requirements

- Activate the virtual environment and create a /update the requirements.txt

### **`macOS`** type the following commands : 
    ```BASH
    source .venv/bin/activate
    pip freeze > requirements.txt
    ```

### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    .venv\Scripts\Activate.ps1
    pip freeze > requirements.txt
    ```

    For `Git-Bash` CLI :
    ```
    source .venv/bin/activate
    pip freeze > requirements.txt
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