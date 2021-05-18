# Narmax
This project is dedicated to translating the R codes from PUC-RIO's professor Helon for arx, armax, and narmax calculation.

# How to run
It is recommended to use a virtual environment to install the necessary libraries. Still, if you prefer, you can run it natively on your python global environment.

## Install python
**If you already have python installed, you can skip this step.**

### Linux
Python comes preinstalled on most Linux distributions and is available as a package on all others.

### Windows
Follow the steps in the [official documentation.](https://docs.python.org/3.9/using/windows.html)

### Verify the installation
On the command line for Linux or PowerShell for Windows, run the following command:
```Bash
python --version
```

## Create a python virtual environment
**If you already have or prefer not to use a virtual environment, you can skip this step.**
You can read the [official documentation](https://docs.python.org/3/library/venv.html) to create your virtual environment. If you don
But for simplicity, the command to create a simple virtual environment is laid below.
Create:
```Bash
python3 -m venv /path/to/new/virtual/environment
```
When using a virtual environment, it is necessary to activate it. To do so, use the following command:
```Bash
source <venv>/bin/activate
```

## Install necessary libraries
In the root folder of the project, run the following command (if using a virtual environment, remember to activate it first):
```Bash
pip install -r requirements.txt
```

## Run the examples:
Now you can use python to run the examples as you would run any other python script.

# Contributors
Anderson Rodrigues: makiander.abr@gmail.com

## Contributing
If you wish to contribute to the project, please feel free to fork it and make a pull request. All contributions are welcome, including translations to other languages.

# Future work
- Finish the translation from the R source code
- Include more examples from Stephen A Billings's book 'Nonlinear System Identification.'