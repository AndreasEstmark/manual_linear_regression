#  how to find your python versions on windows: run in ps: py --list

# find location of python version: py -3.11 -c "import sys; print(sys.executable)"
with output: ....python.exe


Quick way to add them using PowerShell
Run this command:

powershell
Copy
Edit
$pythonPath = ""
$scriptsPath = ""

$currentPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
$newPath = $currentPath + ";$pythonPath;$scriptsPath"

[System.Environment]::SetEnvironmentVariable("Path", $newPath, "User")


# Make sure to disable pyton aliases in windows app execution 


then write pip install poetry

and then poetry init

This command will guide you through creating your pyproject.toml config.

Package name [regression_analysis]:  regression_analysis
Version [0.1.0]:  1
Description []:  regression analysis
Author [>, n to skip]:  n
License []:  
Compatible Python versions [>=3.11]:  

Would you like to define your main dependencies interactively? (yes/no) [yes] y
        You can specify a package in the following forms:
          - A single name (requests): this will search for matches on PyPI
          - A name and a constraint (requests@^2.23.0)
          - A git url (git+https://github.com/python-poetry/poetry.git)
          - A git url with a revision         (git+https://github.com/python-poetry/poetry.git#develop)
          - A file path (../my-package/my-package.whl)
          - A directory (../my-package/)
          - A url (https://example.com/packages/my-package-0.1.0.tar.gz)

Package to add or search for (leave blank to skip):

Would you like to define your development dependencies interactively? (yes/no) [yes] no
Generated file

[project]
name = "regression-analysis"
version = "1"
description = "regression analysis"
authors = [
    {name = "Your Name",email = "you@example.com"}
]
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
]


[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"


Do you confirm generation? (yes/no) [yes] yes
PS > poetry --version
Poetry (version 2.1.3)
PS > poetry add numpy pandas
>>
Creating virtualenv ....-py3.11 in ....virtualenvs
Using version ^2.3.1 for numpy
Using version ^2.3.1 for pandas

Updating dependencies
Resolving dependencies... (0.5s)

Package operations: 6 installs, 0 updates, 0 removals

  - Installing six (1.17.0)
  - Installing numpy (2.3.1)
  - Installing python-dateutil (2.9.0.post0)
  - Installing pytz (2025.2)
  - Installing tzdata (2025.2)
  - Installing pandas (2.3.1)

Writing lock file
PS ...Regression_analysis> 


# notebook
then run: poetry add notebook so you can notebooks. 

# then to get the data:
poetry add kagglehub

Yes—kagglehub.dataset_download() downloads the dataset to your machine.


