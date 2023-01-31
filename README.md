# contextInference
In this repository you can find all the code necessary to run all the simulations and reproduce the figures in the submitted paper. This is only a temporary repository, which will be replaced by a permanent one before the manuscript is published.

## Packages
The code was done in python 3.9, and the following packages are required to run the code:
- numpy
- scipy
- matplotlib
- pandas
- seaborn

To see the exact versions of the packages used during our simulations, refer to the anaconda_environment.txt and anaconda_environment.yml files. For information on how to use these files, see [this link on managing anaconda environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Note that the anaconda_environment.txt file is OS-specific (made with Linux), while the anaconda_environment.yml is not. Depending on the operating system, the yml method might not be able to reproduce the exact version numbers as used in our system. The code should run with the latest versions of these packages too, but the results might be different.

Due to recent changes in seaborn, versions older than 0.12 will raise an exception and cannot be used.

## Executing the code
The file figures.py contains all the functions necessary to reproduce all the simulations and figures in the manuscript. Note, however, that only the simulated data is plotted by this functions; all experimental results were added after plotting using inkscape.

Once a python environment is available with the required packages, the simplest way to reproduce our results is to execute the figures.py file directly (e.g. running "python figures.py"). This could take some time to run (5 to 10 minutes in our system), but all figures from the manuscript will be plotted. Make sure to set the working directory to that in which the files are, or to add it to your python path.

Individual figures can be plotted too by running the corresponding function in figures.py. The function names correspond to the experimental studies being simulated (e.g. davidson_2004()). The functions have a parameter called "fignum" with default values that match their labels in the manuscript; for example, davidson_2004() has a default value of 4, and it is shown as Figure 4 in the manuscript. To run these, import figures from any interactive shell (python, ipython, jupyter) and run e.g. figures.davidson_2004(), without any parameters.

## Files
All simulations are done by functions in simulations.py. There is a function in simulations.py for each of the figures in the manuscript, using the same names as the functions in figures.py (with the exception of figures.oh_2019_kim_2015(), which is split in simulations.py. All simulation parameters are set in the functions in simulations.py, while those in figures.py are mostly visual formatting.

The file task_hold_hand contains the class that simulates the tasks. The simulation of the tasks works in an abstract Error space and is the same for all simulated experiments (using only different parameters). This class is not meant to be used directly and is therefore simply imported by figures.py and used internally. To see the parameters used for simulating the different experimental tasks, see the corresponding functions in simulations.py.
