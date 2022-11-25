# Coursework 1 : Introduction to Machine Learning - Artificial Neural Networks

This project was carried out in a school context by four classmates. 

* Pierre-Antoine ARSAGUET : pea22@ic.ac.uk
* Louis BERTHIER : ldb22@ic.ac.uk
* Nikita DMITRIEFF : nnd19@ic.ac.uk
* Ryan EL KHOURY : re122@ic.ac.uk

Here are the steps to follow to ensure that the python environment is up to date and execute the script properly:

1. To move to the folder containing our script:
```python
cd Path_file_to_the_downloaded_folder/Neural_Networks_095
```

2. To use the same versions of the libraries as we do:
```python
pip install -r requirements.txt
```

3. To run our script, open Jupyter otebook and access our code:
```python
python part2_house_value_regression.py
```

Normally a simulation (training and test) is started for the model with the optimal hyperparameters defined in the report (in particular a progress bar of 50 epochs should also be displayed). 
However, it is possible to modify some general parameters defined in the 'example_main' function in order to modify the training (number of epochs, number of batches, number of hidden layers, train/test proportion).
