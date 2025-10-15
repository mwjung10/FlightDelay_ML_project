**How to use the code?**

First do:
pip install -r requirements.txt

Ensure all packages and libraries install properly.

Then run the entire data_preprocess.ipynb notebook.

That will make sure that in the data folder you will have the preprocessed_flight_data.csv
This is the main data file to use.

In load_data.py there are functions that load the requried columns and prepare X, y matrices for you. Use it like that in your .ipynb file (in the classification folder!!!):

from load_data import load_preprocessed_data

X, y = load_preprocessed_data()


Make your classifier in the classification folder, use a .ipynb notebook.

For evaluation use the evaluation.py file with functions that will help you to do it. Please use them for consistency between us.