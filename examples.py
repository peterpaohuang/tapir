# Example use cases for depablo_box

############################################################################
#							Start to end use case
############################################################################
"""I want to calculate different chemical properties and perform machine learning on them"""
from depablo_box import PDBML, model

# initialize class
dx = PDBML()

# access depablo_box database as pandas dataframe
df = dx.df

# say we want to calculate these descriptors and add to dataframe
descriptor_list = ['BalabanJ', 'BertzCT', 'Ipc', 'HallKierAlpha', 'MolLogP', 'MolMR']
dx.add_descriptors(descriptor_list)

# see if there are any correlations between the above properties
dx.correlation_map(descriptor_list)

# define model arguments
input_properties = descriptor_list
output_property = "glass_transition_temperature"
na_strategy = "remove" # options: 'mean', 'remove' - default is 'remove'

# initialize model
ml = model(df, input_properties, output_property, na_strategy=na_strategy)

# start training
algorithm = "Support Vector Regression"
ml.train(algorithm)

# view model results
print("YOUR MODEL R^2 SCORE: {}".format(ml.r_2))
ml.feature_importances() # plot feature importances

# Predict on new data that follows the correponding index placement of the descriptor_list 
# new_data is [[Any], [Any]] where each array inside represents a single row of input
new_data = [["10.5", "29", "102.1", "91.2", "1.1", "0.15"]]
results = ml.predict(new_data)
print(results)

# Finally, export fitted model as pickle file
outpath = "Tg_prediction_model.pickle"
ml.export_fitted_model(outpath)

############################################################################
#							Experiment use case
############################################################################
from depablo_box import PDBML, model

# initialize class
dx = PDBML()

# check if number of radical electrons is already in dataframe
df = dx.df
print(list(df))

# it seems like this property hasn't been added yet, 
# so we add number of radical electrons for each polymer to dataframe 
descriptor_list = ["NumValenceElectrons"]
dx.add_descriptors(descriptor_list)

# plot number of radical electrons against solubility parameters as a scatterplot
dx.plot_properties(property_x="solubility_parameter", property_y=descriptor_list[0]) 

# get correlation between number of radical electrons against solubility parameters
dx.property_correlation("solubility_parameter", "NumValenceElectrons")

"""hmm... there's doesn't seem to be much of a correlation.
I then want to know what the maximum predictive power is, so I use a
support vector machine regression to fit a function that as maximally predictive"""

# define model arguments
input_properties = descriptor_list
output_property = "solubility_parameter"
na_strategy = "remove" # options: 'mean', 'remove' - default is 'remove'

# initialize model
ml = model(df, input_properties, output_property, na_strategy=na_strategy)

# start training
algorithm = "Lasso Regression"
ml.train(algorithm)

# see the predictiveness of the model
print("YOUR MODEL R^2 SCORE: {}".format(ml.r_2))

"""It turns out there is a very good correlation between the two, so I now want
to save this model as a pickle file"""
# export fitted model 
outpath = "Tg_prediction_model.pickle"
ml.export_fitted_model(outpath)

# load fitted model 
import pickle
with open(outpath, "rb") as f:
  ml = pickle.load(f) # ml = sklearn regressor class
new_data = [[1000]]
results = ml.predict(new_data)
print("YOUR RESULTS: {}".format(results))


############################################################################
#							Generate Quantum Chemistry Files
############################################################################

"""I want to convert SMILES format into some other format (maybe input files for quantum chemistry codes)"""

from depablo_box import PDBML

# initialize class
dx = PDBML()

# list all supported conversions
print(dx.conversion_formats)

# either SMILES format or polymer name
polymer_identifier = 'C=CC(=O)NC(C)C'
conversion_format = 'Gaussian 98/03 Input'
outpath = '/file/path/your_polymer.xyz'

# writes gaussian codes to outpath
dx.create_input_file(polymer_identifier, conversion_format, outpath)

