import pandas as pd
import numpy as np
import ast
import re
import pickle

import matplotlib

# test different image backends for matplotlib
gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
for gui in gui_env:
	try:
		matplotlib.use(gui,warn=False, force=True)
		from matplotlib import pyplot as plt
		break
	except:
		continue

import seaborn as sns
from depablo_box.main.utils import NA_encoder,calculate_descriptor, generate_input_files

class PDBML:
	def __init__(self, na_values=["na", ""]):
		"""
		Parameters
		------------------------
		na_values: [String]
			values that are used to signify NaN values
		"""


		self.df = pd.read_csv('depablo_box/polymer_db.csv').replace(na_values, np.nan)
		# self.df.set_index(["polymer_name"], inplace=True) 
		self.experimental_descriptors = ["molar_volume", "density",
         "solubility_parameter","molar_cohesive_energy", "glass_transition_temperature", "molar_heat_capacity", 
            "entanglement_molecular_weight", "refraction_index", "thermal_expansion_coefficient", 
            "repeat_unit_weight", "waals_volume"]
		self.chemical_descriptors = ['ExactMolWt', 'FpDensityMorgan1', 
		'FpDensityMorgan2', 'FpDensityMorgan3', 'HeavyAtomMolWt', 'MolWt', 'NumRadicalElectrons', 
		'NumValenceElectrons', 'BalabanJ', 'BertzCT', 'Ipc', 'HallKierAlpha', 'MolLogP', 'MolMR', 'HeavyAtomCount', 
		'NHOHCount', 'NOCount', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'RingCount', 
		'FractionCSP3', 'TPSA']
		self.conversion_formats = ["Protein Data Bank", "Gaussian 98/03 Input"]
		self.ml_methods = ["Support Vector Regression", "Linear Regression", "Ridge Regression", 
		"Lasso Regression", "Gaussian Process Regression"]
		# set index as polymer name rather than integer - each name is unique

		self.pd = pd

	def get_smiles_from_identifier(self, polymer_identifier):
		"""
		Standarize format of either polymer name or polymer smiles into only smiles format

		Parameters
		------------------------
		polymer_identifier: String
			Unique identifier of chemical (either polymer name or smiles)

		Return
		------------------------
		smiles: String
			polymer's smiles format
		"""
		if polymer_identifier in self.df["smiles"].tolist(): # if polymer_identifier is smiles
			smiles = polymer_identifier
		elif polymer_identifier in self.df["polymer_name"].tolist(): # if polymer_identifier is polymer_name
			smiles = self.df.loc[self.df["polymer_name"] == polymer_identifier]["smiles"].tolist()[0]
		else:
			raise KeyError("Your input did not match any polymer in our database")

		return smiles
	def create_input_file(self, polymer_identifier, format, outpath):
		"""
		Generate input file for quantum chemistry codes

		Parameters
		------------------------
		polymer_identifier: String
			Unique identifier of chemical (either polymer name or smiles)
		format: String
			File format you want to generate from polymer_identifier
		outpath: String
			file path to write result to
		"""

		smiles = self.get_smiles_from_identifier(polymer_identifier)
		
		result = generate_input_files(smiles, format)
		with open(outpath, 'w+') as f:
			f.write(result)

	def get_descriptors(self, polymer_identifier, descriptor_list):
		"""
		Generate properties for single chemical 

		Parameters
		------------------------
		polymer_identifier: String
			Unique identifier of chemical (either polymer name or smiles)

		descriptor_list: [String]
			List of descriptors

		Returns
		-------------------------
		single_row_df: DataFrame
			one chemical dataframe with each column representing a generated descriptor based on descriptor_list

		"""
		smiles = self.get_smiles_from_identifier(polymer_identifier)

		single_row_df = pd.DataFrame()
		for descriptor in descriptor_list:
			single_row_df[descriptor] = [calculate_descriptor(smiles, descriptor)]

		# # set index of new dataframe as the unique name of chemical
		# single_row_df.set_index(pd.Index([smiles])], inplace=True)

		return single_row_df

	def add_descriptors(self, descriptor_list):
		"""
		Generate column of descriptor values for each descriptor in descriptor_list and append to DataFrame

		Parameters
		-------------------------
		descriptor_list: [String]
			List of descriptors


		Returns
		-------------------------
		None

		"""

		for descriptor in descriptor_list:
			if descriptor not in list(self.df):
				generated_descriptor_series = self.df["smiles"].apply(calculate_descriptor, args=(descriptor,))

				#store generated descriptor series in class df
				self.df[descriptor] = generated_descriptor_series

	def add_molecular_structures(self, structure_list):
		for structure in structure_list:
			if structure not in self.structures:
				generated_structure_series = self.df["smiles"].apply(generate_input_files, args=(structure,))

				#store generated descriptor series in class df
				self.df[descriptor] = generated_descriptor_series

	def property_existence(self, property_list):
		"""
		Check if each property in property_list exists in dataframe
		If not, create new column and store property values for missing property column into class dataframe

		Parameters
		-------------------------
		property_list: [String]
			List of properties


		Returns
		-------------------------
		None
		"""

		try:
			for prop in property_list:
				if prop not in list(self.df):
					self.add_descriptors([prop])
		except:
			raise KeyError("One or multiple of your input properties either do not match any existing\
				thermo-physical properties in dx.df or property does not match supported chemical descriptors")

	def plot_properties(self, property_x=None, property_y=None):
		"""
		Plot a scatterplot of two properties against each other

		Parameters
		-------------------------
		property_x: String
			property on x-axis
		property_y: String
			property on y-axis

		Returns
		-------------------------
		Scatter plot on pyplot	
		"""

		self.property_existence([property_x, property_y])

		fig, ax = plt.subplots()
		plt.plot(self.df[property_x], self.df[property_y], 'o-', ax=ax)
		fig.tight_layout()
		plt.show()

	def plot_many(self, property_list):
		"""
		Plot a pairplot of property_list

		Parameters
		-------------------------
		property_list: [String]
			list of properties

		Returns
		-------------------------
		Pairplot on pyplot	
		"""
		self.property_existence(property_list)

		sns.pairplot(self.df[property_list])
		plt.tight_layout()
		plt.show()

	def property_correlation(self, property_1, property_2):
		"""
		Calculate correlation between two properties based on Pearson correlation

		Parameters
		-------------------------
		property_1: String
		property_2: String

		Returns
		-------------------------
		correlation: Float

		"""

		self.property_existence([property_1, property_2])
		correlation = self.df[property_1].corr(self.df[property_2])

		return correlation

	def correlation_map(self,property_list):
		"""
		Plot a correlation heatmap of property_list based on Pearson correlation

		Parameters
		-------------------------
		property_list: [String]
			list of properties

		Returns
		-------------------------
		Correlation heatmap on pyplot	
		"""
		self.property_existence(property_list)

		fig, ax = plt.subplots()
		corr = self.df[property_list].corr()
		sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, 
			as_cmap=True), annot=True, ax=ax)
		fig.tight_layout()
		plt.show()

	def na_distribution(self):
		fig, ax = plt.subplots()
		nan_df = pd.DataFrame()
		num_nan = [self.df[x].isna().sum() for x in self.df.columns]

		nan_df["num_nan"] = num_nan
		nan_df["column_id"] = self.df.columns 
		nan_df.plot(x="column_id", y="num_nan", kind="bar", ax=ax)

		fig.tight_layout()
		plt.show()

	def export_csv(self, outpath):
		"""
		Export current class dataframe to a csv file

		Parameters
		-------------------------
		outpath: String
			file path to write csv file to

		Returns
		-------------------------
		csv file created in given outpath
		"""

		self.df.to_csv(outpath)
