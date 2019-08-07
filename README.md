# Depablo Box

## Requirements

1. Conda is installed

## Installation
1. `git clone https://github.com/peterpaohuang/depablo_box.git`
2. `conda create -c rdkit -n depablo_box_env rdkit`
3. `conda activate depablo_box_env`
4. Download [polymer_db.csv](https://drive.google.com/file/d/1--OtZ7XLnx_b4n9--5E7b7NqjXOByXGw/view?usp=sharing)
5. Move polymer_db.csv into depablo_box directory
6. `python setup.py` while inside depablo_box_env conda environment

## Initialize
```
from depablo_box import PDBML, model

dx = PDBML()
```
## Understand the database
### Access database as pandas dataframe
```
df = dx.df
```

### List all polymers and corresponding smiles
```
# list both polymer names and smiles
df[["polymer_name", "smiles"]]

# list only polymer names
df["polymer_name"]

# list only smiles
df["smiles"]

# retrieve polymer row by polymer_name
df.loc[df["polymer_name"] == polymer_name]

# retrieve polymer row by smiles
df.loc[df["smiles"] == smiles]
```

### List Descriptors
#### Supported Chemical Descriptors
`dx.chemical_descriptors`
* ExactMolWt
* FpDensityMorgan1
* FpDensityMorgan2
* FpDensityMorgan3
* HeavyAtomMolWt
* MolWt
* etc

#### Supported Thermo-Physical Descriptors
`dx.experimental_descriptors`
* Molar Volume Vm
* Density ρ
* Solubility Parameter δ
* Molar Cohesive Energy Ecoh
* Glass Transition Temperature Tg
* Molar Heat Capacity Cp
* Entanglement Molecular Weight Me
* Index of Refraction n
* Coefficient of Thermal Expansion α
* Molecular Weight of Repeat unit
* Van-der-Waals Volume VvW

### See distribution of NaN values in database for Thermo-Physical Descriptors
```
dx.na_distribution()
```

### List Machine Learning Methods
```
dx.ml_methods
```

### List Conversion Formats Directly from SMILES
```
dx.conversion_formats
```

## How to use
_Note: currently, depablo_box is only able to handle the calculation of chemical descriptors. Experimental descriptors already exists within the database (dx.df)_
### Get Chemical Descriptors 
```
descriptor_list = ["ExactMolWt", "HeavyAtomMolWt"]
polymer_identifier = "C=CC(=O)NC(C)(C)C" # can also be the polymer_name
descriptor_df = dx.get_descriptors(polymer_identifier, descriptor_list)
```

### Generate Input Files for Quantum Chemistry Codes
#### Supported Conversion Formats
1. `Protein Data Bank`
2. `Gaussian 98/03 Input`
```
polymer_identifier = '*C(C*)C' # can also be the polymer_name
conversion_format = 'Gaussian 98/03 Input'
outpath = '/file/path/your_polymer.xyz'
dx.create_input_file(polymer_identifier, conversion_format, outpath)
```

### Add Chemical Descriptors to dataframe
```
dx.add_descriptors(descriptor_list)
```
### Plot Properties as scatterplot
```
dx.plot_properties(property_x="glass_transition_temperature", property_y="ExactMolWt")
```
### Plot Many Properties as Pairplot
```
dx.plot_many(property_list)
```
### Get Correlation Between Two Properties
```
dx.property_correlation("molar_heat_capacity", "HeavyAtomMolWt")
```
### Plot Correlation Heatmap of Many Properties
```
dx.correlation_map(property_list)
```
### Export Dataframe as CSV file
```
dx.export_csv(outpath)
```
## Initialize Model Training
```
# input_properties must have already been added to PDBML().df
input_properties = ["molar_heat_capacity", "ExactMolWt", "HeavyAtomMolWt"]
output_property = "solubility_parameter"
na_strategy = "remove"
ml = model(df, input_properties, output_property, na_strategy=na_strategy)
```
### Train Model
#### Supported Model Types
1. `Support Vector Regression`
2. `Linear Regression`
3. `Ridge Regression`
4. `Lasso Regression`
5. `Gaussian Process Regression`
```
model_type = "Support Vector Regression"
ml.train(model_type)
```
#### View Trained Model R^2 Score
```
ml.r_2
```

### Predict on new data
```
new_data = [["10.5", "29", "102.1"]]
results = ml.predict(new_data)
```
### Plot Feature Importances
_Note: model type Gaussian Process Regression does not support feature importances_
```
ml.feature_importance()
```
### Export Trained Model as Pickle File
```
ml.export_fitted_model(outpath)
```
### Load Pickle File as Trained Model
```
import pickle
with open(outpath, "rb") as f:
  ml = pickle.load(f)
results = ml.predict(new_data)
```
## Scrape CROW Polymer DB for experimental thermo-physical properties
```
from depablo_box import polymer_scraper
```
### Initialize scraper
```
scraper = polymer_scraper()
```
### Start Scraping
```
scraper.start()
```
### Once Finished, Store Scraped Data
```
outpath = /file/path/to/store/FILE.csv
scraper.store_data(outpath)
```

