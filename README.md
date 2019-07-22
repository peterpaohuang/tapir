# Depablo_Box

## Requirements

1. Conda is installed

## Installation
1. `git clone https://github.com/peterpaohuang/depablo_box.git`
2. `conda create -c rdkit -n my-rdkit-env rdkit`
3. `conda activate my-rdkit-env`
4. Download [polymer_db.csv](https://drive.google.com/file/d/1J0MbhEI2AIuihl0YavBBL0xl9xWgDBjQ/view?usp=sharing)
5. Move polymer_db.csv into depablo_box directory
6. `python setup.py`

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

### List all physical properties
```
df.columns
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

### List Chemical Descriptors
```
dx.chemical_descriptors
```

### List Machine Learning Methods
```
dx.ml_methods
```

## How to use
### Get Chemical Descriptors 
```
descriptor_list = ["ExactMolWt", "HeavyAtomMolWt"]
descriptor_df = dx.get_descriptors("Polyethylene", descriptor_list)
```

### Generate Input Files for Quantum Chemistry Codes
```
polymer_identifier = '*C(C*)C'
dx.create_input_file(polymer_identifier, 'Gaussian 98/03 Input', '/file/path/your_polymer.xyz')
```

### Add Descriptors to dataframe
```
dx.add_descriptors(descriptor_list)
```
### Plot Properties as scatterplot
```
dx.plot_properties(property_x="Tg", property_y="ExactMolWt")
```
### Plot Many Properties as Pairplot
```
dx.plot_many(property_list)
```
### Get Correlation Between Two Properties
```
dx.property_correlation("Tm", "HeavyAtomMolWt")
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
input_properties = ["Tm", "ExactMolWt", "HeavyAtomMolWt"]
output_property = "Tg"
na_strategy = "mean"
ml = model(input_properties, output_property, na_strategy)
```
### Train Model
```
ml.train()
```
#### View Trained Model R^2 Score
```
ml.r_2
```

### Predict on new data
```
new_data = ["10.5", "29", "102.1"]
results = ml.predict(new_data)
```
### Plot Feature Importances
```
ml.feature_importances()
```
### Export Trained Model as Pickle File
```
ml.export_fitted_model(outpath)
```
### Load Pickle File as Trained Model
```
import pickle
ml = pickle.load(outpath)
results = ml.predict(new_data)
```
