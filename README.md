# Depablo-Box

## Installation
1. [Setup RDKit environment](http://www.rdkit.org/docs/Install.html)
2. Download polymer_db.csv  
3. Move polymer_db.csv to depablo_box directory
4. `conda install --file requirements.txt`

## How to use
### Initialize
```
from depablo_box import PDBML, model

dx = PDBML()
```

### Access database as pandas dataframe
```
df = dx.df
```
### Get Chemical Descriptors 
```
descriptor_list = ["ExactMolWt", "HeavyAtomMolWt"]
descriptor_df = dx.get_descriptors("Polyethylene", descriptor_list)
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
### Initialize Model Training
```
input_properties = ["Tm", "ExactMolWt", "HeavyAtomMolWt"]
output_property = "Tg"
na_strategy = "mean"
ml = model(input_properties, output_property, na_strategy)
```
### Train Model
```
ml.train()
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
### Export Trained Model
```
ml.export_fitted_model(outpath)

```
