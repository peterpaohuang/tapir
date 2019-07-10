from rdkit import Chem
from rdkit.Chem import Descriptors

def calculate_descriptor(smiles, descriptor):
	"""
	Calculate a single descriptor for single smiles and return descriptor value

	Parameters
	-------------------------
	smiles: String
	descriptor: String

	Returns
	-------------------------
	descriptor_value: String
		value of generated descriptor based on smiles

	"""

	descriptor_method = getattr(Descriptors, descriptor)

	m = Chem.MolFromSmiles(smiles)
	descriptor_value = descriptor_method(m)
	
	return descriptor_value