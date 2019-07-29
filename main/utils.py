import numpy as np
import pandas as pd

import openbabel
from openbabel import *
from rdkit import Chem
from rdkit.Chem import Descriptors,rdmolfiles,AllChem
from sklearn.impute import SimpleImputer as Imputer


def pdb_to_gau(pdb):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "gau")

    mol = openbabel.OBMol()
    obConversion.ReadString(mol, pdb)


    outPDB = obConversion.WriteString(mol)
    return outPDB

def smiles_to_pdbfile(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.MolToPDBBlock(mol)
    return mol

def generate_input_files(smiles, format):
    """
    Generate single input file for quantum chemistry codes

    Parameters
    -------------------------
    smiles: String
    format: String
        format you want to generate from smiles

    Returns
    -------------------------
    out: String
        the resulting conversion format from smiles
    """

    first_level_conversion = {
        'Protein Data Bank': smiles_to_pdbfile
        # can eventually add more
    }
    second_level_conversion = {
        # files that need 3 dimension input to calculate
        'Gaussian 98/03 Input': pdb_to_gau
    }

    if format in first_level_conversion.keys():
        out = first_level_conversion[format](smiles)

        return out
    elif format in second_level_conversion.keys():
        pdb = first_level_conversion["Protein Data Bank"](smiles)
        out = second_level_conversion[format](pdb)

        return out
    else:
        raise KeyError("File format cannot be used to generate files. " + 
            "Please refer to our documentation to see accepted file types")

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

    try:
        m = Chem.MolFromSmiles(smiles)
        descriptor_value = descriptor_method(m)

        return descriptor_value
    except Exception as e:
        return np.nan
    

class NA_encoder():

    """Encodes missing values for numericalfeatures.

    Several strategies are possible in each case.

    Parameters
    ----------
    numerical_strategy : str or float or int. default = "mean"
        The strategy to encode NA for numerical features.
        Available strategies = "mean", "median",
        "most_frequent" or a float/int value
    """

    def __init__(self,
                 numerical_strategy='mean',
                 categorical_strategy='most_frequent'):

        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.__Lcat = []
        self.__Lnum = []
        self.__imp = None
        self.__mode = dict()
        self.__fitOK = False

    def get_params(self, deep=True):
        return {'numerical_strategy': self.numerical_strategy, 'categorical_strategy': self.categorical_strategy}

    def set_params(self, **params):

        self.__fitOK = False

        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for encoder NA_encoder. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`encoder.get_params().keys()`")
            else:
                setattr(self, k, v)

    def fit(self, df_train):

        """Fits NA Encoder.

        Parameters
        ----------
        df_train : pandas dataframe of shape = (n_train, n_features)
            The train dataset with numerical and categorical features.

        Returns
        -------
        object
            self
       """

        self.__Lcat = df_train.dtypes[df_train.dtypes == 'object'].index
        self.__Lnum = df_train.dtypes[df_train.dtypes != 'object'].index

        # Dealing with numerical features
        if (self.numerical_strategy in ['mean', 'median', "most_frequent"]):

            self.__imp = Imputer(strategy=self.numerical_strategy)


            if (len(self.__Lnum) != 0):
                self.__imp.fit(df_train[self.__Lnum])
            else:
                pass
        elif ((type(self.numerical_strategy) == int) | (type(self.numerical_strategy) == float)):
            pass
        else:
            raise ValueError("Numerical strategy for NA encoding is not valid")

        # Dealing with categorical features
        if (type(self.categorical_strategy) == str):
            if (self.categorical_strategy == "most_frequent"):

                na_count = df_train[self.__Lcat].isnull().sum()

                for col in na_count[na_count>0].index:
                    try:
                        self.__mode[col] = df_train[col].mode()[0]
                    except:
                        self.__mode[col] = "<NULL>"
            else:
                pass
        else:
            raise ValueError("Categorical strategy for NA encoding is not valid")

        self.__fitOK = True

        return self

    def fit_transform(self, df_train):

        """Fits NA Encoder and transforms the dataset.

        Parameters
        ----------
        df_train : pandas.Dataframe of shape = (n_train, n_features)
            The train dataset with numerical and categorical features.

        y_train : pandas.Series of shape = (n_train, ), default = None
            The target for classification or regression tasks.

        Returns
        -------
        pandas.Dataframe of shape = (n_train, n_features)
            The train dataset with no missing values.
        """

        self.fit(df_train)

        return self.transform(df_train)

    def transform(self, df):

        """Transforms the dataset

        Parameters
        ----------
        df : pandas.Dataframe of shape = (n, n_features)
            The dataset with numerical and categorical features.

        Returns
        -------
        pandas.Dataframe of shape = (n, n_features)
            The dataset with no missing values.
        """

        if(self.__fitOK):
            if(len(self.__Lnum) == 0):
                if (self.categorical_strategy != "most_frequent"):
                    return df[self.__Lcat].fillna(self.categorical_strategy)
                else:
                    return df[self.__Lcat].fillna(self.__mode)
            else:
                if (self.numerical_strategy in ['mean',
                                                'median',
                                                "most_frequent"]):
                    if (len(self.__Lcat) != 0):
                        if (self.categorical_strategy != "most_frequent"):
                            return pd.concat(
                                (pd.DataFrame(self.__imp.transform(df[self.__Lnum]),
                                              columns=self.__Lnum,
                                              index=df.index),
                                 df[self.__Lcat].fillna(self.categorical_strategy)
                                 ),
                                axis=1)[df.columns]
                        else:
                            return pd.concat(
                                (pd.DataFrame(self.__imp.transform(df[self.__Lnum]),
                                              columns=self.__Lnum,
                                              index=df.index),
                                 df[self.__Lcat].fillna(self.__mode)
                                 ),
                                axis=1)[df.columns]
                    else:
                        return pd.DataFrame(
                            self.__imp.transform(df[self.__Lnum]),
                            columns=self.__Lnum,
                            index=df.index
                        )
                elif ((type(self.numerical_strategy) == int) | (type(self.numerical_strategy) == float)):
                    if (len(self.__Lcat) != 0):
                        if (self.categorical_strategy != "most_frequent"):
                            return pd.concat(
                                (df[self.__Lnum].fillna(self.numerical_strategy),
                                 df[self.__Lcat].fillna(self.categorical_strategy)
                                 ),
                                axis=1)[df.columns]
                        else:
                            return pd.concat(
                                (df[self.__Lnum].fillna(self.numerical_strategy),
                                 df[self.__Lcat].fillna(self.__mode)
                                 ),
                                axis=1)[df.columns]
                    else:
                        return df[self.__Lnum].fillna(self.numerical_strategy)
        else:
            raise ValueError("Call fit or fit_transform function before")