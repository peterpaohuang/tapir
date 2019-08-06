import time
import sys
import re
import requests

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from math import ceil

class polymer_scraper():
    """
    Scraper is only for CROW Polymer Properties Database
    """

    def __init__(self):
        self.columns = {
            "Molar Volume Vm": "molar_volume", 
            "Density ρ": "density",
            "Solubility Parameter δ": "solubility_parameter",
            "Molar Cohesive Energy Ecoh": "molar_cohesive_energy",
            "Glass Transition Temperature Tg": "glass_transition_temperature",
            "Molar Heat Capacity Cp": "molar_heat_capacity",
            "Entanglement Molecular Weight Me": "entanglement_molecular_weight",
            "Index of Refraction n": "refraction_index",
            "Coefficient of Thermal Expansion α": "thermal_expansion_coefficient",
            "Molecular Weight of Repeat unit": "repeat_unit_weight",
            "Van-der-Waals Volume VvW": "waals_volume"
        }
        self.df = pd.DataFrame(columns=["polymer_name", "smiles", "molar_volume", "density",
         "solubility_parameter","molar_cohesive_energy", "glass_transition_temperature", "molar_heat_capacity", 
            "entanglement_molecular_weight", "refraction_index", "thermal_expansion_coefficient", 
            "repeat_unit_weight", "waals_volume"])

        self.ses_req = requests.Session()

    def extract_polymer_properties(self, polymer_url):
        """
        From an individual polymer site, extract the thermo-physical properties in self.columns

        Parameters
        ---------------------------
        polymer_url: String
            url site for individual polymer

        Return 
        ---------------------------
        poly_dict: {COLUMN_NAME: NUMERICAL_VALUE}
            dictionary of with keys as the column names in self.df 
            and the corresponding numerical value
        """

        poly_dict = {}
        single_poly_url = "https://polymerdatabase.com/polymers/{}".format(polymer_url)

        page = self.ses_req.get(single_poly_url)
        soup = BeautifulSoup(page.content,'lxml')

        poly_dict["polymer_name"] = soup.title.string

        try:
            polymer_identifiers = soup.find_all("div", class_="datagrid")[1]
        except:
            return None

        try:
            poly_dict["smiles"] = polymer_identifiers.find_all('tr')[2].find_all('td')[1].string.strip()
        except:
            poly_dict["smiles"] = np.nan

        polymer_properties = soup.find_all("div", class_="datagrid")[2].find_all('tr')
        for x, row in enumerate(polymer_properties):
            if x > 0:
                prop_cols = row.find_all('td')
                # preferred values are top priority - if there are no values there, take value/range
                try:
                    if prop_cols[3].string is not None:
                        if "Average" in prop_cols[3].string:
                            poly_dict[self.columns[prop_cols[0].get_text()]] = float(prop_cols[3].string.
                                split(":")[-1].strip())
                        else:
                            poly_dict[self.columns[prop_cols[0].get_text()]] = float(prop_cols[3].string.strip())
                    elif prop_cols[2].string is not None:
                        try:
                            poly_dict[self.columns[prop_cols[0].get_text()]] = float(prop_cols[2].string.strip())
                        except:
                            poly_dict[self.columns[prop_cols[0].get_text()]] = float(prop_cols[2].string.
                                strip().split()[-1])
                    else:
                        poly_dict[self.columns[prop_cols[0].get_text()]] = np.nan
                except: 
                    # when there are not all the right tds - generally seen in index of refraction when value
                    # is none
                    poly_dict[self.columns[prop_cols[0].get_text()]] = np.nan

        return poly_dict

    def extract_polymers_from_class(self, polymer_class):
        """
        From polymer class site that has a list of polymers in that class, 
        extract the url for each individual polymer

        Parameters
        ---------------------------
        polymer_class: String
            url site for polymer class

        Return 
        ---------------------------
        polymer_urls: [String]
            list of polymer url site

        """

        polymer_urls = []
        
        class_url = "https://polymerdatabase.com/polymer%20index/{}".format(polymer_class)
        page = self.ses_req.get(class_url)
        soup = BeautifulSoup(page.content,'lxml')

        try:
            polymer_list = soup.find_all("ul", class_="auto-style13")[0].find_all('li')
        except:
            # error when polymer_class url is "#.html"
            return None
        for polymer in polymer_list:
            polymer_url = polymer.a.get('href').split('/')[-1]
            polymer_urls.append(polymer_url.strip())

        return polymer_urls

    def extract_classes(self, abc_site_url):
        """
        Parameters
        ---------------------------
        abc_site_url: String
            url site for index letters (A-B, C-D, etc)

        Return 
        ---------------------------
        polymer_classes: [String]
            list of polymer classes' url site

        """

        page = self.ses_req.get(abc_site_url)
        soup = BeautifulSoup(page.content,'lxml')

        two_columns = soup.find_all('td')

        polymer_classes = []

        for column in two_columns:
            for p in column.find_all('p'): 
                if p.a is not None:
                    class_url = p.a.get('href').split('/')[-1]
                    if class_url != "#": # remove tds with dash rather than a polymer class
                        polymer_classes.append(class_url.strip())

        return polymer_classes

    def start(self):
        """
        Start scraping process

        Layout of CROW Polymer Database
        A-B,C-D...
            -> Polymer Classes List
                -> Polymer List
                    -> Thermo-physical Properties for individual polymer

        """

        abc_site_urls = ["https://polymerdatabase.com/home.html", 
        "https://polymerdatabase.com/polymer%20index/C-D%20index.html", 
        "https://polymerdatabase.com/polymer%20index/E-F%20index.html",
        "https://polymerdatabase.com/polymer%20index/G-L%20index.html",
        "https://polymerdatabase.com/polymer%20index/M-P%20index.html",
        "https://polymerdatabase.com/polymer%20index/S-V%20index.html"]

        print("Scraping started...")
        polymers_scraped = 0
        start_time = time.time()

        for abc_site_url in abc_site_urls:
            polymer_classes = self.extract_classes(abc_site_url)

            for polymer_class in polymer_classes:
                polymer_urls = self.extract_polymers_from_class(polymer_class)

                if polymer_urls is not None:
                    for polymer_url in polymer_urls:
                        # log number of polymers scraped
                        if polymers_scraped and polymers_scraped % 100 == 0:
                            print("{} polymers scraped in {} seconds".format(polymers_scraped, 
                                int(time.time()-start_time)))

                        poly_dict = self.extract_polymer_properties(polymer_url)
                        if poly_dict is not None:
                            self.df = self.df.append(poly_dict, ignore_index=True)
                            polymers_scraped += 1

    def store_data(self, outpath):
        """
        Store scraped data from CROW on a CSV file to outpath

        Parameters
        ---------------------------
        outpath: String
            outpath to store CSV file

        """

        self.df.to_csv(outpath)

# scraper = polymer_scraper()
# scraper.start()
# scraper.store_data("polymer_db.csv")

