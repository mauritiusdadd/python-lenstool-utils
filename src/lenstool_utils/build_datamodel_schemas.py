#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 21:54:41 2022

@author: daddona
"""
import os
import sys
import html
import tqdm
from urllib import request
from bs4 import BeautifulSoup
import json

ROOT_URL = "https://projets.lam.fr"
WIKI_URL = f"{ROOT_URL}/projects/lenstool/wiki/Parfile"


def get_identifier_datamodel(page_url):
    with request.urlopen(page_url) as req:
        page_html = req.read()
    soup = BeautifulSoup(page_html, 'html.parser')

    par_list = soup.find_all("a", {"class": "wiki-anchor"})
    parameters = [
        x.parent.text.strip().strip('\u00b6')
        for x in par_list if x.parent.name == 'h3'
    ]
    return parameters

def buld_data_model_schemas_from_url_doc(wiki_url=WIKI_URL):
    with request.urlopen(wiki_url) as req:
        main_page_html = req.read()
    soup = BeautifulSoup(main_page_html, 'html.parser')
    first_ids = soup.find_all("a", {"class": "wiki-page"})
    first_identifiers = {
        x.text: f"{ROOT_URL}/{x['href']}" for x in first_ids
    }

    get_identifier_datamodel(
        list(first_identifiers.values())[0]
    )

    data_model_dic = {}
    for key, val in tqdm.tqdm(first_identifiers.items()):
        try:
            data_model_dic[key] = get_identifier_datamodel(val)
        except Exception:
            print(f"\nError in retrieving information about '{key}'")
            continue

    return data_model_dic


def main(resume=True):
    RAW_DATA_MODEL_JSON_FILE = 'data_model_raw.json'
    if  resume and os.path.isfile(RAW_DATA_MODEL_JSON_FILE):
        print("Loading raw data model from cached file...")
        with open(RAW_DATA_MODEL_JSON_FILE, 'r') as f:
            raw_data_model_dic = json.load(f)
    else:
        print(f"Downloading raw data model info from {ROOT_URL}")
        raw_data_model_dic = buld_data_model_schemas_from_url_doc()
        with open(RAW_DATA_MODEL_JSON_FILE, 'w') as f:
            json.dump(raw_data_model_dic, f)

    data_model = {}
    for id_name, id_data in raw_data_model_dic.items():
        id_params = {}
        for j, par_data in enumerate(id_data):
            par_data = par_data.strip().split()
            par_name = par_data[0]
            par_fields = {}
            for k, field in enumerate(par_data[1:]):
                if field[0] == '[':
                    if field[-1] != ']':
                        print("WoW! Broken syntax for field {field}")
                        sys.exit(1)
                    else:
                        field_required = False
                else:
                    field_required = True

                if 'int' in field:
                    field_type = 'int'
                elif 'float' in field:
                    field_type = 'float'
                elif 'ra' in field.lower() or 'dec' in field.lower():
                    field_type = 'float'
                else:
                    field_type = 'unknown'

                par_fields[k] = {
                    'type': field_type,
                    'required': field_required
                }
            id_params[par_name] = par_fields
        data_model[id_name] = id_params
    return data_model


def print_data_model(data_model):
    for id_name, params in data_model.items():
        print(f'[{id_name}]')
        for par_name, par_fields in params.items():
            print(f'  - "{par_name}"')
            for k, v in par_fields.items():
                print(f'    + {k} : {v}')


if __name__ == '__main__':
    dm = main(resume=True)
    print_data_model(dm)
    with open('data_model.json', 'w') as f:
        json.dump(dm, f, indent=2)
