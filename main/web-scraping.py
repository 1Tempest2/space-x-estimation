import sys

import bs4
import requests
from bs4 import BeautifulSoup
import re
import unicodedata
import pandas as pd


def date_time(table_cells):
    return [data_time.strip() for data_time in list(table_cells.strings)][0:2]


def booster_version(table_cells):
    out = ''.join([booster_version for i, booster_version in enumerate(table_cells.strings) if i % 2 == 0][0:-1])
    return out


def landing_status(table_cells):
    out = [i for i in table_cells.strings][0]
    return out

def get_mass(table_cells):
    mass = unicodedata.normalize("NFKD", table_cells.text).strip()
    if mass:
        mass.find("kg")
        new_mass = mass[0:mass.find("kg") + 2]
    else:
        new_mass = 0
    return new_mass


def extract_column_from_header(row):
    if (row.br):
        row.br.extract()
    if row.a:
        row.a.extract()
    if row.sup:
        row.sup.extract()

    colunm_name = ' '.join(row.contents)

    # Filter the digit and empty names
    if not (colunm_name.strip().isdigit()):
        colunm_name = colunm_name.strip()
        return colunm_name

static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"
response = requests.get(static_url)

if response.status_code == 200:
    # Parse the page content using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    #print(soup.title.string)
else:
    print(f"Failed to fetch the page. Status code: {response.status_code}")

html_tables = soup.find_all("table")
first_launch_table = html_tables[2]
#print(first_launch_table)
column_names = [
    extract_column_from_header(th)
    for th in first_launch_table.find_all("th")
    if extract_column_from_header(th) and len(extract_column_from_header(th)) > 0
]
print(column_names)