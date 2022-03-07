'''
    Step 1
    Scrap the data from airfoil tools website 
'''

# Purpose of this script is to scrape all the airfoil data from airfoil tools
import glob
import os
import time
import random
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
airfoil_tools = "http://airfoiltools.com"
airfoil_search_url="http://airfoiltools.com/search/airfoils"


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    return None

def simple_get(url,override=False):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=False)) as resp:
            if (override):
                return get(url)
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        time.sleep(random.randint(1,10))
        return simple_get(url,override)

def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 
            and content_type is not None 
            and content_type.find('html') > -1)

def log_error(e):
    """
    It is always a good idea to log errors. 
    This function just prints them, but you can
    make it do anything.
    """
    print(e)
def scrape_airfoil_list():
    raw_html = simple_get(airfoil_search_url)
    html = BeautifulSoup(raw_html, 'html.parser')
    airfoilURLList = html.findAll("table", {"class": "listtable"})
    tableRows = airfoilURLList[0].findAll("tr")
    airfoil_urls = []
    airfoil_names = []
    for row in tableRows: # Search through all tables 
        airfoil_link = row.find(lambda tag: tag.name=="a" and tag.has_attr('href'))
        if (airfoil_link):
            airfoil_urls.append(airfoil_tools + airfoil_link['href'])
            airfoil_names.append(airfoil_link.text.replace("\\", "_").replace("/","_"))
    return airfoil_urls,airfoil_names

def scrape_airfoil_coords(airfoil_page,airfoilname):    
    lednicerDAT=airfoil_page.replace("details","lednicerdatfile")
    raw_html=simple_get(lednicerDAT,True).content
    soup=BeautifulSoup(raw_html,'lxml')    
    with open('./scrape/{}.txt'.format(airfoilname), mode='wt', encoding='utf-8') as file:
        file.write(soup.text)

def scrape_details(details_page,airfoil_name,Re,Ncrit):
    raw_html=simple_get(details_page)
    html = BeautifulSoup(raw_html, 'html.parser')
    details_table = html.findAll("table", {"class": "details"})
    table_links = details_table[0].findAll("a")
    polar = table_links[2]['href']
    raw_html2 = simple_get(airfoil_tools + polar,True)
    with open('./data/scrape/{}.txt'.format(airfoil_name+"_polar_"+str(Re)+"_"+str(Ncrit)), mode='wt', encoding='utf-8') as file:
        file.write(raw_html2.text)

def scrape_airfoil_polars(airfoil_page,airfoil_name):    
    raw_html=simple_get(airfoil_page)
    html = BeautifulSoup(raw_html, 'html.parser')
    polar_list = html.findAll("table", {"class": "polar"})
    tableRows = polar_list[0].findAll("tr")
    for row in tableRows: # Search through all rows
        columns = row.findAll("td")
        if (columns):
            if (len(columns)>4):
                Re = float(columns[2].text.replace(',',''))
                Ncrit = float(columns[3].text.replace(',',''))
                dataLink = columns[7].find(lambda tag: tag.name=="a" and tag.has_attr('href'))
                try:
                    dataLink = dataLink['href']
                    details_page = airfoil_tools + dataLink
                    scrape_details(details_page,airfoil_name,Re,Ncrit)
                    # time.sleep(random.uniform(0.1, 1.0))
                except:
                    print('problem: ' + airfoil_name)
                    pass
        
if __name__ == "__main__":
    [airfoil_urls,airfoil_names] = scrape_airfoil_list()
    for i in range(0,len(airfoil_urls)):
        # Check if airfoil is already scraped 
        if (find(airfoil_names[i] + ".txt","./data/scrape")==None):
            scrape_airfoil_coords(airfoil_urls[i],airfoil_names[i])
            scrape_airfoil_polars(airfoil_urls[i],airfoil_names[i])
            time.sleep(random.uniform(0.1, 1.5))