import requests
from lxml import etree
import pandas as pd
from io import StringIO
import os
import pprint
# Make the request to the Wikipedia page
response = requests.get("https://en.wikipedia.org/wiki/Lists_of_cities_by_country")

if response.status_code == 200:
    # Parse the HTML content of the page
    dom = etree.HTML(response.content)
    
    # Remove all <figure> tags (for figures, images, charts, etc.)
    for figure in dom.xpath('//figure'):
        figure.getparent().remove(figure)

    # Remove the specific element at /html/body/div[2]/div/div[3]/main/div[3]/div[3]/div[1]/ul[26]
    element_to_remove = dom.xpath('/html/body/div[2]/div/div[3]/main/div[3]/div[3]/div[1]/ul[26]')
    for element in element_to_remove:
        element.getparent().remove(element)

    # Extract all <a> tags and their 'href' attributes from the cleaned content
    hrefs = dom.xpath('//a/@href')

    # Filter out any hrefs that are None or empty
    hrefs = set([href for href in hrefs if "List_of" in str(href)])
    hrefs.remove('/wiki/List_of_lists')
    hrefs.remove('/wiki/List_of_timelines')
    pprint.pprint(hrefs)
    # exit()
    # Print the list of hrefs
    for href in hrefs:
        print(f'https://en.wikipedia.org{href}')
        response = requests.get(f'https://en.wikipedia.org{href}')
        if response.status_code == 200:
            dom = etree.HTML(response.content)
            element = dom.xpath('//*[@id="mw-content-text"]/div[1]/table')

            if element:
                try:
                    """Parse a generic HTML table into a DataFrame."""
                    # Convert the found element to a string
                    table_html = etree.tostring(element[0], encoding='unicode')

                    # Use StringIO to wrap the HTML string for pandas
                    html_io = StringIO(table_html)

                    # Use pandas to read the HTML table and convert it to a DataFrame
                    df = pd.read_html(html_io)[0]  # Assuming you want the first table if there are multiple

                    # some wikipedia sites and articles have written that they are outdated, so we use 7 as a lucky number to exclude these datapoints
                    try:
                        if len(df) > 7:
                            df.to_csv(f"{os.getcwd()}/data/wikipedia_city_gathering/{str(href).split("in_")[-1]}.csv")
                    except Exception as e:
                        if len(df) > 7:
                            df.to_csv(f"{os.getcwd()}/data/wikipedia_city_gathering/{str(href).split("/")[-1]}.csv")
                except ValueError as e:
                    print(f"Error reading table from {href}")
            else:
                print("no tables found")