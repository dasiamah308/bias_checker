import requests
from bs4 import BeautifulSoup

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove navs, sidebars, etc. (simple heuristic)
    for tag in soup(['script', 'style', 'header', 'footer', 'aside', 'nav']):
        tag.decompose()

    # Get visible text
    paragraphs = soup.find_all('p')
    return "\n".join(p.get_text() for p in paragraphs if p.get_text())
