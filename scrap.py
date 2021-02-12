# Used to fetch data from the web

import requests
import json
from bs4 import BeautifulSoup, Tag
from unidecode import unidecode
from tqdm import tqdm


def fetch_data(verb, norm):
    '''
    Fetches the data for one verb (all conjugations)
    - norm : Normalization functor
    - Returns (title, data) title is unicode and data is a dict
    '''
    data = {}

    url = f'https://la-conjugaison.nouvelobs.com/du/verbe/{verb}.php'
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')

    title = norm(soup.find(class_='titre_fiche').find('span').find('b').text)

    for parent in soup.find_all(class_='mode'):
        mode = norm(parent.text)

        # Not a mode but another title
        if ' ' in mode.strip():
            continue

        data[mode] = {}

        for sibling in parent.next_siblings:
            if not isinstance(sibling, Tag):
                continue

            if 'tempstab' not in sibling['class']:
                break

            a = next(sibling.children)
            tense = norm(a.text)

            items = []
            item = ''
            content = a.next_sibling
            for c in content:
                if c.name == 'br':
                    items.append(norm(item))
                    item = ''
                elif isinstance(c.string, str):
                    item += c.string

            data[mode][tense] = items

    return title, data


def fetch_verbs(start_verb, queries=3):
    '''
    Returns a set of all possible verbs in the website (using the website's
    recommandations).
    '''
    verbs = set()
    to_query = set([start_verb])

    bar = tqdm(range(queries))
    for _ in bar:
        if len(to_query) == 0:
            print('Warning : Stopped fetch_verbs before maximum number' +
                'of queries')
            break

        verb = to_query.pop()
        verbs.add(verb)

        bar.set_postfix({ 'verb': verb })

        url = f'https://la-conjugaison.nouvelobs.com/du/verbe/{verb}.php'
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')

        try:
            header = next(i for i in soup.find_all('div', class_='mode') \
                    if 'au hasard' in i.text).next_sibling
        except:
            print('Warning : Failed to parse', verb, 'in fetch_verbs')
            continue

        # For each link containing each verb
        for link in header:
            v = unidecode(link.string)

            # Remove prefix
            v.replace("s'", '')
            v.replace("se ", '')

            if v not in verbs and v not in to_query:
                to_query.add(v)

    return verbs | to_query


path = 'dataset.json'
start_verb = 'etre'
max_queries = 300

# We can also use unidecode or lower to remove special chars such as accents
# or to lowercase the string
norm = lambda s: s.strip()

print('### Fetching list of verbs')
verbs = fetch_verbs(start_verb, max_queries)
print('> Found', len(verbs), 'verbs')

print('### Fetching verbs data')
data = {}
bar = tqdm(verbs)
for v in bar:
    try:
        vname, vdata = fetch_data(v, norm)

        data[vname] = vdata
    except Exception as e:
        print('Warning : Cannot parse verb', v)
        print(e)

print('> Saving dataset in file', path)

with open(path, 'w') as f:
    json.dump(data, f)

print('Done')
