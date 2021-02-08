# Load and parse dataset

import random
import json


def load(path):
    '''
    Loads the dataset
    '''
    # Parse json
    with open(path) as f:
        return json.load(f)


def filter_data(data, mode, tense):
    '''
    Selects one mode and one tense for every verb
    '''
    filtered_data = {}
    for v in data:
        filtered_data[v] = data[v][mode][tense]

    return filtered_data


def get_voc(data):
    '''
    Returns possible tokens within keys and values
    - Returns (voc, voc_keys, voc_values)
    '''
    # Find all possible tokens
    voc_keys = set()
    voc_values = set()
    for v in data:
        voc_keys |= set(v)
        for item in data[v]:
            voc_values |= set(item)

    tovoc = lambda s: ''.join(sorted(list(s)))

    # Use voc.index(token) to have its id
    voc = tovoc(voc_keys | voc_values)
    voc_keys = tovoc(voc_keys)
    voc_values = tovoc(voc_values)

    return voc, voc_keys, voc_values


def iter_data(data,
        voc_keys=None, voc_values=None,
        pad_tok_keys=None, pad_tok_values=None,
        start_tok_keys=None, start_tok_values=None,
        end_tok_keys=None, end_tok_values=None,
        batch_size=1, shuffle=True):
    '''
    Iterates within all the data
    - voc_{keys,values}: If None, yields strings as keys/values,
        yields indices otherwise
    - pad_tok_{keys,values} : If None, no padding, otherwise this item is added
        such that all items have the same length in a batch (use list not int)
    - {start,end}_tok_{keys,values} : Add this token at the start / end of
        every key / value if not None (use list not int)
    - Yields keys, values
    '''
    keys = list(data.keys())

    if shuffle:
        random.shuffle(keys)

    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i : i + batch_size]
        batch_values = [data[k] for k in batch_keys]

        # Index
        if voc_keys is not None:
            batch_keys = [[voc_keys.index(k) for k in batch]
                    for batch in batch_keys]

        if voc_values is not None:
            batch_values = [[[voc_values.index(v) for v in c]
                    for c in batch]
                    for batch in batch_values]

        # Add start
        if start_tok_keys is not None:
            batch_keys = [start_tok_keys + k for k in batch_keys]

        if start_tok_values is not None:
            batch_values = [[start_tok_values + v for v in b] for
                    b in batch_values]

        # Add end
        if end_tok_keys is not None:
            batch_keys = [k + end_tok_keys for k in batch_keys]

        if end_tok_values is not None:
            batch_values = [[v + end_tok_values for v in b] for
                    b in batch_values]

        # Pad
        if pad_tok_keys is not None:
            maxlen = max((len(k) for k in batch_keys))
            batch_keys = [k + pad_tok_keys * (maxlen - len(k)) for
                    k in batch_keys]

        if pad_tok_values is not None:
            maxlen = 0
            for b in batch_values:
                for v in b:
                    maxlen = max(maxlen, len(v))

            batch_values = [[v + pad_tok_values * (maxlen - len(v)) for
                    v in b] for b in batch_values]

        yield batch_keys, batch_values


if __name__ == '__main__':
    data = load('dataset.json')

    ### Simple query
    # Fetch a verb
    verb_name = next(iter(data))
    verb = data[verb_name]

    # Fetch tense mode
    modes = [mode for mode in verb]
    indicatif = verb[modes[0]]

    # Fetch a conjugation for a tense
    indicatif_tenses = [tense for tense in indicatif]
    present = indicatif['Présent']

    # Just to not display
    if 0:
        print('Found', len(data), 'items')
        print('Example :', list(data)[:8], '...')
        print('Verb :', verb_name)
        print('Modes :', modes)
        print('Indicatif Tenses :', modes)
        print('Indicatif Présent :', present)

    ### Deep learning preprocessing (for NLP)
    target_mode = 'Indicatif'
    target_tense = 'Présent'

    filtered_data = filter_data(data, target_mode, target_tense)

    voc, voc_keys, voc_values = get_voc(filtered_data)

    print('Key tokens :', f'"{voc_keys}"', f'({len(voc_keys)})')
    print('Value tokens :', f'"{voc_values}"', f'({len(voc_values)})')
    print('Total tokens :', f'"{voc}"', f'({len(voc)})')

    print()
    # Note that we can use a vocabulary (don't forget to add start / end / pad
    # tokens to the vocabulary)
    # for names, values in iter_data(
    #         filtered_data,
    #         voc_keys=voc, voc_values=voc_values,
    #         pad_tok_keys=[43], pad_tok_values=[43],
    #         start_tok_keys=[1], start_tok_values=[1],
    #         end_tok_keys=[42], end_tok_values=[42],
    #         batch_size=5):
    for names, values in iter_data(
            filtered_data,
            pad_tok_keys='_', pad_tok_values='_',
            start_tok_keys='^', start_tok_values='^',
            end_tok_keys='$', end_tok_values='$',
            batch_size=5):
        for name, value in zip(names, values):
            print('>', name)
            for v in value:
                print(v)

        break
