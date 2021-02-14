# Conjugator
Conjugator is an AI capable of conjugating french verbs.
Multiple models have been implemented,
including RNNs and Transformers for seq2seq translation at character level.
Made with PyTorch.

## How it works ?
We give a french verb as input (start ^ and end $ tokens are then added) :
```
"^jouer$"
```

In addition, the start of the output is given :
```
"^je "
```

We can provide only the start token for full generation :
```
"^"
```

The model then outputs the resulting string :
```
"^je joue"
```

## Examples
Results using the transformer model trained with a dataset of 5k verbs during
about 5 minutes :
```
Input : "^jouer$"
Output : "^je joue"

Input : "^rougir$"
Output : "^nous rougisssons"

Input : "^déglutir$"
Output : "^ils déglutissent"

Input : "^poulier$"
Output : "^je poulie"

# Does not exist
Input : "^vollir$"
Output : "^tu volis"

# Does not exist
Input : "^mager$"
Output : "^tu mages"

# Does not exist
Input : "^praxiter$"
Output : "^vous praxitez"

# Does not exist
Input : "^patriarcher$"
Output : "^tu patriarches"

# Does not exist
Input : "^anticonstituer$"
Output : "j'anticonsistisre"
```

## Dataset
The data has been scraped on the web ([this website](https://la-conjugaison.nouvelobs.com)), use [src/scrap.py](src/scrap.py) to generate the dataset.
