# Conjugator
Conjugator is an AI capable of conjugating french verbs.
Multiple models have been implemented,
including RNNs and Transformers for seq2seq translation.

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
Results using the transformer model trained with a dataset of 5k verbs for
about 5 minutes :
```
Input : "^jouer$"
Output : "^je joue"

Input : "^rougir$"
Output : "^nous rougisssons"

# Does not exist
Input : "^vollir$"
Output : "^tu volis"

# Does not exist
Input : "^mager$"
Output : "^tu mages"

Input : "^déglutir$"
Output : "^ils déglutissent"

# Does not exist
Input : "^praxiter$"
Output : "^vous praxitez"

Input : "^poulier$"
Output : "^je poulie"

# Does not exist
Input : "^patriarcher$"
Output : "^tu patriarches"

# Does not exist
Input : "^anticonstituer$"
Output : "j'anticonsistisre"
```
