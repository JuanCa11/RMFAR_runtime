# Fuzzy Assocation Rules

## What's here

This repository contains utilities to work with Fuzzy Assocation Rules using FP-Growth to generate frequent itemsets and generate CARs (Classification Association Rules) to build a recommender system

## Installing fuzzy assocation rules 

To install it:

```bash
$ pip install git+https://github.com/JuanCa11/fuzzy-association-rules.git
```

## Usage

If you need the `"bar"` string, just import it from this module.

```python
from fuzzy_assocation_rules.fuzzy_utils.fuzzification import Fuzzification

fuzzification = Fuzzification(dataset, fuzzy_sets, cluster_feature)
fuzzification.fuzzify()
```
