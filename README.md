# Recommender Model with Fuzzy Association Rules (RMFAR)

## What's here

This repository contains utilities to work with Fuzzy Assocation Rules using FP-Growth to generate frequent itemsets and generate CARs (Classification Association Rules) to build a recommender system

## Installing fuzzy assocation rules 

To install it:

```bash
$ pip install git+https://github.com/JuanCa11/fuzzy-association-rules.git
```

## Usage

If you need to fuzzify your dataset, you just have to import it from this module.

```python
from fuzzy_ar.fuzzification import Fuzzification

fuzzification = Fuzzification(dataset, fuzzy_sets, cluster_feature)
fuzzification.fuzzify()
fuzzified_data = fuzzification.fuzzified_data
```

If you need to create class association rules with fp-growth, you just have to import it from this module.

```python
from fuzzy_ar.cars_fp_growth import CARs

cars = CARs(wildcards, transactions, positive, negative)
cars.generate_frequent_itemsets()
rules = cars.get_car_rules()

```

If you need to create a recommender based on class association rules, you just have to import it from this module.

```python
from fuzzy_ar.recommender import Recommender

recommender = Recommender(rules, fuzzy_sets)
trigger_rules = recommender.trigger_rules(test_data, memberships_test_data)
new_rules = recommender.get_new_rules()
```
