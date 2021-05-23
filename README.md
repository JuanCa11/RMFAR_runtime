# Recommender Model with Fuzzy Association Rules (RMFAR)

## What's here

This repository contains utilities to work with Fuzzy Assocation Rules using FP-Growth to generate frequent itemsets and generate CARs (Classification Association Rules) to build a recommender system

## Installing fuzzy assocation rules 

To install it:

```bash
$ pip install https://github.com/JuanCa11/RMFAR_runtime.git
```

## Usage

If you need to fuzzify your dataset, you just have to import it from this module.

```python
from fuzzy_ar.fuzzification import Fuzzification

dataset = pd.read_csv('dataset.csv')
fuzzification = Fuzzification(dataset, FUZZY_SETS, CLUSTER_FEATURE)
fuzzification.fuzzify()
data = fuzzification.fuzzified_data
```

If you need to create class association rules with fp-growth, you just have to import it from this module.

```python
from fuzzy_ar.cars_fp_growth import CARs

cars = CARs(wildcards, transactions, target, 0.25, 0.25, 0.25, 0.25)
cars.generate_frequent_itemsets()
rules = cars.get_car_rules()
rules.to_csv('rules.csv', index=False)

```

If you need to create a recommender based on class association rules, you just have to import it from this module.

```python
from fuzzy_ar.recommender import Recommender

recommender = Recommender(rules_case1, fuzzy_sets, no_fuzzy_sets, default_class)
y_pred = recommender.predict_all(X_test)

raw_data = X_test.iloc[6,:]
recommender.predict(raw_data)
trigger_rules = recommender._trigger_rules
new_rules = recommender.get_new_rules(trigger_rules)
```
