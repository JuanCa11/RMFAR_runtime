from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
import pandas as pd


class CARs():
    def __init__(self, wildcards, transactions, classes: list,
                 p_support=0.25, p_confidence=0.25, p_antecedents=0.25,
                 p_wildcards=0.25):
        self.p_support = p_support
        self.p_confidence = p_confidence
        self.p_antecedents = p_antecedents
        self.p_wildcards = p_wildcards
        self.wildcards = wildcards
        self.transactions = transactions
        self.classes = classes

    def generate_frequent_itemsets(self, min_support=0.014):
        te = TransactionEncoder()
        te_ary = te.fit(self.transactions).transform(self.transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        self.frequent_itemsets = fpgrowth(df, min_support=min_support,
                                          use_colnames=True)

    def filter_fn(self, row):
        for clas in self.classes:
            if row['consequents'] == {clas}:
                return True
        return False

    def get_car_rules(self, metric='confidence', min_threshold=0.6):
        rules = association_rules(self.frequent_itemsets, metric=metric,
                                  min_threshold=min_threshold)

        rules = rules[rules.apply(self.filter_fn, axis=1)]

        rules['antecedents_len'] = rules['antecedents'].apply(lambda x: len(x))
        rules['n_wildcards'] = rules['antecedents'].apply(
                                            lambda x: self.get_n_wildcards(x))

        rules["nor_antecedents"] = ((rules['antecedents_len']
                                    - rules['antecedents_len'].min()) /
                                    (rules['antecedents_len'].max()
                                    - rules['antecedents_len'].min()))

        rules['nor_wilcards'] = ((rules['n_wildcards']
                                 - rules['n_wildcards'].min()) /
                                 (rules['n_wildcards'].max()
                                 - rules['n_wildcards'].min()))

        rules['weigth'] = rules.apply(lambda x: self.get_rule_weigth(x),
                                      axis=1)
        rules['antecedents'] = rules['antecedents'].apply(set)
        rules['consequents'] = rules['consequents'].apply(set)
        self.rules = rules.sort_values(by=['weigth'], ascending=False)
        return self.rules

    def get_car_rules_weigth(self):
        self.rules['weigth'] = self.rules.apply(
                                lambda x: self.get_rule_weigth(x),
                                axis=1)
        return self.rules

    def get_n_wildcards(self, x):
        n_wilcards = 0
        for item in x:
            if item in self.wildcards_transpose:
                n_wilcards += 1
        return n_wilcards

    @property
    def wildcards_transpose(self):
        wildcards_dict = {}
        for key, value in self.wildcards.items():
            for val in value:
                wildcards_dict[val] = key
        return wildcards_dict

    def get_rule_weigth(self, x):
        return x['support']*self.p_confidence + \
               x['confidence']*self.p_confidence + \
               x['nor_antecedents']*self.p_antecedents + \
               x['nor_wilcards']*self.p_wildcards
