import pandas as pd


class Recommender():
    def __init__(self, rules: pd.DataFrame, fuzzy_sets: dict, no_fuzzy_sets: dict,
                 default_class: str):
        self._rules = rules
        self._fuzzy_sets = fuzzy_sets
        self._no_fuzzy_sets = no_fuzzy_sets
        self._default_class = default_class

    def trigger_rules(self, data):

        self._rules['memberships'] = self._rules['antecedents'].apply(
            lambda x: self.get_memberships(x, data))

        self._rules['mu_rule'] = self._rules['memberships'].apply(
            lambda x: self.get_mu_rule(x))

        self._rules['score'] = self._rules['weigth']*0.5 + \
            self._rules['mu_rule']*0.5

        features = ['antecedents', 'consequents', 'confidence',
                            'weigth', 'memberships', 'mu_rule', 'score']

        return self._rules.loc[self._rules['mu_rule'] > 0,features].sort_values(
                by=['score'], ascending=False)

    def get_memberships(self, consequent, data):
        memberships_rule = []
        for index, item in enumerate(consequent):
            if item in self.fuzzy_sets_transpose:
                fuzzy_set_idx = self._fuzzy_sets[
                    self.fuzzy_sets_transpose[item]
                    ].index(item)
                memberships_rule.append(data[
                    f"MEM_{self.fuzzy_sets_transpose[item]}"][fuzzy_set_idx])
            else:
                if data[self.no_fuzzy_sets_transpose[item]] != item:
                    memberships_rule = []
                    break
        return memberships_rule

    def get_mu_rule(self, memberships):
        if memberships:
            return min(memberships)
        else:
            return 0

    def new_rules(self, rule: set, index: int, fuzzy_sets: list):
        new_rules_list = []
        for fuzzy_set in fuzzy_sets:
            if fuzzy_set != fuzzy_sets[index]:
                new_rule = rule.copy()
                new_rule.remove(fuzzy_sets[index])
                new_rule.add(fuzzy_set)
                new_rules_list.append(new_rule)

        return new_rules_list

    def get_new_rules(self, trigger_rules):
        new_rules = []
        for _, rule in trigger_rules.iterrows():
            for index, item in enumerate(rule['antecedents']):
                if item in self.fuzzy_sets_transpose:
                    new_rule = self.new_rules(
                        (rule['antecedents']),
                        self._fuzzy_sets[
                            self.fuzzy_sets_transpose[item]
                            ].index(item),
                        self._fuzzy_sets[self.fuzzy_sets_transpose[item]])
                else:
                    new_rule = self.new_rules(
                        (rule['antecedents']),
                        self._no_fuzzy_sets[
                            self.no_fuzzy_sets_transpose[item]
                            ].index(item),
                        self._no_fuzzy_sets[self.no_fuzzy_sets_transpose[item]])

                rule_without_item = rule['antecedents'].copy()
                rule_without_item.remove(item)
                new_rule.append(rule_without_item)
                new_rules.append((rule, new_rule, item))

        rules_wildcards = pd.DataFrame()
        wildcards_flag = set()
        for rule in new_rules:
            if rule[2] not in wildcards_flag:
                for x in rule[1]:
                    out = self._rules[
                            self._rules['antecedents'] == x]
                    if len(out) > 0:
                        values = [[rule[0]['antecedents'], rule[0]['consequents'],
                                rule[0]['score'],
                                out.iloc[0]['antecedents'],out.iloc[0]['consequents'],
                                out.iloc[0]['score'],
                                rule[2]]]
                        columns = ['rule', 'class', 'score',
                                'new_rule', 'new_class', 'new_score', 'wildcard']
                        wildcards_flag.add(rule[2])
                        rules_wildcards = pd.concat([rules_wildcards, pd.DataFrame(
                            values, columns=columns)], ignore_index=True)
        if len(rules_wildcards):
            rules_wildcards['diff'] = rules_wildcards['score']-rules_wildcards['new_score']
            rules_wildcards.sort_values(by=['diff'], ascending=False)
            return rules_wildcards.drop_duplicates(subset=['wildcard'])
        else:
            return rules_wildcards


    @property
    def fuzzy_sets_transpose(self):
        fuzzy_sets_dict = {}
        for key, value in self._fuzzy_sets.items():
            for val in value:
                fuzzy_sets_dict[val] = key
        return fuzzy_sets_dict

    @property
    def no_fuzzy_sets_transpose(self):
        no_fuzzy_sets_dict = {}
        for key, value in self._no_fuzzy_sets.items():
            for val in value:
                no_fuzzy_sets_dict[val] = key
        return no_fuzzy_sets_dict

    def predict(self, data):
        trigger_rules = self.trigger_rules(data).reset_index(drop=True)
        if not trigger_rules.empty:
            return list(trigger_rules.loc[0, 'consequents'])[0]
        else:
            return self._default_class

    def predict_all(self, x_test):
        predictions = []
        for _, data in x_test.iterrows():
            predictions.append(self.predict(data))
        return predictions
