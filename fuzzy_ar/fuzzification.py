import pandas as pd
from sklearn.cluster import KMeans


class Fuzzification():
    def __init__(self, dataset: pd.DataFrame, fuzzy_sets: dict,
                 cluster_feature: str):
        self._fuzzy_sets = fuzzy_sets
        self._features = fuzzy_sets.keys()
        self._dataset = dataset
        self._cluster_feature = cluster_feature

    def fuzzify(self):
        data_fuzzified = []
        data_membership_fuzzified = []
        for feature in self._features:
            data_normalized = self.normalize(self._dataset[feature])
            data_clustering = self.kmCluster(
                                data_normalized,
                                list(self._dataset[self._cluster_feature]))
            data_membership = self.membership(data_normalized, data_clustering)
            data_membership_fuzzified.append(data_membership)
            data_fuzzified.append(self.fuzzy(data_membership, feature))

        self._fuzzified_data = pd.DataFrame(data_fuzzified).transpose()
        self._membership_data = pd.DataFrame(
                                data_membership_fuzzified).transpose()

    def normalize(self, crispVal):
        converted = []
        for x in crispVal:
            x = float(x)
            converted.append(x)

        minimum = min(converted)
        maximum = max(converted)

        normalized = []

        for x in converted:
            x = 100 * ((x - minimum) / (maximum - minimum))
            normalized.append(x)

        return normalized

    def membership(self, normVal, clusterVal):
        cA = clusterVal[0]
        cB = clusterVal[1]
        cC = clusterVal[2]
        cD = clusterVal[3]

        mem_val = []

        # R-Function
        def A(x):
            if (x > cB):
                mem_A = 0
            elif (cA <= x <= cB):
                mem_A = (cB - x) / (cB - cA)
            elif (x < cA):
                mem_A = 1

            return mem_A

        # Triangle Function
        def B(x):

            if (x <= cA):
                mem_B = 0
            elif (cA < x <= cB):
                mem_B = (x - cA) / (cB - cA)
            elif (cB < x < cC):
                mem_B = (cC - x) / (cC - cB)
            elif (x >= cC):
                mem_B = 0

            return mem_B

        # Triangle Function
        def C(x):

            if (x <= cB):
                mem_C = 0
            elif (cB < x <= cC):
                mem_C = (x - cB) / (cC - cB)
            elif (cC < x < cD):
                mem_C = (cD - x) / (cD - cC)
            elif (x >= cD):
                mem_C = 0

            return mem_C

        # L-Function
        def D(x):

            if (x < cC):
                mem_D = 0
            elif (cC <= x <= cD):
                mem_D = (x - cC) / (cD - cC)
            elif (x > cD):
                mem_D = 1

            return mem_D

        for x in normVal:
            mem_val.append([A(x), B(x), C(x), D(x)])
        # mem_vals.append(mem_val)

        return mem_val

    def fuzzy(self, memsVal, col_header):
        fuzzy_vals = self._fuzzy_sets.get(col_header)

        final_fuzzy = []
        for mem in memsVal:
            y = mem.index(max(mem))
            final_fuzzy.append(fuzzy_vals[y])

        return final_fuzzy

    def kmCluster(self, toCluster, weekNo):
        kmc = []
        for x, y in zip(weekNo, toCluster):
            # smc = []
            smc = [x, y]
            kmc.append(smc)
        # print(kmc)
        mem_means = []

        kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0)
        kmeans.fit(kmc)

        kk = kmeans.cluster_centers_
        # print(kk)
        for x, y in kk:
            mem_means.append(y)

        return sorted(mem_means)

    @property
    def fuzzified_data(self):
        fuzzified_df = self._dataset[set(self._dataset.columns.to_list()) -
                                     set(self._features)]
        fuzzified_df = fuzzified_df.copy()
        fuzzified_df.loc[:, self._cluster_feature] = self._dataset.loc[
                            :, self._cluster_feature].apply(
                                lambda x: f"MONTH_{str(x)}")
        for index, feature in enumerate(self._features):
            fuzzified_df.insert(0, feature, self._fuzzified_data[index])
            fuzzified_df.insert(0, f"MEM_{feature}",
                                self._membership_data[index])
        return fuzzified_df

    @property
    def membership_data(self):
        fuzzified_membership_df = pd.DataFrame()
        for index, feature in enumerate(self._features):
            fuzzified_membership_df.insert(0, feature,
                                           self._membership_data[index])
        return fuzzified_membership_df
