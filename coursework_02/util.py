import numpy as np
import scipy.stats as ss


class Util:

    def read_dataset(filepath):
        """
        Read dataset, used to deal with cw1 data file only
        """
        data = np.genfromtxt(filepath, dtype=str, delimiter=",")
        features = np.array(data[:, :-1], dtype=int)
        labels = data[:, -1]
        return (features, labels)

    def analyze_data(features, labels):
        """
        give descriptive data info
        """
        instance_num = features.shape[0]
        feature_num = features.shape[1]
        print(
            f"load {instance_num} data instances, each having {feature_num} features")
        labels_unique = np.unique(labels)
        labels_counts = dict(
            sorted({item: labels.tolist().count(item)
                    for item in labels}.items()))
        labels_dist = {
            key: str(round(labels_counts[key] / instance_num * 100)) + "%"
            for key in labels_counts}
        print(
            f"There are {labels_unique.size} unique label(s) with the following numbers and distribution")
        print(labels_counts)
        print(labels_dist)
        _ = np.array(list(labels_counts.values()))
        _ = _ / _.sum()
        # consider each percentage in a balanced data set should with in range(mean*0.8, mean*1.2)
        # and have a norm noise
        unbalanced_pos = np.logical_or(_ > _.mean() * 1.2, _ < _.mean() * 0.8)
        unbalanced = np.any(unbalanced_pos)
        shapiro = 0
        if _.size >= 3:
            shapiro = int(ss.shapiro(_)[1] > 0.05)
        print("The labels are " + ("unbalanced towards" if unbalanced else
                                   "slightly unbalanced" if shapiro else "balanced"))
        if np.any(unbalanced_pos):
            print(labels_unique[unbalanced_pos])
        return

    def analyze_feature(features, labels):
        """
        get some feature info
        """
        
        labels_unique = np.unique(labels)
        features_avg = np.array([
            features[labels == label].mean(axis=0) for label in labels_unique])
        features_std = np.array([
            features[labels == label].std(axis=0) for label in labels_unique])
        features_max = np.array([
            features[labels == label].max(axis=0) for label in labels_unique])
        features_min = np.array([
            features[labels == label].min(axis=0) for label in labels_unique])
        print("Average feature value for each char")
        for i in range(labels_unique.size):
            print(labels_unique[i], features_avg[i].astype(float))
        print("Standard error of features for each char")
        for i in range(labels_unique.size):
            print(labels_unique[i], features_std[i].astype(float))
            
        print("Max and min feature value for the feature")
        print("max:", features.max(axis = 0).astype(int))
        print("min:", features.min(axis = 0).astype(int))
        return

    def read_and_analyze(filepaths):
        """
        read and analyze data simultaneously
        """
        for filepath in filepaths:
            print(filepath.split("/")[-1], ":")
            features, labels = Util.read_dataset(filepath)
            Util.analyze_data(features, labels)
            Util.analyze_feature(features, labels)
            print()
        return
