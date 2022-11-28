import pandas as pd
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from data import make_dataset
from models import MODELS

N_CLASSES = [2, 5, 10, 20, 50]
SAMPLES_PER_CLASS = [10, 100, 1000]


def main() -> None:
    results = []
    for n_classes in tqdm(N_CLASSES, "Number of classes"):
        for samples_per_class in tqdm(SAMPLES_PER_CLASS, "Samples per class"):
            X, y = make_dataset(n_classes, samples_per_class)
            for name, model in tqdm(MODELS.items(), "Model"):
                result = cross_val_score(model, X, y, scoring="accuracy")
                results.append(
                    {
                        "name": name,
                        "n_classes": n_classes,
                        "samples_per_class": samples_per_class,
                        "result": result.mean(),
                    }
                )

    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv")


if __name__ == "__main__":
    main()
