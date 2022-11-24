from sklearn.datasets import make_classification
import numpy.typing as npt


def make_dataset(n_classes: int, class_size: int) -> tuple[npt.NDArray, npt.NDArray]:
    return make_classification(
        n_samples=n_classes * class_size,
        n_informative=10,
        n_redundant=0,
        n_features=20,
        n_classes=n_classes,
        random_state=12345,
    )
