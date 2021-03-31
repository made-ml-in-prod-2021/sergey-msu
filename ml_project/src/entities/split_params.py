from dataclasses import dataclass


@dataclass()
class SplittingParams:
    """ Data splitting config ORM class. """
    val_size: float = 0.1
    random_state: int = 42
