from typing import (
    Optional,
    Tuple)


class Rater:
    """A class representing a rater."""

    def __init__(self,
                 name: str,
                 abbreviation: Optional[str] = None
                 ) -> None:
        """Constructs a rater.

        Args:
            name (str): The name of the rater.
            abbreviation (Optional[str]): The abbreviation of the rater (default=None).
        """
        super().__init__()
        self.name = name
        self.abbreviation = abbreviation

    def get_name(self) -> str:
        """Gets the name of the rater.

        Returns:
            str: The name of the rater.
        """
        return self.name

    def get_abbreviation(self) -> Optional[str]:
        """Gets the abbreviation of the rater.

        Returns:
            Optional[str]: The abbreviation of the rater if contained otherwise None.
        """
        return self.abbreviation

    def __str__(self) -> str:
        if self.abbreviation:
            return f'{self.name} ({self.abbreviation})'

        return self.name


class RaterFactory:
    """A factory class producing raters."""

    def __init__(self,
                 raters: Tuple[str, ...],
                 abbreviations: Optional[Tuple[str, ...]] = None
                 ) -> None:
        """Constructs a rater factory.

        Args:
            raters (Tuple[str, ...]): All available rater names.
            abbreviations (Optional[Tuple[str, ...]]): All abbreviations to the raters (default=None).
        """
        super().__init__()

        if abbreviations:
            assert len(raters) == len(raters), f'The number of raters must be equal to the number of abbreviations ' \
                                               f'({len(raters)} (raters) vs. {len(abbreviations)} (abbrev.))!'

        self.raters = raters
        self.abbreviations = abbreviations

    def produce(self, name: str) -> Rater:
        """Produces a new rater is possible.

        Args:
            name (str): The name of the rater.

        Returns:
            Rater: The constructed rater.
        """
        if name not in self.raters:
            raise ValueError(f'The name {name} is not specified as a rater!')

        name_idx = self.raters.index(name)
        if self.abbreviations:
            abbreviation = self.abbreviations[name_idx]

        else:
            abbreviation = None

        return Rater(name, abbreviation)
