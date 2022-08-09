from abc import (
    ABC,
    abstractmethod)
from typing import Any


class Converter(ABC):
    """An abstract converter class. The subclasses of this class are used to transfer and rearrange
    information from one type to another.
    """

    @abstractmethod
    def convert(self) -> Any:
        """Converts the data accordingly.

        Returns:
            Any: The converted data.
        """
        raise NotImplementedError()
