from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseJudge(ABC):
    """
    Abstract base class for all judges.
    A judge is responsible for producing qualitative or quantitative
    judgments from system outputs (e.g., debates, syntheses).
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def judge(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform judgment on the input data.

        Args:
            input_data (dict): Output from debate or expert synthesis

        Returns:
            dict: Structured judgment results (scores, rationales, labels)
        """
        pass
