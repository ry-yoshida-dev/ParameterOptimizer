from enum import Enum

from optuna.study import StudyDirection


class Direction(Enum):
    """
    Whether the objective value should become smaller or larger.

    Attributes
    ----------
    MINIMIZE : str
        Smaller is better.
    MAXIMIZE : str
        Larger is better.
    """

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

    def to_study_direction(self) -> StudyDirection:
        """
        Convert to the Optuna equivalent.

        Returns
        -------
        StudyDirection
            Direction understood by Optuna.
        """
        match self:
            case Direction.MINIMIZE:
                return StudyDirection.MINIMIZE
            case Direction.MAXIMIZE:
                return StudyDirection.MAXIMIZE
