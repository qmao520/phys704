# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This file contains code that is used for logging of the structure code.
"""
import enum
import logging

logging.getLogger().setLevel(logging.INFO)


class LoggingPurpose(enum.Enum):
    """Enum class for the different logging styles we have. Two of them are
    for internal logging (standard and debug mode) and one is for user-friendly
    prints in the batch client.
    """
    INTERNAL = 0
    INTERNAL_DEBUG = 1
    CLIENT = 2


class ComponentLoggingManager():
    """Log progress of component saving.
    """

    def __init__(self, name: str, logging_purpose: LoggingPurpose) -> None:
        """Initialize component logging manager.

        Args:
            name (str): Component name.
            logging_purpose (LoggingPurpose):  Logging purpose.
        """
        self._name: str = name
        self._logging_purpose: LoggingPurpose = logging_purpose
        return

    def log_name(self) -> None:
        """Log component name for client.
        """
        if self._logging_purpose == LoggingPurpose.CLIENT:
            print(f"\nComponent `{self._name}`:\t", end="\r")
        elif self._logging_purpose == LoggingPurpose.INTERNAL_DEBUG \
                or self._logging_purpose == LoggingPurpose.INTERNAL:
            logging.info(f"Saving Component `{self._name}`.")
        return None

    def log_percentage(self, percentage: float) -> None:
        """Log percentage to standard output be replacing the previous line, so
        that it gives the impression of an animation in the terminal output.

        Args:
            percentage (float): Percentage that should be printed.
        """
        if self._logging_purpose == LoggingPurpose.INTERNAL_DEBUG:
            logging.info(f"{percentage: .1f}% of the component "
                         f"`{self._name}` saved.")
        elif self._logging_purpose == LoggingPurpose.INTERNAL:
            if percentage == 100:
                logging.info(f"{percentage: .1f}% of the component "
                             f"`{self._name}` saved.")
        elif self._logging_purpose == LoggingPurpose.CLIENT:
            msg = (f"{percentage:.1f}%")
            print(f"Component `{self._name}`:\t{msg}", end="\r")
        return None

    def log_warning(self, warning: str) -> None:
        """Log a warning.

        Args:
            warning (str): Warning message to log.
        """
        if self._logging_purpose == LoggingPurpose.INTERNAL_DEBUG \
                or self._logging_purpose == LoggingPurpose.INTERNAL:
            logging.warning(warning)
        return None

    def log_error(self, error: str) -> None:
        """Log an error.

        Args:
            error (str): Error message to log.
        """
        if self._logging_purpose == LoggingPurpose.INTERNAL_DEBUG \
                or self._logging_purpose == LoggingPurpose.INTERNAL:
            logging.error(error)
        return None
