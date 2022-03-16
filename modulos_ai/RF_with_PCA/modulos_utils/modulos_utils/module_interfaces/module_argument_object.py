# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
from typing import Dict, List, Optional, Tuple

ALL_AVAILABLE_KEYWORDS = [
    "--input-data-file", "--output-data-file", "--config-choice-file",
    "--weights-dir", "--label-data-file", "--transformed-label-data-file",
    "--history-file", "--config-file", "--params-path", "--num-cp",
    "--predictions-paths", "--labels-paths", "--output-file",
    "--train-labels-paths"
]

ALL_AVAILABLE_FLAGS = ["--train-new", "--minimization"]


class ModuleArgumentObjectError(Exception):
    """Errors in the ModuleArgumentObject class.
    """
    pass


class ModuleArgumentObject:
    """Class used to save module command line arguments.
    """

    def __init__(self) -> None:
        """Initialize class by defining member variables and setting them to a
        default value.
        """
        self._keyword_args: Optional[Dict[str, str]] = None
        self._flags: Optional[List[str]] = None

    def add_keyword_args(self, argdict: Dict[str, str]) -> None:
        """Add keyword arguments to ModuleArgumentObject.

        Arguments:
            argdict (Dict[str, str]): Dictionaries of keywords and values.
        """
        for key in argdict:
            if key not in ALL_AVAILABLE_KEYWORDS:
                if "_" in key:
                    raise ModuleArgumentObjectError(
                        "Keywords must not contain an underscore.")
                elif key[:2] != "--":
                    raise ModuleArgumentObjectError(
                        f"{key} is not a valid keyword. All keywords "
                        "must start with two dashes, e.g. "
                        "'--input-data-file'.")
                else:
                    raise ModuleArgumentObjectError(
                        f"Keyword {key} is not a valid keyword for any "
                        "module.")
        if self._keyword_args is None:
            self._keyword_args = argdict
        else:
            self._keyword_args.update(argdict)
        return None

    def add_flags(self, flag_names: List[str]) -> None:
        """Add a list of names as flags to the ModuleArgumentObject.

        Args:
            flag_names (List[str]): List of flags to add.
        """
        for flag in flag_names:
            if flag not in ALL_AVAILABLE_FLAGS:
                if "_" in flag:
                    raise ModuleArgumentObjectError(
                        "Flags must not contain an underscore.")
                elif flag[:2] != "--":
                    raise ModuleArgumentObjectError(
                        f"{flag} is not a valid flag. All flags "
                        "must start with two dashes, e.g. '--train-new'.")
                else:
                    raise ModuleArgumentObjectError(
                        f"Keyword {flag} is not a valid flag for any "
                        "module.")
        if self._flags is None:
            self._flags = flag_names
        else:
            self._flags += flag_names

    def get_keyword_args(self) -> Dict[str, str]:
        """Get a dictionary of keyword arguments.

        Returns:
            Dict[str, str]: Dictionary with argument names and values as keys
                and values.
        """
        if self._keyword_args is None:
            return {}
        else:
            return self._keyword_args

    def get_flags(self) -> List[str]:
        """Get a list of flags.

        Returns:
            List[str]: List of flag names.
        """
        if self._flags is None:
            return []
        else:
            return self._flags

    def get_all(self) -> Tuple[Dict[str, str], List[str]]:
        """Get all arguments (keyword args and flags).

        Returns:
            Tuple[Dict[str, str], List[str]]: Tuple of keyword args and flags.
        """
        return self.get_keyword_args(), self.get_flags()
