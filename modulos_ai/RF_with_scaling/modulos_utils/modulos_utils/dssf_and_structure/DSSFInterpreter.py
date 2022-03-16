# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
import json
import logging
import os
import shutil
from typing import Tuple, Dict, List, Any

from modulos_utils.dssf_and_structure import DSSFComponent
from modulos_utils.dssf_and_structure import DSSFErrors
from modulos_utils.dssf_and_structure import structure_logging as struc_log
from modulos_utils.dssf_and_structure import dssf_converter
from modulos_utils.dssf_and_structure import dssf_utils


VERSION = dssf_utils.DSSF_VERSION
ID_PLACEHOLDER = "{id}"


class DSSFileInterpreter:
    """
    This class is reading in the json and checks if all the mandatory
    information is given. It fills the optional keys with default values, if
    they are not provided.
    """
    def __init__(self, json_path: str) -> None:
        self.components: Dict[str, DSSFComponent.ComponentClass] = {}
        self.opt_infos: Dict[str, Dict] = {}
        self.error_messages = ""
        self.json_path = json_path
        self.json_loaded = False
        self.dssf: Any = []
        self.version = ""
        self.input_optional_info: Dict = {}
        return None

    def initialize_interpreter(self) -> None:
        """Read the json file, get the version number, check that all
        mandatory fields are there and that the paths are unique.

        Raises:
            DSSFFormatError: Thrown if it is not a valid json file.
            DSSFEntryError: Thrown if not all mandatory keys are present or
            paths are not unique.
        """
        err_msg = "\nError!\n"
        try:
            with open(self.json_path) as json_file:
                self.dssf = json.load(json_file)
        except (json.JSONDecodeError, ValueError) as err:
            self.error_messages += (err_msg +
                                    "Could not read dataset_structure."
                                    f"json:\n{err}\n")
            raise DSSFErrors.DSSFFormatError(self.error_messages)
        if not isinstance(self.dssf, list):
            self.error_messages += (err_msg + "The DSSF format is outdated!")
            raise DSSFErrors.DSSFOldFormatError(self.error_messages)

        self.version = (self.dssf.pop(-1)["_version"]
                        if list(self.dssf[-1].keys())[0] == "_version" else "")
        if not self.version:
            raise DSSFErrors.DSSFVersionError(
                err_msg + "There is no version specifier! Please  check that "
                "your dssf conforms to the norms specified in version "
                f"{VERSION} of the dssf.")
        elif self.version != VERSION:
            try:
                is_old_version = float(self.version) < float(VERSION)
            except ValueError:
                raise DSSFErrors.DSSFVersionError(
                    err_msg + "The version is unknown. Please check the dssf "
                    f"manual for the specification of version {VERSION} and "
                    "update your DSSF.")
            if is_old_version:
                try:
                    old_version = self.version
                    dssf = dssf_converter.convert_legacy_to_latest_stable_dssf(
                        self.dssf + [{dssf_converter.VERSION_KEY:
                                      self.version}])
                    basename = os.path.basename(self.json_path)
                    dirname = os.path.dirname(self.json_path)
                    dssf_name = basename.split(".json")[0]
                    old_dssf_path = os.path.join(
                        dirname, f"{dssf_name}_version_{old_version}.json")
                    shutil.move(self.json_path, old_dssf_path)
                    with open(self.json_path, "w") as f:
                        json.dump(dssf, f)
                    dssf.pop(-1)
                    self.dssf = dssf
                    self.version = VERSION
                except DSSFErrors.DSSFKeywordNotValidInVersion:
                    raise
                except Exception:
                    raise DSSFErrors.DSSFVersionError(
                        err_msg + f"The version of the DSSF ({self.version})"
                        f" is not the latest one ({VERSION}) and updating it "
                        "automatically failed. Please update the DSSF "
                        "manually!")
            else:
                raise DSSFErrors.DSSFVersionError(
                    err_msg + "The version is "
                    "unknown. Please check the dssf "
                    "manual for the specification of "
                    f"version {VERSION} and update "
                    "your DSSF.")
        self._check_required_keys()
        self._check_uniqueness_of_paths()
        self._check_uniqueness_of_component_names()
        if self.error_messages:
            raise DSSFErrors.DSSFEntryError(err_msg + self.error_messages)
        self.input_optional_info = {i["name"]: (dict(i["optional_info"])
                                                if "optional_info" in i
                                                else {})
                                    for i in self.dssf}
        self._str_bools_to_booleans()
        self.json_loaded = True
        return None

    def _str_bools_to_booleans(self) -> None:
        """Check for booleans saved as strings in the key 'categorical' in
        the optional_info and save them as booleans.
        """
        for entry in self.dssf:
            if "optional_info" not in entry or entry["type"] == "table":
                continue
            opt_info = entry["optional_info"]
            not_cat = opt_info.get("not_categorical", None)
            try:
                if not_cat is not None:
                    entry["optional_info"]["not_categorical"] = \
                        dssf_utils.str_bools_to_booleans(not_cat)
            except ValueError:
                pass
            not_dt = opt_info.get("not_datetime", None)
            try:
                if not_dt is not None:
                    entry["optional_info"]["not_datetime"] = \
                        dssf_utils.str_bools_to_booleans(not_dt)
            except Exception:
                pass
        return None

    def _check_required_keys(self) -> None:
        """Check that all required keys are present and no invalid ones are
        present.
        """

        required_keys = ["path", "type", "name"]
        for n, comp in enumerate(self.dssf):
            for key in required_keys:
                if key not in comp:
                    self.error_messages += (
                        f"The required key '{key}' is missing in the " +
                        f"definition of composition {n + 1}.\n"
                    )
                    return None
                elif key == "type":
                    self._check_type()

            for key in comp:
                if key not in required_keys + ["optional_info"]:
                    msg = (f"The key '{key}' of '{comp['name']}' is invalid.\n"
                           f"The following keys are allowed: "
                           f"{required_keys  + ['optional_info']}.")
                    logging.error(msg)
                    self.error_messages += msg
        return None

    def _check_uniqueness_of_component_names(self) -> None:
        """Check uniqueness of component names."""
        component_names: List[str] = []
        for comp in self.dssf:
            if "name" in comp:
                name = comp["name"]
                if name in component_names:
                    self.error_messages += (
                        f"The DSSF component name '{name}' is "
                        "duplicated! But it has to be "
                        "unique!\n"
                    )
                else:
                    component_names.append(name)
        return None

    def _check_uniqueness_of_paths(self) -> None:
        """Check that the paths are unique.
        """
        paths: List[str] = []
        for comp in self.dssf:
            if "path" in comp:
                path = comp["path"]
                if path in paths:
                    self.error_messages += (
                        f"The path '{path}' exists in "
                        "multiple nodes! But it has to be "
                        "unique!\n"
                    )
                else:
                    paths.append(path)
        return None

    def _check_type(self) -> None:
        """Check that the type has a valid entry."""
        for comp in self.dssf:
            if comp["type"] not in ["num", "str", "table"]:
                self.error_messages += (
                    f"The type of component {comp['name']} ({comp['type']}) "
                    "is not valid."
                )
        return None

    def get_components(self) -> Dict[str, DSSFComponent.ComponentClass]:
        """Return a dict of the component objects.

        Returns:
            dict: Dictionary with name and object of all components.
        """
        return self.components

    def initialize_components(
            self, dataset_dir_path: str, batch_size: int, logging_purpose:
            struc_log.LoggingPurpose = struc_log.LoggingPurpose.INTERNAL) \
            -> Tuple[Dict, Dict, Tuple[Dict, Dict]]:
        """For each component initialize a ComponentClass object and fill
        the missing optional_info of self.dssf with default values. Check also
        the given optional_info is sensible.

        Returns:
            (dict, dict, (dict, dict)): dictionary with new and old optional
            info for each component.
        """
        if not self.json_loaded:
            self.initialize_interpreter()
        for entry in self.dssf:
            comp_class: DSSFComponent.ComponentClass

            if entry["type"] == "table":
                if isinstance(entry["path"], list):
                    comp_class = DSSFComponent.TableComponentMultiFile(
                        entry, dataset_dir_path, batch_size, logging_purpose)
                elif ID_PLACEHOLDER in entry["path"]:
                    comp_class = DSSFComponent.TableComponentOneFilePerSample(
                        entry, dataset_dir_path, batch_size, logging_purpose)
                else:
                    comp_class = DSSFComponent.TableComponentSingleFile(
                        entry, dataset_dir_path, batch_size, logging_purpose)
            else:
                comp_class = DSSFComponent.SingleNodeComponent(
                    entry, dataset_dir_path, batch_size, logging_purpose)

            opt_info = comp_class.opt_info
            self.opt_infos[entry["name"]] = opt_info
            self.components[entry["name"]] = comp_class
        return (self.opt_infos, self.input_optional_info,
                ({i["name"]: self.components[i["name"]].ignored_keys
                  for i in self.dssf},
                 {i["name"]: self.components[i["name"]].wrong_type_keys
                  for i in self.dssf}))
