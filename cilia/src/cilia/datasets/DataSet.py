from abc import ABC, abstractmethod
from typing import List, Optional
import os
import json
import yaml 
import numpy as np
from cilia.datastructures.DataName import DataName
from cilia.datastructures.special_source_names import (
    ORIGINAL, GIVEN, GROUNDTHRUTH
)
import torch 
from cilia.datasets.DataItem import DataItem
from cilia.datasets.Realization import Realization


class DataSet(ABC):
    def __init__(self, path: str):
        self.path = path
        self.folder_name = os.path.basename(os.path.normpath(path))

        self.realizations: List[Realization] = []

        self._initialize_realizations()
        self._load_realization_data()

    def __iter__(self):
        for realization in self.realizations:
            yield realization

    def __len__(self) -> int:
        return len(self.realizations)

    def add_data_to_realization(self, realization: Realization, item: DataItem) -> None:
        exp_path = os.path.join(self.path, realization.experiment_id)
        item.save(exp_path)
        realization.add_data_item(item)

    @abstractmethod
    def _initialize_realizations(self) -> None:
        pass

    def _load_realization_data(self) -> None:
        for realization in self.realizations:
            exp_path = os.path.join(self.path, realization.experiment_id)
            if not os.path.exists(exp_path):
                continue  # Skip if directory doesn't exist (e.g., in tests)
            for file_name in os.listdir(exp_path):
                if file_name.endswith(".json") and file_name != "MetaData.json":
                    json_path = os.path.join(exp_path, file_name)
                    try:
                        with open(json_path, "r") as f:
                            meta = json.load(f)
                    except Exception as e:
                        print(f"[ERROR] Failed to load JSON at: {json_path}")
                        raise e

                    data_name_key = meta["data_name_key"]
                    dependencies = meta.get("source_ids") or meta.get("dependencies", [])
                    if not isinstance(dependencies, list):
                        dependencies = [dependencies]
                    algorithm = meta["algorithm"]

                    data_meta = meta.get("data", {})
                    data_name = DataName.get(data_name_key)
                    item = DataItem(
                        data_name=data_name,
                        data=data_meta,
                        dependencies=dependencies,
                        algorithm=algorithm,
                    )
                    # # Load data immediately to keep hot
                    # item.resolve(self, exp_path)
                    realization.add_data_item(item)

    def default_loader(self, exp_path: str, data_meta: dict) -> dict:
        """
        Default loader: loads all keys in `data_meta` that are not 'filenames',
        and loads .npy, .npz, .yaml, .pt, and .pkl files listed under
        'filenames' into a dictionary.
        """
        result = {}

        # Include metadata keys that are not 'filenames'
        for key, value in data_meta.items():
            if key != "filenames":
                result[key] = value

        # Load filenames if present
        filenames = data_meta.get("filenames", [])
        for fname in filenames:
            fpath = os.path.join(exp_path, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"[default_loader] File not found: {fpath}")

            result_name = fname.split("__")[0]

            try:
                if fname.endswith(".npy") or fname.endswith(".npz"):
                    result[result_name] = np.load(fpath, allow_pickle=True)

                elif fname.endswith(".yaml"):
                    with open(fpath, "r") as yf:
                        result[result_name] = yaml.safe_load(yf)

                elif fname.endswith(".pt"):
                    result[result_name] = torch.load(
                        fpath,
                        map_location=torch.device("cpu"),
                        weights_only=True,
                    )

                elif fname.endswith(".pkl"):
                    import pickle
                    with open(fpath, "rb") as pf:
                        result[result_name] = pickle.load(pf)

                else:
                    raise ValueError(f"[default_loader] Unsupported file extension: {fname}")

            except Exception as e:
                raise RuntimeError(f"[default_loader] Failed to load {fpath}: {e}")

        return result

    def clean(self, delete_only: Optional[List[str]] = None, 
              keep_keys: Optional[List[str]] = None, 
              delete_only_exp_ids: Optional[List[str]] = None,
              dry = False) -> None:
        """
        Cleans realization directories by deleting DataItem files based on the given mode:
        - Default mode (delete_only=None, keep_keys=None): removes items without ORIGINAL/GIVEN/GROUNDTHRUTH keys (as before).
        - delete_only mode: when delete_only is a list of DataName keys, deletes only those items and leaves all others intact.
        - keep_keys mode: behaves like default but also preserves items whose name is in keep_keys.
        """
        # -----------------------------------------
        # Select realizations to process
        # -----------------------------------------
        if delete_only_exp_ids is not None:
            target_ids = set(delete_only_exp_ids)
            realizations = [
                r for r in self.realizations
                if r.experiment_id in target_ids
            ]
        else:
            realizations = self.realizations
        for realization in realizations:
            exp_path = os.path.join(self.path, realization.experiment_id)
            if not os.path.exists(exp_path):
                continue

            # Collect files to delete
            blacklist = set()
            whitelist = set()

            # Iterate over JSON metadata files
            for file_name in os.listdir(exp_path):
                if not file_name.endswith('.json') or file_name == 'MetaData.json':
                    continue
                json_path = os.path.join(exp_path, file_name)
                try:
                    with open(json_path, 'r') as f:
                        meta = json.load(f)
                except Exception as e:
                    print(f"[ERROR] Reading JSON: {json_path}: {e}")
                    continue
                name = meta.get('data_name_key')
                data_meta = meta.get('data', {})
                # `data` can be a dict with filenames or a scalar (e.g., float) with no files
                data_files = data_meta.get('filenames', []) if isinstance(data_meta, dict) else []

                # Determine deletion based on modes
                if delete_only is not None:
                    # Only delete specified keys
                    if name in delete_only:
                        blacklist.add(file_name)
                        for fname in data_files:
                            blacklist.add(fname)
                        if not dry:
                            realization.data_items.pop(os.path.splitext(file_name)[0], None)
                    # else: leave everything else
                else:
                    # Default deletion criteria
                    sources = meta.get('source_ids') or meta.get('dependencies') or []
                    if not isinstance(sources, list):
                        sources = [sources]
                    protected = any(
                        sid in (ORIGINAL.key, GIVEN.key, GROUNDTHRUTH.key)
                        for sid in sources
                    )
                    # Keep if protected or in keep_keys
                    if protected or (keep_keys and name in keep_keys):
                        whitelist.add(file_name)
                        for fname in data_files:
                            whitelist.add(fname)
                    else:
                        blacklist.add(file_name)
                        for fname in data_files:
                            blacklist.add(fname)
                        if not dry:
                            realization.data_items.pop(os.path.splitext(file_name)[0], None)

            # Remove blacklisted files (except whitelisted)
            for file_name in os.listdir(exp_path):
                fpath = os.path.join(exp_path, file_name)
                if file_name in blacklist and file_name not in whitelist:
                    try:
                        if not dry:
                            os.remove(fpath)
                        action = "Would delete" if dry else "Deleted"
                        print(f"[clean] {action} blacklisted file: {fpath}")
                    except Exception as e:
                        print(f"[ERROR] Deleting file: {fpath}: {e}")
                # Also remove stray binaries if not whitelisted
                elif file_name.endswith(('.npy', '.npz', '.pt', '.yaml', '.pkl')) and file_name not in whitelist and delete_only is None:
                    try:
                        if not dry:
                            os.remove(fpath)
                        action = "Would delete" if dry else "Deleted"
                        print(f"[clean] {action} unlisted data file: {fpath}")
                    except Exception as e:
                        print(f"[ERROR] Deleting file: {fpath}: {e}")
