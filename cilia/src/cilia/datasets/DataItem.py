from dataclasses import dataclass, field
from typing import Any, List
import hashlib
import os
import json
import yaml
import numpy as np
import torch
from cilia.datastructures.DataName import DataName
from cilia.datasets.data_loaders import _dataitem_loader_registry

@dataclass
class DataItem:
    data_name: DataName  
    data: Any  # Serialized data (e.g., {"filenames": [...]})
    dependencies: List[str]  # IDs of direct input DataItems 
    algorithm: str  # Algorithm that produced this DataItem
    id: str = field(init=False)
    loaded_data: Any = field(default=None, init=False)  # Cached loaded data

    def __post_init__(self):
        self.id = self.generate_id()

    def generate_id(self) -> str:
        # Use data_name.key for consistency
        base_str = f"{self.data_name.key}|{sorted(self.dependencies)}|{self.algorithm}"
        hash_digest = hashlib.sha256(base_str.encode()).hexdigest()[:10]
        return f"{self.data_name.key}_{hash_digest}"

    def unload(self) -> None:
        """Unload cached data to free memory."""
        self.loaded_data = None


    def resolve(self, dataset, exp_path: str) -> Any:
        if self.loaded_data is not None:
            return self.loaded_data

        # Loader resolution
        def resolve_loader(dataset_cls, name, algorithm):
            for key in [
                (dataset_cls, name, algorithm),
                (dataset_cls, name, None),
                (dataset_cls, None, algorithm),
                (dataset_cls, None, None),
            ]:
                if key in _dataitem_loader_registry:
                    return _dataitem_loader_registry[key]
            return None

        loader = resolve_loader(type(dataset), self.data_name.key, self.algorithm)
        data_meta = self.data

        #  PRIORITY: use loader if available
        if loader is not None:
            resolved = loader(exp_path, data_meta)
        else:
            # Helper to load files explicitly
            def _load_files(filenames):
                loaded_map = {}
                for fname in filenames:
                    fpath = os.path.join(exp_path, fname)
                    if fname.endswith(".npy") or fname.endswith(".npz"):
                        loaded_map[fname] = np.load(fpath, allow_pickle=True)
                    elif fname.endswith(".pt"):
                        loaded_map[fname] = torch.load(
                            fpath,
                            map_location=torch.device("cpu"),
                            weights_only=True,
                        )
                    elif fname.endswith(".yaml"):
                        with open(fpath, "r") as yf:
                            loaded_map[fname] = yaml.safe_load(yf)
                    elif fname.endswith(".pkl"):
                        import pickle
                        with open(fpath, "rb") as pf:
                            loaded_map[fname] = pickle.load(pf)
                    else:
                        raise ValueError(f"Unsupported extension in {fname}")
                return loaded_map

            # Reconstruct from template and loaded files
            def _reconstruct(template, loaded_map):
                if isinstance(template, dict):
                    if "__file__" in template:
                        return loaded_map.get(template["__file__"])
                    return {k: _reconstruct(v, loaded_map) for k, v in template.items()}
                elif isinstance(template, list):
                    return [_reconstruct(v, loaded_map) for v in template]
                else:
                    return template

            # Fallback logic (unchanged, but NO loader here)
            if isinstance(data_meta, dict) and "metadata" in data_meta and "filenames" in data_meta:
                loaded_map = _load_files(data_meta["filenames"])
                resolved = _reconstruct(data_meta["metadata"], loaded_map)

            elif isinstance(data_meta, dict) and "filenames" in data_meta and data_meta.get("filenames"):
                loaded_map = _load_files(data_meta["filenames"])
                resolved = list(loaded_map.values())[0] if len(loaded_map) == 1 else loaded_map

            elif isinstance(data_meta, dict):
                resolved = dataset.default_loader(exp_path, data_meta)

            else:
                resolved = data_meta

        # Normalization 
        if isinstance(resolved, dict) and set(resolved.keys()) == {"data"}:
            resolved = resolved["data"]
        if isinstance(resolved, np.ndarray) and resolved.dtype == object:
            resolved = list(resolved)
        if isinstance(resolved, list):
            cleaned = []
            for x in resolved:
                if isinstance(x, np.ndarray) and x.dtype == object:
                    cleaned.append(np.asarray(x, dtype=float))
                else:
                    cleaned.append(x)
            resolved = cleaned

        # Validation
        schema = self.data_name.schema
        if schema is not None:
            # print(exp_path,self.id)
            schema.validate(resolved)

        self.loaded_data = resolved
        return self.loaded_data
        
    def save(self, exp_path: str) -> None:
        os.makedirs(exp_path, exist_ok=True)

        # Recursively walk the payload, externalizing numpy arrays and other
        # non-JSON values to separate files.  Returned object is a mostly-
        # JSON-friendly copy containing ``{"__file__": filename}``
        # placeholders wherever something was spooled out.
        filenames: list[str] = []

        def _serialize(obj, prefix):
            # simple scalars pass through
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj

            # array and tensor handling unchanged
            if isinstance(obj, np.ndarray):
                fname = f"{prefix}.npy"
                fpath = os.path.join(exp_path, fname)
                np.save(fpath, obj, allow_pickle=True)
                filenames.append(fname)
                return {"__file__": fname}
            if isinstance(obj, torch.Tensor):
                fname = f"{prefix}.pt"
                fpath = os.path.join(exp_path, fname)
                torch.save(obj, fpath)
                filenames.append(fname)
                return {"__file__": fname}

            # detect whether we should pickle an entire list/tuple
            if isinstance(obj, (list, tuple)):
                # compute simple heuristics: total scalar count and presence of
                # complex items (arrays, tensors, dicts, lists)
                count = 0
                complex_flag = False

                def _count(x):
                    nonlocal count, complex_flag
                    if isinstance(x, (str, bool)):
                        count += 1
                    elif isinstance(x, (int, float)):
                        count += 1
                    elif isinstance(x, np.ndarray):
                        count += x.size
                        complex_flag = True
                    elif isinstance(x, torch.Tensor):
                        count += x.numel()
                        complex_flag = True
                    elif isinstance(x, dict):
                        complex_flag = True
                        for y in x.values():
                            _count(y)
                    elif isinstance(x, (list, tuple)):
                        # don't mark complex just because it's a list; inspect elements
                        for y in x:
                            _count(y)
                    else:
                        # any other object treat as complex
                        complex_flag = True
                        count += 1

                _count(obj)
                # decide to pickle if we saw complex content or too many totals
                if complex_flag or count > 100:
                    fname = f"{prefix}.pkl"
                    fpath = os.path.join(exp_path, fname)
                    import pickle
                    with open(fpath, "wb") as pf:
                        pickle.dump(obj, pf)
                    filenames.append(fname)
                    return {"__file__": fname}
                # else fall through to elementwise serialization
                return [_serialize(v, f"{prefix}__{i}") for i, v in enumerate(obj)]

            if isinstance(obj, dict):
                return {k: _serialize(v, f"{prefix}__{k}") for k, v in obj.items()}

            # fallback for any other object
            fname = f"{prefix}.yaml"
            fpath = os.path.join(exp_path, fname)
            with open(fpath, "w") as yf:
                yaml.dump(obj, yf, default_flow_style=False, sort_keys=False)
            filenames.append(fname)
            return {"__file__": fname}

        cleaned = _serialize(self.data, f"data__{self.id}")
        self.data = {"metadata": cleaned, "filenames": filenames}

        # Save .json sidecar with metadata
        json_obj = {
            "data_name_key": self.data_name.key,  
            "dependencies": self.dependencies,
            "algorithm": self.algorithm,
            "data": self.data
        }
        json_path = os.path.join(exp_path, f"{self.id}.json")
        with open(json_path, "w") as jf:
            json.dump(json_obj, jf, indent=4)


# Note: _dataitem_loader_registry is defined in DataSet.py, so this import or reference might need adjustment.
