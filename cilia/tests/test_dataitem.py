import pytest
import tempfile
import os
import sys
import numpy as np
import torch

# ensure 'src' is on import path for the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cilia.datastructures.DataName import TANGENT_ANGLE_SERIES
from cilia.datasets.DataItem import DataItem
from cilia.datasets.data_loaders import _dataitem_loader_registry


class TestDataItem:
    def test_dataitem_creation(self):
        """Test DataItem creation and ID generation."""
        data_name = TANGENT_ANGLE_SERIES
        data = {"test": "data"}
        dependencies = ["dep1", "dep2"]
        algorithm = "test_algo"
        
        item = DataItem(
            data_name=data_name,
            data=data,
            dependencies=dependencies,
            algorithm=algorithm
        )
        
        assert item.data_name == data_name
        assert item.data == data
        assert item.dependencies == dependencies
        assert item.algorithm == algorithm
        assert item.id.startswith(f"{data_name.key}_")
        assert len(item.id) > len(data_name.key) + 1

    def test_dataitem_id_uniqueness(self):
        """Test that different inputs produce different IDs."""
        item1 = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={},
            dependencies=["a"],
            algorithm="algo1"
        )
        item2 = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={},
            dependencies=["b"],
            algorithm="algo1"
        )
        assert item1.id != item2.id

    def test_dataitem_id_is_dependency_order_independent(self):
        """IDs should not change when equivalent dependency lists are reordered."""
        item1 = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={},
            dependencies=["dep2", "dep1"],
            algorithm="algo1"
        )
        item2 = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={},
            dependencies=["dep1", "dep2"],
            algorithm="algo1"
        )
        assert item1.id == item2.id

    def test_dataitem_save_load(self, tmp_path):
        """Test saving and loading DataItem."""
        # Create test data
        test_data = {
            "array": np.array([1, 2, 3]),
            "scalar": 42,
            "dict": {"nested": "value"}
        }
        
        # use a temporary DataName whose schema accepts anything so we
        # can exercise the persistence logic without worrying about
        # validation rules from a real DataName.
        class AnySchema:
            def validate(self, data):
                return None

        from cilia.datastructures.DataName import DataName
        permissive = DataName(key="permissive_test", schema=AnySchema())

        item = DataItem(
            data_name=permissive,
            data=test_data,
            dependencies=[],
            algorithm="test"
        )
        
        # Save
        item.save(str(tmp_path))
        
        # Check files created
        files = list(tmp_path.glob("*"))
        assert len([f for f in files if f.suffix == ".json"]) == 1
        assert len([f for f in files if f.suffix in [".npy", ".yaml"]]) >= 1
        
        # Check JSON metadata
        json_file = next(tmp_path.glob("*.json"))
        import json
        with open(json_file) as f:
            meta = json.load(f)
        assert meta["data_name_key"] == permissive.key
        # top-level data is now a wrapper containing metadata and filenames
        assert "filenames" in meta["data"]
        assert "metadata" in meta["data"]
        # simple values should appear inside metadata
        assert meta["data"]["metadata"]["scalar"] == 42
        # array field has been replaced by a file marker
        assert isinstance(meta["data"]["metadata"]["array"], dict)
        assert "__file__" in meta["data"]["metadata"]["array"]
        
        # Simulate loading (create new item with saved data)
        loaded_item = DataItem(
            data_name=permissive,
            data=meta["data"],
            dependencies=meta["dependencies"],
            algorithm=meta["algorithm"]
        )
        
        # Mock dataset with default_loader
        class MockDataset:
            def default_loader(self, exp_path, data_meta):
                result = {}
                filenames = data_meta.get("filenames", [])
                for fname in filenames:
                    fpath = os.path.join(exp_path, fname)
                    if fname.endswith(".npy"):
                        result[fname.split("__")[0]] = np.load(fpath)
                    elif fname.endswith(".yaml"):
                        import yaml
                        with open(fpath) as yf:
                            result[fname.split("__")[0]] = yaml.safe_load(yf)
                return result
        
        dataset = MockDataset()
        loaded_data = loaded_item.resolve(dataset, str(tmp_path))
        
        # Check loaded data matches original
        np.testing.assert_array_equal(loaded_data["array"], test_data["array"])
        assert loaded_data["scalar"] == test_data["scalar"]
        assert loaded_data["dict"] == test_data["dict"]

    def test_dataitem_save_load_complex(self, tmp_path):
        """Lists containing nested dicts/arrays are pickled rather than split.

        Prior implementation produced many .npy files; now we expect a single
        pickle.  The JSON sidecar should simply contain a file marker and
        loading should reconstruct the original list exactly.
        """
        complex_list = [
            {"phase": np.arange(4), "aux": {"foo": np.ones(2)}},
            {"phase": np.arange(3), "aux": {"foo": np.zeros(2)}},
        ]
        class AnySchema:
            def validate(self, data):
                return None

        from cilia.datastructures.DataName import DataName
        dummy_name = DataName(key="complex_list_test", schema=AnySchema())

        item = DataItem(
            data_name=dummy_name,
            data=complex_list,
            dependencies=[],
            algorithm="test",
        )
        item.save(str(tmp_path))

        files = list(tmp_path.glob("*"))
        # expect exactly one .pkl file (not many .npy files)
        assert any(f.suffix == ".pkl" for f in files)
        assert not any(f.suffix == ".npy" for f in files)

        json_file = next(tmp_path.glob("*.json"))
        import json
        with open(json_file) as f:
            meta = json.load(f)
        assert "filenames" in meta["data"]
        md = meta["data"]["metadata"]
        # metadata should just be a file indicator for the whole list
        assert isinstance(md, dict) and "__file__" in md

        loaded_item = DataItem(
            data_name=dummy_name,
            data=meta["data"],
            dependencies=meta["dependencies"],
            algorithm=meta["algorithm"],
        )
        class MockDataset2:
            def default_loader(self, exp_path, data_meta):
                # default_loader will ignore pkl, DataItem.resolve loads it
                return {}
        loaded_data = loaded_item.resolve(MockDataset2(), str(tmp_path))
        assert isinstance(loaded_data, list)
        assert np.array_equal(loaded_data[0]["phase"], complex_list[0]["phase"])
        assert np.array_equal(loaded_data[0]["aux"]["foo"], complex_list[0]["aux"]["foo"])

    def test_dataitem_save_load_tensor(self, tmp_path):
        """Torch tensors are stored as .pt files and reconstructed from metadata."""
        class AnySchema:
            def validate(self, data):
                return None

        from cilia.datastructures.DataName import DataName
        name = DataName(key="tensor_test", schema=AnySchema())
        payload = {"tensor": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "label": "x"}
        item = DataItem(data_name=name, data=payload, dependencies=[], algorithm="x")

        item.save(str(tmp_path))

        assert any(f.suffix == ".pt" for f in tmp_path.glob("*"))

        import json
        json_file = next(tmp_path.glob("*.json"))
        with open(json_file) as f:
            meta = json.load(f)

        loaded_item = DataItem(
            data_name=name,
            data=meta["data"],
            dependencies=meta["dependencies"],
            algorithm=meta["algorithm"],
        )

        class MockDataset:
            def default_loader(self, exp_path, data_meta):
                raise RuntimeError("metadata reconstruction should not use default_loader")

        resolved = loaded_item.resolve(MockDataset(), str(tmp_path))
        assert torch.equal(resolved["tensor"], payload["tensor"])
        assert resolved["label"] == payload["label"]

    def test_dataitem_small_and_large_lists(self, tmp_path):
        """Simple lists are inline; large ones get pickled."""
        small = [1, 2, 3]  # below threshold
        class AnySchema:
            def validate(self, data):
                return None
        from cilia.datastructures.DataName import DataName
        name1 = DataName(key="small_list", schema=AnySchema())
        item1 = DataItem(data_name=name1, data=small, dependencies=[], algorithm="x")
        item1.save(str(tmp_path))
        # locate the JSON sidecar by reading all and picking our data_name
        import json
        meta1 = None
        for jf in tmp_path.glob("*.json"):
            with open(jf) as f:
                m = json.load(f)
            if m["data_name_key"] == name1.key:
                meta1 = m
                break
        assert meta1 is not None
        # small list should not be pickled; metadata is directly a list
        assert isinstance(meta1["data"]["metadata"], list)
        assert not any(f.suffix == ".pkl" for f in tmp_path.glob("*.pkl"))

        # list containing numpy arrays should also be pickled
        array_list = [np.arange(5), np.zeros((2,2))]
        name_arr = DataName(key="array_list", schema=AnySchema())
        item_arr = DataItem(data_name=name_arr, data=array_list, dependencies=[], algorithm="x")
        item_arr.save(str(tmp_path))
        assert any(f.suffix == ".pkl" for f in tmp_path.glob("*.pkl"))
        # ensure its metadata refers to pickle
        meta_arr = None
        for jf in tmp_path.glob("*.json"):
            with open(jf) as f:
                m = json.load(f)
            if m["data_name_key"] == name_arr.key:
                meta_arr = m
                break
        assert meta_arr is not None
        assert isinstance(meta_arr["data"]["metadata"], dict) and "__file__" in meta_arr["data"]["metadata"]

        # long numeric list triggers pickle
        long = list(range(200))
        name2 = DataName(key="long_list", schema=AnySchema())
        item2 = DataItem(data_name=name2, data=long, dependencies=[], algorithm="x")
        item2.save(str(tmp_path))
        # there should now be at least one .pkl file
        assert any(f.suffix == ".pkl" for f in tmp_path.glob("*.pkl"))
        # read the new metadata for long_list
        meta2 = None
        for jf in tmp_path.glob("*.json"):
            with open(jf) as f:
                m = json.load(f)
            if m["data_name_key"] == name2.key:
                meta2 = m
                break
        assert meta2 is not None
        loaded_item2 = DataItem(
            data_name=name2,
            data=meta2["data"],
            dependencies=meta2["dependencies"],
            algorithm=meta2["algorithm"],
        )
        class MockD:
            def default_loader(self, exp_path, data_meta):
                return {}
        out2 = loaded_item2.resolve(MockD(), str(tmp_path))
        assert out2 == long

    def test_dataitem_nested_dict(self, tmp_path):
        """Only the deepest arrays should be moved to files; dicts stay inline."""
        nested = {
            "level1": {
                "a": np.arange(2),
                "b": {"c": np.zeros(3), "d": 5},
            },
            "scalar": "hello",
        }
        class AnySchema:
            def validate(self, data):
                return None
        from cilia.datastructures.DataName import DataName
        name = DataName(key="nested_test", schema=AnySchema())
        item = DataItem(data_name=name, data=nested, dependencies=[], algorithm="x")
        item.save(str(tmp_path))
        json_file = next(tmp_path.glob("*.json"))
        import json
        with open(json_file) as f:
            meta = json.load(f)
        md = meta["data"]["metadata"]
        # ensure that 'a' and 'c' were replaced with file markers
        assert md["level1"]["a"]["__file__"].endswith(".npy")
        assert md["level1"]["b"]["c"]["__file__"].endswith(".npy")
        # scalar values remain inline
        assert md["level1"]["b"]["d"] == 5
        assert md["scalar"] == "hello"

    def test_dataitem_legacy_format_no_metadata(self, tmp_path):
        """Test loading legacy/intermediate data with filenames but no metadata key.
        
        This can happen when data was saved with older code or in transitional states.
        The resolve method should handle this gracefully by loading files explicitly.
        """
        class AnySchema:
            def validate(self, data):
                return None

        from cilia.datastructures.DataName import DataName
        name = DataName(key="legacy_test", schema=AnySchema())

        # Create a save .npy file manually
        test_array = np.array([1.5, 2.5, 3.5])
        npy_path = tmp_path / "data__legacy_test_999.npy"
        np.save(npy_path, test_array)

        # Create a DataItem with the legacy/intermediate format:
        # data = {"filenames": [...]} without metadata key
        legacy_data = {"filenames": ["data__legacy_test_999.npy"]}
        item = DataItem(
            data_name=name,
            data=legacy_data,
            dependencies=[],
            algorithm="legacy",
        )

        # Manually create the JSON sidecar to simulate legacy save
        import json
        json_obj = {
            "data_name_key": name.key,
            "dependencies": [],
            "algorithm": "legacy",
            "data": legacy_data,
        }
        json_path = tmp_path / f"{item.id}.json"
        with open(json_path, "w") as jf:
            json.dump(json_obj, jf)

        # Simulate loading: resolve should extract the array
        class MockDataset:
            def default_loader(self, exp_path, data_meta):
                return {}

        resolved = item.resolve(MockDataset(), str(tmp_path))
        np.testing.assert_array_equal(resolved, test_array)

    def test_dataitem_scalar_data(self):
        """Test that scalar data (old format) is handled transparently.
        
        Old code may have saved scalar values directly without wrapping.
        resolve() should return them unchanged.
        """
        class AnySchema:
            def validate(self, data):
                return None

        from cilia.datastructures.DataName import DataName
        name = DataName(key="scalar_test", schema=AnySchema())

        # Create item with scalar data (not wrapped)
        scalar_value = 0.001
        item = DataItem(
            data_name=name,
            data=scalar_value,
            dependencies=[],
            algorithm="test",
        )

        class MockDataset:
            def default_loader(self, exp_path, data_meta):
                # Should not be called for scalar data
                raise RuntimeError("default_loader should not be called for scalar")

        resolved = item.resolve(MockDataset(), "/tmp")
        assert resolved == scalar_value

    def test_dataitem_dict_without_filenames(self):
        """Test dict data that has no filenames (e.g., pre-wrapped metadata)."""
        class AnySchema:
            def validate(self, data):
                return None

        from cilia.datastructures.DataName import DataName
        name = DataName(key="dict_test", schema=AnySchema())

        # Dict without filenames (old style)
        dict_data = {"key1": "value1", "key2": 42}
        item = DataItem(
            data_name=name,
            data=dict_data,
            dependencies=[],
            algorithm="test",
        )

        class MockDataset:
            def default_loader(self, exp_path, data_meta):
                # For dict without filenames, just return it as-is
                return data_meta

        resolved = item.resolve(MockDataset(), "/tmp")
        assert resolved == dict_data

    def test_dataitem_registered_loader_takes_priority(self):
        """A registered loader should be used before built-in metadata handling."""
        class AnySchema:
            def validate(self, data):
                return None

        from cilia.datastructures.DataName import DataName
        name = DataName(key="registered_loader_test", schema=AnySchema())

        class MockDataset:
            def default_loader(self, exp_path, data_meta):
                raise RuntimeError("registered loader should take priority")

        key = (MockDataset, name.key, "algo")
        calls = []

        def loader(exp_path, data_meta):
            calls.append((exp_path, data_meta))
            return {"loaded": True, "source": data_meta["metadata"]}

        _dataitem_loader_registry[key] = loader
        try:
            item = DataItem(
                data_name=name,
                data={"metadata": {"inline": 1}, "filenames": []},
                dependencies=[],
                algorithm="algo",
            )

            resolved = item.resolve(MockDataset(), "/exp")

            assert resolved == {"loaded": True, "source": {"inline": 1}}
            assert calls == [("/exp", {"metadata": {"inline": 1}, "filenames": []})]
        finally:
            _dataitem_loader_registry.pop(key, None)

    def test_dataitem_cache(self):
        """Test that resolve caches loaded data."""
        item = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={"test": "data"},
            dependencies=[],
            algorithm="test"
        )
        
        # Initially not loaded
        assert item.loaded_data is None
        
        # Mock resolve
        item.loaded_data = "cached_data"
        
        # Should return cached
        assert item.resolve(None, "") == "cached_data"
        
        # Unload
        item.unload()
        assert item.loaded_data is None


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__]))
