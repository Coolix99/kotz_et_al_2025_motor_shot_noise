import pytest
import tempfile
import os
import sys
import numpy as np
import torch
from unittest.mock import Mock

# ensure package import works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cilia.datastructures.DataName import TANGENT_ANGLE_SERIES
from cilia.datastructures.special_source_names import ORIGINAL
from cilia.datasets.DataSet import DataSet
from cilia.datasets.Realization import Realization
from cilia.datasets.DataItem import DataItem


class MockDataSet(DataSet):
    def _initialize_realizations(self):
        self.realizations = [
            Realization("exp1"),
            Realization("exp2")
        ]


class TestDataSet:
    def test_dataset_creation(self, tmp_path):
        """Test DataSet initialization."""
        dataset = MockDataSet(str(tmp_path))
        
        assert len(dataset.realizations) == 2
        assert dataset.realizations[1].experiment_id == "exp2"

    def test_dataset_iteration(self, tmp_path):
        """Test iterating over dataset."""
        dataset = MockDataSet(str(tmp_path))
        
        realizations = []
        for real in dataset:
            realizations.append(real)
        assert len(realizations) == 2

    def test_add_data_to_realization(self, tmp_path):
        """Test adding data to realization, including nested structures."""
        dataset = MockDataSet(str(tmp_path))
        realization = dataset.realizations[0]
        
        nested = {"outer": {"arr": np.array([1, 2])}, "value": 7}
        item = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data=nested,
            dependencies=[],
            algorithm="test"
        )
        
        dataset.add_data_to_realization(realization, item)
        
        # Check item was added
        assert item.id in realization.data_items
        
        # Check files were created and metadata structure
        exp_path = tmp_path / "exp1"
        assert exp_path.exists()
        json_files = list(exp_path.glob("*.json"))
        assert len(json_files) == 1
        # inspect JSON to ensure nested array replaced by marker
        with open(json_files[0]) as jf:
            import json
            meta = json.load(jf)
        md = meta["data"]["metadata"]
        assert "arr" in md["outer"]
        assert "__file__" in md["outer"]["arr"]

    def test_default_loader(self, tmp_path):
        """Test default loader."""
        dataset = MockDataSet(str(tmp_path))
        
        # Create test files
        data_meta = {
            "filenames": ["test__id.npy", "scalar__id.yaml"],
            "other": "value"
        }
        
        exp_path = tmp_path / "test"
        exp_path.mkdir(parents=True)
        
        # Create numpy file
        import numpy as np
        np.save(exp_path / "test__id.npy", np.array([1, 2, 3]))
        
        # Create yaml file
        import yaml
        with open(exp_path / "scalar__id.yaml", "w") as f:
            yaml.dump(42, f)
        
        result = dataset.default_loader(str(exp_path), data_meta)
        
        assert "other" in result
        assert result["other"] == "value"
        np.testing.assert_array_equal(result["test"], np.array([1, 2, 3]))
        assert result["scalar"] == 42
        # add pickle file
        import pickle
        with open(exp_path / "pickled__id.pkl", "wb") as pf:
            pickle.dump({"foo": 123}, pf)
        data_meta2 = {"filenames": ["pickled__id.pkl"]}
        result2 = dataset.default_loader(str(exp_path), data_meta2)
        assert isinstance(result2.get("pickled"), dict)
        assert result2["pickled"]["foo"] == 123

        torch.save(torch.tensor([1.0, 2.0]), exp_path / "tensor__id.pt")
        data_meta3 = {"filenames": ["tensor__id.pt"]}
        result3 = dataset.default_loader(str(exp_path), data_meta3)
        assert torch.equal(result3["tensor"], torch.tensor([1.0, 2.0]))

    def test_load_realization_data_from_json(self, tmp_path):
        """Existing JSON sidecars should be loaded into matching realizations."""
        exp_path = tmp_path / "exp1"
        exp_path.mkdir(parents=True)

        import json
        with open(exp_path / "MetaData.json", "w") as f:
            json.dump({"ignored": True}, f)
        with open(exp_path / "item.json", "w") as f:
            json.dump(
                {
                    "data_name_key": TANGENT_ANGLE_SERIES.key,
                    "source_ids": ORIGINAL.key,
                    "algorithm": "from_disk",
                    "data": {"value": 1},
                },
                f,
            )

        dataset = MockDataSet(str(tmp_path))
        realization = dataset.realizations[0]

        assert len(realization.data_items) == 1
        item = next(iter(realization.data_items.values()))
        assert item.data_name == TANGENT_ANGLE_SERIES
        assert item.dependencies == [ORIGINAL.key]
        assert item.algorithm == "from_disk"
        assert item.data == {"value": 1}

    def test_default_loader_errors_for_missing_and_unsupported_files(self, tmp_path):
        """Default loader should fail clearly for bad filename metadata."""
        dataset = MockDataSet(str(tmp_path))
        exp_path = tmp_path / "test"
        exp_path.mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            dataset.default_loader(str(exp_path), {"filenames": ["missing.npy"]})

        bad_path = exp_path / "bad.txt"
        bad_path.write_text("unsupported")
        with pytest.raises(RuntimeError, match="Unsupported file extension"):
            dataset.default_loader(str(exp_path), {"filenames": ["bad.txt"]})

    def test_clean_default_keeps_protected_and_removes_derived_and_stray_pickle(self, tmp_path):
        """Default clean preserves source-backed data and removes derived outputs."""
        dataset = MockDataSet(str(tmp_path))
        realization = dataset.realizations[0]

        protected = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data=np.array([[1.0, 2.0]]),
            dependencies=[ORIGINAL.key],
            algorithm="protected",
        )
        derived = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data=np.array([[3.0, 4.0]]),
            dependencies=[protected.id],
            algorithm="derived",
        )

        dataset.add_data_to_realization(realization, protected)
        dataset.add_data_to_realization(realization, derived)

        exp_path = tmp_path / "exp1"
        stray_pickle = exp_path / "stray.pkl"
        stray_pickle.write_bytes(b"not referenced")

        protected_json = exp_path / f"{protected.id}.json"
        derived_json = exp_path / f"{derived.id}.json"
        protected_file = exp_path / protected.data["filenames"][0]
        derived_file = exp_path / derived.data["filenames"][0]

        dataset.clean()

        assert protected_json.exists()
        assert protected_file.exists()
        assert not derived_json.exists()
        assert not derived_file.exists()
        assert not stray_pickle.exists()
        assert protected.id in realization.data_items
        assert derived.id not in realization.data_items

    def test_clean_dry_run_does_not_delete_files_or_memory_items(self, tmp_path):
        """Dry clean should report candidates without mutating disk or realization state."""
        dataset = MockDataSet(str(tmp_path))
        realization = dataset.realizations[0]
        derived = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data=np.array([[3.0, 4.0]]),
            dependencies=[],
            algorithm="derived",
        )
        dataset.add_data_to_realization(realization, derived)

        exp_path = tmp_path / "exp1"
        derived_json = exp_path / f"{derived.id}.json"
        derived_file = exp_path / derived.data["filenames"][0]

        dataset.clean(dry=True)

        assert derived_json.exists()
        assert derived_file.exists()
        assert derived.id in realization.data_items


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__]))
