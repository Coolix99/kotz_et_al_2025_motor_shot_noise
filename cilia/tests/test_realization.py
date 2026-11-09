import pytest
import os, sys

# ensure package import works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cilia.datastructures.DataName import TANGENT_ANGLE_SERIES
from cilia.datasets.DataItem import DataItem
from cilia.datasets.Realization import Realization


class TestRealization:
    def test_realization_creation(self):
        """Test Realization creation."""
        condition = "test_condition"
        experiment_id = "exp_001"
        
        realization = Realization(experiment_id)
        assert realization.experiment_id == experiment_id
        assert realization.data_items == {}

    def test_add_data_item(self):
        """Test adding DataItem to Realization."""
        realization = Realization("exp")
        
        item = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={},
            dependencies=[],
            algorithm="test"
        )
        
        realization.add_data_item(item)
        
        assert item.id in realization.data_items
        assert realization.data_items[item.id] == item

    def test_get_by_name(self):
        """Test getting DataItems by name."""
        realization = Realization("exp")
        
        item1 = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={},
            dependencies=[],
            algorithm="test1"
        )
        item2 = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={},
            dependencies=[],
            algorithm="test2"
        )
        
        realization.add_data_item(item1)
        realization.add_data_item(item2)
        
        results = realization.get_by_name(TANGENT_ANGLE_SERIES)
        assert len(results) == 2
        assert item1 in results
        assert item2 in results

    def test_unload_data(self):
        """Test unloading all cached data."""
        realization = Realization("exp")
        
        item1 = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={},
            dependencies=[],
            algorithm="test1"
        )
        item2 = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={},
            dependencies=[],
            algorithm="test2"
        )
        
        # Simulate loaded data
        item1.loaded_data = "data1"
        item2.loaded_data = "data2"
        
        realization.add_data_item(item1)
        realization.add_data_item(item2)
        
        # Unload
        realization.unload_data()
        
        assert item1.loaded_data is None
        assert item2.loaded_data is None


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__]))