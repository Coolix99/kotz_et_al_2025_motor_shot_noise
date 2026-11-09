from typing import Dict, List
from cilia.datasets.DataItem import DataItem
from cilia.datastructures.DataName import DataName

class Realization:
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.data_items: Dict[str, DataItem] = {}  # key = data_id

    def add_data_item(self, data_item: DataItem) -> None:
        self.data_items[data_item.id] = data_item

    def get_by_name(self, name: DataName) -> List[DataItem]:
        return [d for d in self.data_items.values() if d.data_name.key == name.key]

    def unload_data(self) -> None:
        """Unload all cached data in this realization to free memory."""
        for item in self.data_items.values():
            item.unload()
