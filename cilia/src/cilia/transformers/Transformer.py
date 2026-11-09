from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import hashlib
import inspect
import os

from tqdm import tqdm

from cilia.datastructures.DataName import DataName
from cilia.datasets.DataItem import DataItem
from cilia.datasets.DataSet import DataSet
from cilia.datasets.Realization import Realization


class Transformer:
    def __init__(
        self,
        func: Callable,
        inputs: Tuple[DataName, ...],
        outputs: Tuple[DataName, ...],
        algorithm: Optional[str] = None,
    ):
        self.func = func
        self.inputs = inputs
        self.outputs = outputs

        self.func_name = func.__name__
        self.module = func.__module__
        self.code_hash = self._compute_hash()

        self.algorithm = algorithm or (
            func.__name__ + "_" + self.module + "_" + self.code_hash
        )

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def _compute_hash(self) -> str:
        try:
            source = inspect.getsource(self.func)
            return hashlib.sha256(source.encode("utf-8")).hexdigest()[:10]
        except Exception:
            return "unknown"


def data_transformer(
    inputs: Tuple[DataName, ...],
    outputs: Tuple[DataName, ...],
    algorithm: Optional[str] = None,
):
    def decorator(func: Callable):
        if not isinstance(inputs, tuple) or not all(isinstance(i, DataName) for i in inputs):
            raise TypeError("Inputs must be a tuple of DataName instances")
        if not isinstance(outputs, tuple) or not all(isinstance(o, DataName) for o in outputs):
            raise TypeError("Outputs must be a tuple of DataName instances")
        return Transformer(func, inputs, outputs, algorithm)

    return decorator


def _compute_item_id(data_name: DataName, dependencies: Iterable[str], algorithm: str) -> str:
    base_str = f"{data_name.key}|{sorted(dependencies)}|{algorithm}"
    hash_digest = hashlib.sha256(base_str.encode()).hexdigest()[:10]
    return f"{data_name.key}_{hash_digest}"


def _exp_path(dataset: DataSet, realization: Realization) -> str:
    condition = getattr(realization, "condition", None)
    if condition:
        return os.path.join(dataset.path, condition, realization.experiment_id)
    return os.path.join(dataset.path, realization.experiment_id)


def _as_key(data_name_or_key: Union[DataName, str]) -> str:
    return data_name_or_key.key if isinstance(data_name_or_key, DataName) else data_name_or_key


def _as_data_name_map_key(data_name_or_key: Union[DataName, str]) -> str:
    return _as_key(data_name_or_key)


def _iter_dataset(dataset: DataSet) -> Iterable[Realization]:
    return iter(dataset)


def _item_ancestor_ids(realization: Realization, item: DataItem) -> set:
    ancestors = set()
    stack = list(item.dependencies)

    while stack:
        dep_id = stack.pop()
        if dep_id in ancestors:
            continue
        ancestors.add(dep_id)

        dep = realization.data_items.get(dep_id)
        if dep is not None:
            stack.extend(dep.dependencies)

    return ancestors


def _item_ancestor_ids_by_key(
    realization: Realization,
    item: DataItem,
    data_name_keys: Iterable[str],
) -> set:
    keys = set(data_name_keys)
    matches = set()
    stack = [item.id]
    visited = set()

    while stack:
        current_id = stack.pop()
        if current_id in visited:
            continue
        visited.add(current_id)

        current = realization.data_items.get(current_id)
        if current is None:
            continue

        if current.data_name.key in keys:
            matches.add(current.id)

        stack.extend(current.dependencies)

    return matches


def _share_ancestor_by_key(
    realization: Realization,
    items: Iterable[DataItem],
    data_name_keys: Iterable[str],
) -> bool:
    item_list = list(items)
    if len(item_list) <= 1:
        return True

    ancestor_sets = [
        _item_ancestor_ids_by_key(realization, item, data_name_keys)
        for item in item_list
    ]
    if any(not s for s in ancestor_sets):
        return False
    return bool(set.intersection(*ancestor_sets))


@dataclass(frozen=True)
class Selector:
    """Declarative DataItem selection.

    A selector is intentionally small: it describes what items are acceptable,
    and Pipeline is responsible for applying it per realization.
    """

    data_name: Optional[DataName] = None
    data_name_key: Optional[str] = None
    ids: Optional[frozenset] = None
    algorithms: Optional[frozenset] = None
    sources: Optional[frozenset] = None
    scoped_ids: Dict[str, frozenset] = field(default_factory=dict)
    predicate: Optional[Callable[[DataItem, Realization], bool]] = None

    @property
    def key(self) -> Optional[str]:
        if self.data_name is not None:
            return self.data_name.key
        return self.data_name_key

    def for_name(self, data_name: DataName) -> "Selector":
        if self.key is not None and self.key != data_name.key:
            raise ValueError(
                f"Selector for '{self.key}' cannot be used for '{data_name.key}'"
            )
        return self._replace(data_name=data_name, data_name_key=data_name.key)

    def only_ids(self, ids: Iterable[str]) -> "Selector":
        return self._replace(ids=frozenset(ids))

    def algorithm(self, *algorithms: str) -> "Selector":
        return self.algorithms_in(algorithms)

    def algorithms_in(self, algorithms: Iterable[str]) -> "Selector":
        return self._replace(algorithms=frozenset(algorithms))

    def source(self, *sources: Union[str, object]) -> "Selector":
        return self.sources_in(_source_key(s) for s in sources)

    def sources_in(self, sources: Iterable[Union[str, object]]) -> "Selector":
        return self._replace(sources=frozenset(_source_key(s) for s in sources))

    def where(
        self,
        *,
        source: Optional[Union[str, object]] = None,
        algorithm: Optional[str] = None,
        predicate: Optional[Callable[[DataItem, Realization], bool]] = None,
    ) -> "Selector":
        out = self
        if source is not None:
            out = out.source(source)
        if algorithm is not None:
            out = out.algorithm(algorithm)
        if predicate is not None:
            out = out._replace(predicate=predicate)
        return out

    def matches(self, item: DataItem, realization: Realization) -> bool:
        key = self.key
        if key is not None and item.data_name.key != key:
            return False

        if self.ids is not None and item.id not in self.ids:
            return False

        scoped = self.scoped_ids.get(realization.experiment_id)
        if scoped is not None and item.id not in scoped:
            return False

        if self.algorithms is not None and item.algorithm not in self.algorithms:
            return False

        if self.sources is not None and not set(item.dependencies).intersection(self.sources):
            return False

        if self.predicate is not None and not self.predicate(item, realization):
            return False

        return True

    def candidates(self, realization: Realization) -> List[DataItem]:
        return [
            item
            for item in realization.data_items.values()
            if self.matches(item, realization)
        ]

    def _replace(self, **kwargs) -> "Selector":
        values = {
            "data_name": self.data_name,
            "data_name_key": self.data_name_key,
            "ids": self.ids,
            "algorithms": self.algorithms,
            "sources": self.sources,
            "scoped_ids": self.scoped_ids,
            "predicate": self.predicate,
        }
        values.update(kwargs)
        return Selector(**values)


def _source_key(source: Union[str, object]) -> str:
    return getattr(source, "key", source) # type: ignore


def select(data_name: Optional[DataName] = None) -> Selector:
    return Selector(data_name=data_name, data_name_key=data_name.key if data_name else None)


@dataclass(frozen=True)
class PipelineResult:
    outputs: Dict[str, Selector]

    def output(self, data_name_or_key: Union[DataName, str]) -> Selector:
        return self.outputs[_as_key(data_name_or_key)]

    def only(self) -> Selector:
        if len(self.outputs) != 1:
            raise ValueError("PipelineResult has multiple outputs; use output(data_name)")
        return next(iter(self.outputs.values()))


class Pipeline:
    def __init__(self, dataset: DataSet, *, show_progress: bool = True):
        self.dataset = dataset
        self.show_progress = show_progress

    def run(
        self,
        transformer: Transformer,
        *,
        inputs: Optional[Dict[Union[DataName, str], Selector]] = None,
        realizations: Optional[Iterable[Realization]] = None,
        allowed_ids: Optional[Dict[str, Iterable[str]]] = None,
        allowed_algorithms: Optional[Iterable[str]] = None,
        allowed_algorithms_by_key: Optional[Dict[str, Iterable[str]]] = None,
        require_roots: Optional[Iterable[str]] = None,
        match_ancestor_keys: Optional[Iterable[Union[DataName, str]]] = None,
        skip_existing: bool = True,
    ) -> PipelineResult:
        output_ids = {dn.key: {} for dn in transformer.outputs}

        for realization in self._realizations(realizations, "Processing realizations"):
            created = self.run_on_realization(
                realization,
                transformer,
                inputs=inputs,
                allowed_ids=allowed_ids,
                allowed_algorithms=allowed_algorithms,
                allowed_algorithms_by_key=allowed_algorithms_by_key,
                require_roots=require_roots,
                match_ancestor_keys=match_ancestor_keys,
                skip_existing=skip_existing,
            )
            for key, ids in created.items():
                if ids:
                    output_ids[key][realization.experiment_id] = frozenset(ids)

        return PipelineResult(
            outputs={
                key: Selector(data_name_key=key, scoped_ids=scoped)
                for key, scoped in output_ids.items()
            }
        )

    def run_on_realization(
        self,
        realization: Realization,
        transformer: Transformer,
        *,
        inputs: Optional[Dict[Union[DataName, str], Selector]] = None,
        allowed_ids: Optional[Dict[str, Iterable[str]]] = None,
        allowed_algorithms: Optional[Iterable[str]] = None,
        allowed_algorithms_by_key: Optional[Dict[str, Iterable[str]]] = None,
        require_roots: Optional[Iterable[str]] = None,
        match_ancestor_keys: Optional[Iterable[Union[DataName, str]]] = None,
        skip_existing: bool = True,
    ) -> Dict[str, List[str]]:
        exp_path = _exp_path(self.dataset, realization)
        created = {dn.key: [] for dn in transformer.outputs}

        for combo in self.iter_combinations(
            transformer.inputs,
            realization,
            selectors=inputs,
            allowed_ids=allowed_ids,
            allowed_algorithms=allowed_algorithms,
            allowed_algorithms_by_key=allowed_algorithms_by_key,
            require_roots=require_roots,
            match_ancestor_keys=match_ancestor_keys,
        ):
            dep_ids = [item.id for item in combo]

            if skip_existing and self._outputs_exist(realization, transformer, dep_ids):
                for output in transformer.outputs:
                    created[output.key].append(
                        _compute_item_id(output, dep_ids, transformer.algorithm)
                    )
                continue

            resolved_inputs = [
                item.resolve(self.dataset, exp_path)
                for item in combo
            ]
            # print(realization.experiment_id)
            result = transformer(*resolved_inputs)
            self._validate_result(transformer, result)

            for data_name in transformer.outputs:
                new_item = DataItem(
                    data_name=data_name,
                    data=result[data_name.key],
                    dependencies=dep_ids,
                    algorithm=transformer.algorithm,
                )
                self.dataset.add_data_to_realization(realization, new_item)
                created[data_name.key].append(new_item.id)

        realization.unload_data()
        return created

    def view(
        self,
        inputs: Tuple[DataName, ...],
        *,
        viewer_func: Optional[Callable[..., None]] = None,
        on_combo: Optional[Callable[[Realization, Tuple[DataItem, ...], List[dict]], None]] = None,
        on_realization_end: Optional[Callable[[Realization, List[dict]], None]] = None,
        give_path_info: bool = False,
        realizations: Optional[Iterable[Realization]] = None,
        selectors: Optional[Dict[Union[DataName, str], Selector]] = None,
        allowed_ids: Optional[Dict[str, Iterable[str]]] = None,
        allowed_algorithms: Optional[Iterable[str]] = None,
        allowed_algorithms_by_key: Optional[Dict[str, Iterable[str]]] = None,
        require_roots: Optional[Iterable[str]] = None,
        match_ancestor_keys: Optional[Iterable[Union[DataName, str]]] = None,
        load_dependencies_of: Optional[Tuple[DataName, ...]] = None,
        dependency_depth: Optional[int] = None,
    ) -> None:
        dep_keys = {dn.key for dn in load_dependencies_of or ()}

        for realization in self._realizations(realizations, "Viewing realizations"):
            exp_path = _exp_path(self.dataset, realization)
            real_rows = {}

            for combo in self.iter_combinations(
                inputs,
                realization,
                selectors=selectors,
                allowed_ids=allowed_ids,
                allowed_algorithms=allowed_algorithms,
                allowed_algorithms_by_key=allowed_algorithms_by_key,
                require_roots=require_roots,
                match_ancestor_keys=match_ancestor_keys,
            ):
                resolved_inputs = [
                    item.resolve(self.dataset, exp_path)
                    for item in combo
                ]
                call_inputs = list(resolved_inputs)

                if give_path_info:
                    call_inputs.append({
                        "experiment_id": realization.experiment_id,
                        "dataset_path": self.dataset.path,
                    })

                if viewer_func is not None:
                    viewer_func(*call_inputs)

                combo_rows = []
                for item, value in zip(combo, resolved_inputs):
                    row = _row_from_item(realization, item, value)
                    real_rows[item.id] = row
                    combo_rows.append(row)

                if dep_keys:
                    deps = _collect_dependencies(
                        realization,
                        combo,
                        include_keys=dep_keys,
                        max_depth=dependency_depth,
                    )
                    for dep in deps:
                        real_rows.setdefault(
                            dep.id,
                            _row_from_item(realization, dep, dep.resolve(self.dataset, exp_path)),
                        )

                if on_combo is not None:
                    on_combo(realization, combo, list(combo_rows))

            if on_realization_end is not None:
                on_realization_end(realization, list(real_rows.values()))

            realization.unload_data()

    def iter_combinations(
        self,
        inputs: Tuple[DataName, ...],
        realization: Realization,
        *,
        selectors: Optional[Dict[Union[DataName, str], Selector]] = None,
        allowed_ids: Optional[Dict[str, Iterable[str]]] = None,
        allowed_algorithms: Optional[Iterable[str]] = None,
        allowed_algorithms_by_key: Optional[Dict[str, Iterable[str]]] = None,
        require_roots: Optional[Iterable[str]] = None,
        match_ancestor_keys: Optional[Iterable[Union[DataName, str]]] = None,
    ) -> Iterable[Tuple[DataItem, ...]]:
        selector_by_key = {
            _as_data_name_map_key(k): v
            for k, v in (selectors or {}).items()
        }
        id_filters = {k: set(v) for k, v in (allowed_ids or {}).items()}
        algo_set = set(allowed_algorithms) if allowed_algorithms else None
        algo_filters = {
            k: set(v)
            for k, v in (allowed_algorithms_by_key or {}).items()
        }

        candidate_lists = []
        for data_name in inputs:
            selector = selector_by_key.get(data_name.key, select(data_name)).for_name(data_name)
            candidates = selector.candidates(realization)

            if data_name.key in id_filters:
                candidates = [c for c in candidates if c.id in id_filters[data_name.key]]
            if algo_set is not None:
                candidates = [c for c in candidates if c.algorithm in algo_set]
            if data_name.key in algo_filters:
                candidates = [c for c in candidates if c.algorithm in algo_filters[data_name.key]]

            if not candidates:
                return
            candidate_lists.append(candidates)

        root_ids = set(require_roots or [])
        ancestor_keys = {_as_key(k) for k in (match_ancestor_keys or [])}

        for combo in product(*candidate_lists):
            if root_ids and not all(_item_matches_roots(realization, item, root_ids) for item in combo):
                continue
            if ancestor_keys and not _share_ancestor_by_key(realization, combo, ancestor_keys):
                continue
            yield combo

    def _realizations(
        self,
        realizations: Optional[Iterable[Realization]],
        desc: str,
    ) -> Iterable[Realization]:
        iterable = realizations if realizations is not None else _iter_dataset(self.dataset)
        if self.show_progress:
            return tqdm(iterable, desc=desc, unit="real")
        return iterable

    @staticmethod
    def _outputs_exist(
        realization: Realization,
        transformer: Transformer,
        dep_ids: Iterable[str],
    ) -> bool:
        return all(
            _compute_item_id(data_name, dep_ids, transformer.algorithm) in realization.data_items
            for data_name in transformer.outputs
        )

    @staticmethod
    def _validate_result(transformer: Transformer, result) -> None:
        if not isinstance(result, dict):
            raise ValueError("Transformer output must be a dict")

        for data_name in transformer.outputs:
            if data_name.key not in result:
                raise KeyError(
                    f"Transformer '{transformer.func_name}' did not return '{data_name.key}'"
                )


def _item_matches_roots(
    realization: Realization,
    item: DataItem,
    root_ids: set,
) -> bool:
    if item.id in root_ids:
        return True
    return bool(_item_ancestor_ids(realization, item).intersection(root_ids))


def _row_from_item(realization: Realization, item: DataItem, value) -> dict:
    return {
        "experiment_id": realization.experiment_id,
        "data_item_id": item.id,
        "data_name": item.data_name.key,
        "data_name_obj": item.data_name,
        "algorithm": item.algorithm,
        "dependencies": item.dependencies,
        "value": value,
    }


def _collect_dependencies(
    realization: Realization,
    start_items: Iterable[DataItem],
    include_keys: set,
    max_depth: Optional[int],
) -> List[DataItem]:
    collected = {}
    visited = set()
    stack = [(item, 0) for item in start_items]

    while stack:
        item, depth = stack.pop()

        if item.id in visited:
            continue
        visited.add(item.id)

        if max_depth is not None and depth >= max_depth:
            continue

        for dep_id in item.dependencies:
            dep = realization.data_items.get(dep_id)
            if dep is None:
                continue

            if dep.data_name.key in include_keys:
                collected.setdefault(dep.id, dep)

            stack.append((dep, depth + 1))

    return list(collected.values())


def run_transformer_on_dataset(
    dataset: DataSet,
    transformer: Transformer,
    *,
    allowed_ids: Optional[Dict[str, Iterable[str]]] = None,
    allowed_algorithms: Optional[Iterable[str]] = None,
    allowed_algorithms_by_key: Optional[Dict[str, Iterable[str]]] = None,
    require_roots: Optional[Iterable[str]] = None,
    match_ancestor_keys: Optional[Iterable[Union[DataName, str]]] = None,
    skip_existing: bool = True,
) -> PipelineResult:
    return Pipeline(dataset).run(
        transformer,
        allowed_ids=allowed_ids,
        allowed_algorithms=allowed_algorithms,
        allowed_algorithms_by_key=allowed_algorithms_by_key,
        require_roots=require_roots,
        match_ancestor_keys=match_ancestor_keys,
        skip_existing=skip_existing,
    )


def run_transformer_on_realization(
    dataset: DataSet,
    realization: Realization,
    transformer: Transformer,
    *,
    allowed_ids=None,
    allowed_algorithms=None,
    allowed_algorithms_by_key=None,
    require_roots=None,
    match_ancestor_keys=None,
    skip_existing=True,
) -> Dict[str, List[str]]:
    return Pipeline(dataset, show_progress=False).run_on_realization(
        realization,
        transformer,
        allowed_ids=allowed_ids,
        allowed_algorithms=allowed_algorithms,
        allowed_algorithms_by_key=allowed_algorithms_by_key,
        require_roots=require_roots,
        match_ancestor_keys=match_ancestor_keys,
        skip_existing=skip_existing,
    )


def run_viewer_on_dataset(
    dataset: DataSet,
    inputs: Tuple[DataName, ...],
    *,
    viewer_func: Optional[Callable[..., None]] = None,
    on_combo: Optional[Callable[[Realization, Tuple[DataItem, ...], List[dict]], None]] = None,
    on_realization_end: Optional[Callable[[Realization, List[dict]], None]] = None,
    realizations: Optional[Iterable[Realization]] = None,
    give_Path_info: bool = False,
    allowed_ids: Optional[Dict[str, Iterable[str]]] = None,
    allowed_algorithms: Optional[Iterable[str]] = None,
    allowed_algorithms_by_key: Optional[Dict[str, Iterable[str]]] = None,
    require_roots: Optional[Iterable[str]] = None,
    match_ancestor_keys: Optional[Iterable[Union[DataName, str]]] = None,
    load_dependencies_of: Optional[Tuple[DataName, ...]] = None,
    dependency_depth: Optional[int] = None,
) -> None:
    Pipeline(dataset).view(
        inputs,
        viewer_func=viewer_func,
        on_combo=on_combo,
        on_realization_end=on_realization_end,
        realizations=realizations,
        give_path_info=give_Path_info,
        allowed_ids=allowed_ids,
        allowed_algorithms=allowed_algorithms,
        allowed_algorithms_by_key=allowed_algorithms_by_key,
        require_roots=require_roots,
        match_ancestor_keys=match_ancestor_keys,
        load_dependencies_of=load_dependencies_of,
        dependency_depth=dependency_depth,
    )


def run_viewer_on_realization(
    dataset: DataSet,
    realization: Realization,
    inputs,
    *,
    viewer_func=None,
    allowed_ids=None,
    allowed_algorithms=None,
    allowed_algorithms_by_key=None,
    require_roots=None,
    match_ancestor_keys=None,
    match_roots: bool = False,
):
    return Pipeline(dataset, show_progress=False).view(
        inputs,
        viewer_func=viewer_func,
        realizations=[realization],
        allowed_ids=allowed_ids,
        allowed_algorithms=allowed_algorithms,
        allowed_algorithms_by_key=allowed_algorithms_by_key,
        require_roots=require_roots,
        match_ancestor_keys=match_ancestor_keys if not match_roots else require_roots,
    )
