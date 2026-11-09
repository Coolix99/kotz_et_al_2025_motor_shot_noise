import pytest
import os, sys
from unittest.mock import Mock
import numpy as np

# make sure package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cilia.datastructures.DataName import PHASE_SEGMENTS, TANGENT_ANGLE_SERIES, DataName
from cilia.datastructures.special_source_names import ORIGINAL
from cilia.datastructures.data_schemas import ScalarSchema

# create a temporary DataName for pca output (not defined in core)
try:
    TANGENT_ANGLE_PCA_SERIES = DataName(key="tangent_angle_pca_series", schema=ScalarSchema(float))
except ValueError:
    # already registered (possible when running tests directly multiple
    # times), just look it up instead
    TANGENT_ANGLE_PCA_SERIES = DataName.get("tangent_angle_pca_series")
from cilia.datasets.DataItem import DataItem
from cilia.datasets.Realization import Realization
from cilia.transformers.Transformer import (
    Pipeline,
    data_transformer,
    run_transformer_on_dataset,
    run_transformer_on_realization,
    run_viewer_on_realization,
    select,
)


class TestTransformer:
    def test_transformer_creation(self):
        """Test transformer creation."""
        @data_transformer(
            inputs=(TANGENT_ANGLE_SERIES,),
            outputs=(TANGENT_ANGLE_PCA_SERIES,)
        )
        def mock_transform(data):
            return {"tangent_angle_pca_series": "pca_data"}
        
        assert hasattr(mock_transform, 'inputs')
        assert hasattr(mock_transform, 'outputs')
        assert mock_transform.inputs == (TANGENT_ANGLE_SERIES,)
        assert mock_transform.outputs == (TANGENT_ANGLE_PCA_SERIES,)

    def test_transformer_creation_rejects_invalid_inputs(self):
        with pytest.raises(TypeError, match="Inputs must be a tuple of DataName instances"):
            @data_transformer(
                inputs=[TANGENT_ANGLE_SERIES], # type: ignore
                outputs=(TANGENT_ANGLE_PCA_SERIES,)
            )
            def invalid_transform(data):
                return {"tangent_angle_pca_series": "pca_data"}

    def test_data_names_are_hashable_by_key(self):
        selectors = {PHASE_SEGMENTS: select(PHASE_SEGMENTS)}

        assert selectors[PHASE_SEGMENTS].key == PHASE_SEGMENTS.key

    def test_run_transformer_on_dataset_rejects_non_dict_output(self):
        realization = Realization("exp")
        item = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={"filenames": []},
            dependencies=[],
            algorithm="original"
        )
        realization.add_data_item(item)

        dataset = Mock()
        dataset.path = ''
        dataset.__iter__ = Mock(side_effect=lambda: iter([realization]))
        dataset.default_loader = Mock(side_effect=lambda exp_path, meta: np.array([[0.0]]))
        dataset.add_data_to_realization = Mock()

        @data_transformer(
            inputs=(TANGENT_ANGLE_SERIES,),
            outputs=(TANGENT_ANGLE_PCA_SERIES,)
        )
        def bad_transform(data):
            return "not a dict"

        with pytest.raises(ValueError, match="Transformer output must be a dict"):
            run_transformer_on_dataset(dataset, bad_transform)
        assert not dataset.add_data_to_realization.called

    def test_run_transformer_on_dataset_requires_all_output_keys(self):
        realization = Realization("exp")
        item = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={"filenames": []},
            dependencies=[],
            algorithm="original"
        )
        realization.add_data_item(item)

        dataset = Mock()
        dataset.path = ''
        dataset.__iter__ = Mock(side_effect=lambda: iter([realization]))
        dataset.default_loader = Mock(side_effect=lambda exp_path, meta: np.array([[0.0]]))
        dataset.add_data_to_realization = Mock()

        @data_transformer(
            inputs=(TANGENT_ANGLE_SERIES,),
            outputs=(TANGENT_ANGLE_PCA_SERIES,)
        )
        def incomplete_transform(data):
            return {}

        with pytest.raises(KeyError, match="did not return 'tangent_angle_pca_series'"):
            run_transformer_on_dataset(dataset, incomplete_transform)
        assert not dataset.add_data_to_realization.called

    def test_run_transformer_on_dataset_skips_existing_outputs(self):
        realization = Realization("exp")
        item = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={"filenames": []},
            dependencies=[],
            algorithm="original"
        )
        realization.add_data_item(item)

        dataset = Mock()
        dataset.path = ''
        dataset.__iter__ = Mock(side_effect=lambda: iter([realization]))
        dataset.default_loader = Mock(side_effect=lambda exp_path, meta: np.array([[0.0]]))
        dataset.add_data_to_realization = Mock()

        @data_transformer(
            inputs=(TANGENT_ANGLE_SERIES,),
            outputs=(TANGENT_ANGLE_PCA_SERIES,),
            algorithm="mock_algo"
        )
        def mock_transform(data):
            return {"tangent_angle_pca_series": {"pca": "data"}}

        existing_output = DataItem(
            data_name=TANGENT_ANGLE_PCA_SERIES,
            data={"filenames": []},
            dependencies=[item.id],
            algorithm=mock_transform.algorithm,
        )
        realization.add_data_item(existing_output)

        run_transformer_on_dataset(dataset, mock_transform)
        assert not dataset.add_data_to_realization.called

    def test_run_transformer_on_dataset(self):
        """Test running transformer on dataset."""
        # Create mock dataset
        realization = Realization("exp")
        item = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={"filenames": []},
            dependencies=[],
            algorithm="original"
        )
        realization.add_data_item(item)
        
        dataset = Mock()
        dataset.path = ''
        dataset.__iter__ = Mock(side_effect=lambda: iter([realization]))
        # Provide a default_loader that returns a numpy array so DataItem.resolve
        # receives the expected type (tests use Mock dataset, which otherwise
        # returns Mock objects and fails schema validation).
        dataset.default_loader = Mock(side_effect=lambda exp_path, meta: np.array([[0.0]]))
        
        @data_transformer(
            inputs=(TANGENT_ANGLE_SERIES,),
            outputs=(TANGENT_ANGLE_PCA_SERIES,)
        )
        def mock_transform(data):
            return {"tangent_angle_pca_series": {"pca": "data"}}
        
        # Mock add_data_to_realization
        dataset.add_data_to_realization = Mock()
        
        run_transformer_on_dataset(dataset, mock_transform)
        
        # Check that add_data_to_realization was called
        assert dataset.add_data_to_realization.called
        args = dataset.add_data_to_realization.call_args
        new_item = args[0][1]  # second argument
        assert new_item.data_name == TANGENT_ANGLE_PCA_SERIES
        assert new_item.algorithm == mock_transform.algorithm


    def test_run_transformer_on_realization(self):
        """Test running transformer on a single realization."""
        realization = Realization("exp")
        item = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={"filenames": []},
            dependencies=[],
            algorithm="original"
        )
        realization.add_data_item(item)

        dataset = Mock()
        dataset.path = ''
        dataset.default_loader = Mock(side_effect=lambda exp_path, meta: np.array([[0.0]]))
        dataset.add_data_to_realization = Mock()

        @data_transformer(
            inputs=(TANGENT_ANGLE_SERIES,),
            outputs=(TANGENT_ANGLE_PCA_SERIES,)
        )
        def mock_transform(data):
            return {"tangent_angle_pca_series": {"pca": "data"}}

        run_transformer_on_realization(dataset, realization, mock_transform)

        assert dataset.add_data_to_realization.called
        args = dataset.add_data_to_realization.call_args
        assert args[0][0] == realization  # first arg is realization
        new_item = args[0][1]
        assert new_item.data_name == TANGENT_ANGLE_PCA_SERIES

    def test_run_transformer_on_realization_skips_existing(self):
        """Test that run_transformer_on_realization skips if output exists."""
        realization = Realization("exp")
        item = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={"filenames": []},
            dependencies=[],
            algorithm="original"
        )
        realization.add_data_item(item)

        dataset = Mock()
        dataset.path = ''
        dataset.default_loader = Mock(side_effect=lambda exp_path, meta: np.array([[0.0]]))
        dataset.add_data_to_realization = Mock()

        @data_transformer(
            inputs=(TANGENT_ANGLE_SERIES,),
            outputs=(TANGENT_ANGLE_PCA_SERIES,),
            algorithm="mock_algo"
        )
        def mock_transform(data):
            return {"tangent_angle_pca_series": {"pca": "data"}}

        existing_output = DataItem(
            data_name=TANGENT_ANGLE_PCA_SERIES,
            data={"filenames": []},
            dependencies=[item.id],
            algorithm=mock_transform.algorithm,
        )
        realization.add_data_item(existing_output)

        run_transformer_on_realization(dataset, realization, mock_transform)

        assert not dataset.add_data_to_realization.called

    def test_run_viewer_on_realization(self):
        """Test running viewer on a single realization."""
        realization = Realization("exp")
        item = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={"filenames": []},
            dependencies=[],
            algorithm="original"
        )
        realization.add_data_item(item)

        dataset = Mock()
        dataset.path = ''
        dataset.default_loader = Mock(side_effect=lambda exp_path, meta: np.array([[0.0]]))

        viewer_func = Mock()

        run_viewer_on_realization(
            dataset, realization, (TANGENT_ANGLE_SERIES,),
            viewer_func=viewer_func
        )

        assert viewer_func.called

    # def test_share_root(self):
    #     """Test _share_root function."""
    #     from cilia.transformers.Transformer import _share_root

    #     realization = Realization("exp")

    #     # Single item: should return True
    #     item1 = DataItem(
    #         data_name=TANGENT_ANGLE_SERIES,
    #         data={},
    #         dependencies=[],
    #         algorithm="algo1"
    #     )
    #     assert _share_root(realization, [item1])

    #     # Two items with common dependency
    #     item2 = DataItem(
    #         data_name=TANGENT_ANGLE_PCA_SERIES,
    #         data={},
    #         dependencies=[item1.id],
    #         algorithm="algo2"
    #     )
    #     assert _share_root(realization, [item1, item2])

    #     # Two items with no common root
    #     item3 = DataItem(
    #         data_name=TANGENT_ANGLE_PCA_SERIES,
    #         data={},
    #         dependencies=["other"],
    #         algorithm="algo3"
    #     )
    #     assert not _share_root(realization, [item1, item3])

    def test_run_transformer_on_realization_rejects_non_dict(self):
        """Test that run_transformer_on_realization rejects non-dict output."""
        realization = Realization("exp")
        item = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={"filenames": []},
            dependencies=[],
            algorithm="original"
        )
        realization.add_data_item(item)

        dataset = Mock()
        dataset.path = ''
        dataset.default_loader = Mock(side_effect=lambda exp_path, meta: np.array([[0.0]]))
        dataset.add_data_to_realization = Mock()

        @data_transformer(
            inputs=(TANGENT_ANGLE_SERIES,),
            outputs=(TANGENT_ANGLE_PCA_SERIES,)
        )
        def bad_transform(data):
            return "not dict"

        with pytest.raises(ValueError, match="Transformer output must be a dict"):
            run_transformer_on_realization(dataset, realization, bad_transform)
        assert not dataset.add_data_to_realization.called

    def test_pipeline_run_returns_output_selector(self):
        realization = Realization("exp")
        item = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={"filenames": []},
            dependencies=[ORIGINAL.key],
            algorithm="original",
        )
        realization.add_data_item(item)

        dataset = Mock()
        dataset.path = ""
        dataset.__iter__ = Mock(side_effect=lambda: iter([realization]))
        dataset.default_loader = Mock(side_effect=lambda exp_path, meta: np.array([[0.0]]))
        dataset.add_data_to_realization = Mock(side_effect=lambda real, new_item: real.add_data_item(new_item))

        @data_transformer(
            inputs=(TANGENT_ANGLE_SERIES,),
            outputs=(TANGENT_ANGLE_PCA_SERIES,),
            algorithm="pipeline_algo",
        )
        def mock_transform(data):
            return {"tangent_angle_pca_series": 1.0}

        result = Pipeline(dataset, show_progress=False).run(
            mock_transform,
            inputs={TANGENT_ANGLE_SERIES: select(TANGENT_ANGLE_SERIES).source(ORIGINAL)},
        )

        output_selector = result.only()
        output_candidates = output_selector.candidates(realization)
        assert len(output_candidates) == 1
        assert output_candidates[0].algorithm == "pipeline_algo"

    def test_pipeline_matches_inputs_by_shared_ancestor_key(self):
        try:
            PHASE_TEST = DataName(key="phase_test_for_pipeline", schema=ScalarSchema(float))
        except ValueError:
            PHASE_TEST = DataName.get("phase_test_for_pipeline")
        try:
            AVG_TEST = DataName(key="avg_test_for_pipeline", schema=ScalarSchema(float))
        except ValueError:
            AVG_TEST = DataName.get("avg_test_for_pipeline")

        realization = Realization("exp")
        tangent_1 = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={"filenames": []},
            dependencies=[ORIGINAL.key],
            algorithm="original_1",
        )
        tangent_2 = DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data={"filenames": []},
            dependencies=[ORIGINAL.key],
            algorithm="original_2",
        )
        phase_1 = DataItem(
            data_name=PHASE_TEST,
            data=1.0,
            dependencies=[tangent_1.id],
            algorithm="phase",
        )
        phase_2 = DataItem(
            data_name=PHASE_TEST,
            data=2.0,
            dependencies=[tangent_2.id],
            algorithm="phase",
        )
        for item in (tangent_1, tangent_2, phase_1, phase_2):
            realization.add_data_item(item)

        dataset = Mock()
        dataset.path = ""
        dataset.default_loader = Mock(side_effect=lambda exp_path, meta: np.array([[0.0]]))
        dataset.add_data_to_realization = Mock(side_effect=lambda real, new_item: real.add_data_item(new_item))

        calls = []

        @data_transformer(
            inputs=(TANGENT_ANGLE_SERIES, PHASE_TEST),
            outputs=(AVG_TEST,),
            algorithm="avg",
        )
        def combine(tangent, phase):
            calls.append(float(phase))
            return {"avg_test_for_pipeline": float(phase)}

        Pipeline(dataset, show_progress=False).run_on_realization(
            realization,
            combine,
            match_ancestor_keys={TANGENT_ANGLE_SERIES.key},
        )

        assert sorted(calls) == [1.0, 2.0]
        assert dataset.add_data_to_realization.call_count == 2


if __name__ == "__main__":
    # allow running this file directly to execute its tests
    import pytest, sys
    sys.exit(pytest.main([__file__]))
