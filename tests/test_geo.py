import numpy as np
import pytest

from burncal.geo import aoi_cells, cell_index, points_in_polygon, validate_aoi


def test_cell_index_negative_and_boundaries():
    assert cell_index(-0.05, 0.1) == -1
    assert cell_index(0.0, 0.1) == 0
    assert cell_index(4.2, 0.1) == 42      # exact multiples must not drift down
    assert cell_index(-4.2, 0.1) == -42


def test_point_in_polygon_triangle():
    tri = np.array([[0, 0], [10, 0], [0, 10]], float)
    inside = points_in_polygon([1, 6, -1], [1, 6, 1], tri)
    assert inside.tolist() == [True, False, False]


def test_bbox_cells_count_and_polygon_subset():
    cells, small = aoi_cells({"bbox": [-72.0, 3.0, -71.0, 4.0]}, 0.1)
    assert not small and len(cells) == 100
    tri, _ = aoi_cells({"polygon": [[-72, 3], [-71, 3], [-72, 4]]}, 0.1)
    assert 40 <= len(tri) <= 60            # about half the square


def test_tiny_aoi_falls_back_to_single_cell():
    cells, small = aoi_cells({"bbox": [-72.0, 4.0, -71.99, 4.01]}, 0.1)
    assert small and len(cells) == 1


@pytest.mark.parametrize("aoi", [
    {}, {"bbox": [1, 2, 3, 4], "polygon": [[0, 0], [1, 1], [1, 0]]}, {"bbox": [5, 0, 1, 1]},
    {"bbox": [0, 0, 1]}, {"polygon": [[0, 0], [1, 1]]}, {"bbox": [-200, 0, 1, 1]},
])
def test_invalid_aoi_rejected(aoi):
    with pytest.raises(ValueError):
        validate_aoi(aoi)
