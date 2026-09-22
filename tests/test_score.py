import numpy as np
import pytest

from gradpath.match.score import TOP_K, cosine_matrix, score_person


def test_score_is_mean_of_top_three():
    result = score_person({1: 0.9, 2: 0.8, 3: 0.7, 4: 0.1, 5: 0.0})
    assert result.score == pytest.approx(0.8)
    assert result.sparse is False


def test_prolific_author_is_not_diluted_by_irrelevant_work():
    """The whole point of mean-of-top-3: 200 unrelated papers must not bury 3 great ones."""
    focused = score_person({1: 0.9, 2: 0.85, 3: 0.8})
    prolific = score_person({1: 0.9, 2: 0.85, 3: 0.8, **{i: 0.05 for i in range(4, 204)}})
    assert prolific.score == pytest.approx(focused.score)


def test_single_lucky_match_does_not_dominate():
    """Mean-of-top-3, not max: one coincidental abstract should not promote someone."""
    lucky = score_person({1: 0.95, 2: 0.10, 3: 0.05})
    steady = score_person({1: 0.70, 2: 0.68, 3: 0.66})
    assert steady.score > lucky.score


def test_top_work_ids_are_the_contributing_works():
    result = score_person({7: 0.2, 8: 0.9, 9: 0.5, 10: 0.7})
    assert result.top_work_ids == [8, 10, 9]


def test_fewer_than_k_works_is_marked_sparse_without_padding():
    result = score_person({1: 0.8, 2: 0.6})
    assert result.score == pytest.approx(0.7)
    assert result.sparse is True


def test_single_work_is_sparse():
    result = score_person({1: 0.5})
    assert result.score == pytest.approx(0.5)
    assert result.sparse is True


def test_no_works_raises():
    with pytest.raises(ValueError, match="no works"):
        score_person({})


def test_cosine_matrix_matches_manual_computation():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    vector = np.array([1.0, 0.0], dtype=np.float32)
    got = cosine_matrix(matrix, vector)
    assert got == pytest.approx([1.0, 0.0, 0.7071], abs=1e-4)


def test_cosine_matrix_tolerates_zero_vectors():
    matrix = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    vector = np.array([1.0, 0.0], dtype=np.float32)
    assert cosine_matrix(matrix, vector)[0] == pytest.approx(0.0)


def test_top_k_is_three():
    assert TOP_K == 3
