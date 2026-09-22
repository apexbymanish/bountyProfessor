import pytest

from gradpath.match.faculty import PublicationHistory, score_faculty

CLEAR_PI = PublicationHistory(
    first_year=2005, last_year=2026, works_count=120,
    last_author_count=80, first_author_count=5,
)
PHD_STUDENT = PublicationHistory(
    first_year=2022, last_year=2026, works_count=6,
    last_author_count=0, first_author_count=4,
)
ALPHABETICAL_MATHEMATICIAN = PublicationHistory(
    first_year=2001, last_year=2026, works_count=60,
    last_author_count=15, first_author_count=20, alphabetical_field=True,
)
AMBIGUOUS_MID_CAREER = PublicationHistory(
    first_year=2016, last_year=2026, works_count=25,
    last_author_count=8, first_author_count=6,
)


def test_clear_pi_scores_high():
    assert score_faculty(CLEAR_PI).confidence == "high"


def test_phd_student_scores_low():
    assert score_faculty(PHD_STUDENT).confidence == "low"


def test_alphabetical_field_is_not_penalised_for_author_position():
    """Maths and economics list authors alphabetically; last-author means nothing there."""
    assessment = score_faculty(ALPHABETICAL_MATHEMATICIAN)
    assert assessment.confidence == "high"
    assert assessment.signals["alphabetical_field"] is True
    assert "last_author_ratio" not in assessment.signals


def test_ambiguous_mid_career_scores_medium():
    assert score_faculty(AMBIGUOUS_MID_CAREER).confidence == "medium"


def test_directory_confirmation_overrides_a_weak_heuristic():
    """An authoritative signal must beat the heuristic, never be averaged with it."""
    confirmed = PublicationHistory(
        first_year=2024, last_year=2026, works_count=2,
        last_author_count=0, first_author_count=2, directory_confirmed=True,
    )
    assessment = score_faculty(confirmed)
    assert assessment.confidence == "high"
    assert assessment.signals["directory_confirmed"] is True


def test_orcid_professor_title_overrides_a_weak_heuristic():
    titled = PublicationHistory(
        first_year=2024, last_year=2026, works_count=3,
        last_author_count=0, first_author_count=3,
        orcid_title="Associate Professor of Computer Science",
    )
    assert score_faculty(titled).confidence == "high"


def test_orcid_student_title_does_not_confer_faculty_status():
    titled = PublicationHistory(
        first_year=2024, last_year=2026, works_count=3,
        last_author_count=0, first_author_count=3,
        orcid_title="PhD Candidate",
    )
    assert score_faculty(titled).confidence == "low"


def test_signals_explain_the_score():
    signals = score_faculty(CLEAR_PI).signals
    assert signals["career_span"] == 21
    assert signals["works_count"] == 120
    assert signals["last_author_ratio"] == pytest.approx(0.667, abs=1e-3)


def test_zero_works_does_not_divide_by_zero():
    empty = PublicationHistory(
        first_year=2026, last_year=2026, works_count=0,
        last_author_count=0, first_author_count=0,
    )
    assert score_faculty(empty).score == pytest.approx(0.0, abs=1e-6)
