from gradpath.sources.crawler import extract_emails


def test_extracts_plain_mailto():
    html = '<a href="mailto:kim@kaist.ac.kr">Prof Kim</a>'
    assert extract_emails(html) == ["kim@kaist.ac.kr"]


def test_extracts_bare_text_address():
    assert extract_emails("<p>contact: park@gist.ac.kr</p>") == ["park@gist.ac.kr"]


def test_decodes_common_at_obfuscation():
    html = "<p>lee [at] snu.ac.kr</p>"
    assert extract_emails(html) == ["lee@snu.ac.kr"]


def test_decodes_dot_obfuscation():
    html = "<p>choi [at] kaist [dot] ac [dot] kr</p>"
    assert extract_emails(html) == ["choi@kaist.ac.kr"]


def test_deduplicates_and_preserves_order():
    html = "a@x.ac.kr b@x.ac.kr a@x.ac.kr"
    assert extract_emails(html) == ["a@x.ac.kr", "b@x.ac.kr"]


def test_ignores_image_and_asset_filenames():
    assert extract_emails('<img src="logo@2x.png">') == []


def test_returns_empty_for_no_addresses():
    assert extract_emails("<p>no contact here</p>") == []
