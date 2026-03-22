from paper_reading.utils import estimate_token_num


def test_estimate_token_num_allows_literal_special_token_text() -> None:
    text = "prefix <|endoftext|> suffix"

    token_num, token_strings = estimate_token_num(text)

    assert token_num > 0
    assert len(token_strings) == token_num


def test_estimate_token_num_returns_zero_for_blank_text() -> None:
    token_num, token_strings = estimate_token_num("   ")

    assert token_num == 0
    assert token_strings == []
