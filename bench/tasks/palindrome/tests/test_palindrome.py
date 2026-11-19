from palindrome import is_palindrome


def test_simple_palindromes() -> None:
    assert is_palindrome("racecar") is True
    assert is_palindrome("madam") is True
    assert is_palindrome("hello") is False


def test_case_and_punctuation() -> None:
    assert is_palindrome("A man, a plan, a canal: Panama") is True
    assert is_palindrome("No 'x' in Nixon") is True
    assert is_palindrome("Was it a car or a cat I saw?") is True


def test_digits_and_empty() -> None:
    assert is_palindrome("") is True
    assert is_palindrome("   ") is True
    assert is_palindrome("12321") is True
    assert is_palindrome("12345") is False
