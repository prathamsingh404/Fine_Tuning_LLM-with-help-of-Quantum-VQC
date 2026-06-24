import pytest
from train import tokenize

def test_tokenize_basic():
    vocab = {"<PAD>": 0, "<UNK>": 1, "hello": 2, "world": 3}
    sentence = "hello world"
    result = tokenize(sentence, vocab, max_len=5)
    assert result == [2, 3, 0, 0, 0]

def test_tokenize_unknown_word():
    vocab = {"<PAD>": 0, "<UNK>": 1, "hello": 2}
    sentence = "hello unknown_word"
    result = tokenize(sentence, vocab, max_len=5)
    assert result == [2, 1, 0, 0, 0]

def test_tokenize_truncation():
    vocab = {"<PAD>": 0, "<UNK>": 1, "a": 2, "b": 3, "c": 4}
    sentence = "a b c d e f"
    result = tokenize(sentence, vocab, max_len=3)
    assert result == [2, 3, 4]

def test_tokenize_padding():
    vocab = {"<PAD>": 0, "<UNK>": 1, "only": 2}
    sentence = "only"
    result = tokenize(sentence, vocab, max_len=4)
    assert result == [2, 0, 0, 0]

def test_tokenize_empty_sentence():
    vocab = {"<PAD>": 0, "<UNK>": 1}
    sentence = ""
    result = tokenize(sentence, vocab, max_len=4)
    assert result == [0, 0, 0, 0]

def test_tokenize_uppercase():
    vocab = {"<PAD>": 0, "<UNK>": 1, "hello": 2}
    sentence = "HELLO"
    result = tokenize(sentence, vocab, max_len=3)
    assert result == [2, 0, 0]

def test_tokenize_non_string():
    vocab = {"<PAD>": 0, "<UNK>": 1, "123": 2}
    sentence = 123
    result = tokenize(sentence, vocab, max_len=2)
    assert result == [2, 0]

def test_tokenize_exact_length():
    vocab = {"<PAD>": 0, "<UNK>": 1, "one": 2, "two": 3}
    sentence = "one two"
    result = tokenize(sentence, vocab, max_len=2)
    assert result == [2, 3]

def test_tokenize_handles_none():
    vocab = {"<PAD>": 0, "<UNK>": 1, "none": 2}
    sentence = None
    result = tokenize(sentence, vocab, max_len=3)
    # str(None) is "none"
    assert result == [2, 0, 0]
