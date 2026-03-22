from src.features.extractor import extract_features
import sys
sys.path.append('.')


def test_extract_features_returns_dict():
    result = extract_features('This is a test sentence.')
    assert isinstance(result, dict)


def test_extract_features_has_all_keys():
    result = extract_features('This is a test sentence.')
    expected_keys = {
        'we_ratio', 'they_ratio', 'exclaim_ratio',
        'question_ratio', 'modal_ratio', 'logic_count',
        'adj_ratio', 'verb_ratio', 'avg_sent_len', 'caps_ratio'
    }
    assert set(result.keys()) == expected_keys


def test_exclaim_ratio_detected():
    result = extract_features('Danger! Enemies are coming! Run!')
    assert result['exclaim_ratio'] > 0


def test_we_ratio_detected():
    result = extract_features('We must protect our children and our future.')
    assert result['we_ratio'] > 0


def test_logic_count_detected():
    result = extract_features(
        'We must act because the evidence shows this is true. Therefore we should proceed.')
    assert result['logic_count'] > 0


def test_all_values_are_numeric():
    result = extract_features('Some random text here.')
    for key, value in result.items():
        assert isinstance(value, (int, float)), f'{key} is not numeric'


def test_empty_text_doesnt_crash():
    result = extract_features('')
    assert isinstance(result, dict)
