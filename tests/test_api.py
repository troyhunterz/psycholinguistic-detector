import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'


def test_predict_fear_appeal():
    response = client.post('/predict', json={
        'text': 'Enemies want to destroy everything we love! Our children are in danger!'
    })

    assert response.status_code == 200
    data = response.json()

    assert 'label' in data
    assert 'confidence' in data
    assert 'all_scores' in data

    assert 0 <= data['confidence'] <= 1

    expected_classes = {
        'fear_appeal', 'emotional_manipulation',
        'demagogy_tricks', 'authority_appeal', 'rational_argument'
    }

    assert set(data['all_scores'].keys()) == expected_classes

    # assert data['label'] == 'fear_appeal'


def test_predict_rational():
    response = client.post('/predict', json={
        'text': 'According to Federal Reserve data, inflation decreased by 2.3 percent.'
    })
    assert response.status_code == 200
    data = response.json()
    assert data['label'] in ['rational_argument', 'authority_appeal']


def test_predict_empty_text():
    response = client.post('/predict', json={'text': ''})
    assert response.status_code == 200


def test_predict_missing_field():
    response = client.post('/predict', json={'wrong_field': 'text'})
    assert response.status_code == 422


def test_predict_long_text():
    long_text = 'This is a test.' * 100
    response = client.post('/predict', json={'text': long_text})
    assert response.status_code == 200
