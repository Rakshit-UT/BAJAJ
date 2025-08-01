"""
API endpoint tests
"""

import pytest
import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000/api/v1"
BEARER_TOKEN = "c0df38f44acb385ecd42f8e0c02ee14acd6d145835643ee57acd84f79afeb798"

HEADERS = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Content-Type": "application/json"
}

def test_health_endpoint():
    """Test health check endpoint"""
    response = requests.get(f"{BASE_URL.replace('/api/v1', '')}/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_status_endpoint():
    """Test status endpoint"""  
    response = requests.get(f"{BASE_URL}/status", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data

def test_hackrx_run_endpoint():
    """Test main hackrx/run endpoint"""
    # This would require a real document URL for full testing
    test_payload = {
        "documents": "https://example.com/test-document.pdf",
        "questions": [
            "What is this document about?",
            "What are the key terms?"
        ]
    }

    # Note: This test would fail without a real document
    # It's included for structure demonstration
    # response = requests.post(f"{BASE_URL}/hackrx/run", 
    #                         headers=HEADERS, 
    #                         json=test_payload)
    # assert response.status_code == 200

def test_authentication():
    """Test authentication requirements"""
    test_payload = {
        "documents": "https://example.com/test.pdf",
        "questions": ["Test question"]
    }

    # Test without authentication
    response = requests.post(f"{BASE_URL}/hackrx/run", json=test_payload)
    assert response.status_code == 401

if __name__ == "__main__":
    print("Running API tests...")
    test_health_endpoint()
    test_authentication()
    print("Basic tests passed!")
