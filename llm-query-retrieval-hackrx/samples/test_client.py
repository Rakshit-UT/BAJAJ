#!/usr/bin/env python3
"""
Sample client for testing the LLM Query-Retrieval System
"""

import requests
import json
import sys

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
BEARER_TOKEN = "c0df38f44acb385ecd42f8e0c02ee14acd6d145835643ee57acd84f79afeb798"

HEADERS = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Content-Type": "application/json"
}

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL.replace('/api/v1', '')}/health")
        print(f"Health Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_status():
    """Test status endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/status", headers=HEADERS)
        print(f"\nAPI Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Status check failed: {e}")
        return False

def test_query_processing(document_url, questions):
    """Test document processing and query answering"""
    payload = {
        "documents": document_url,
        "questions": questions
    }

    try:
        print(f"\nTesting query processing...")
        print(f"Document: {document_url}")
        print(f"Questions: {questions}")

        response = requests.post(f"{BASE_URL}/hackrx/run", 
                               headers=HEADERS, 
                               json=payload,
                               timeout=300)

        print(f"\nResponse Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\nAnswers:")
            for i, answer in enumerate(result.get("answers", []), 1):
                print(f"{i}. {answer}")
        else:
            print(f"Error: {response.text}")

        return response.status_code == 200

    except requests.exceptions.Timeout:
        print("Request timed out")
        return False
    except Exception as e:
        print(f"Query processing failed: {e}")
        return False

def test_advanced_processing(document_url, questions):
    """Test advanced processing with citations"""
    payload = {
        "documents": document_url,
        "questions": questions
    }

    try:
        print(f"\nTesting advanced processing...")

        response = requests.post(f"{BASE_URL}/hackrx/run-advanced", 
                               headers=HEADERS, 
                               json=payload,
                               timeout=300)

        print(f"Response Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            print("\nAnswers with Citations:")
            for i, (answer, citations) in enumerate(zip(
                result.get("answers", []), 
                result.get("citations", [])
            ), 1):
                print(f"{i}. {answer}")
                if citations:
                    print(f"   Citations: {citations}")
        else:
            print(f"Error: {response.text}")

        return response.status_code == 200

    except Exception as e:
        print(f"Advanced processing failed: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("LLM Query-Retrieval System - Test Client")
    print("="*60)

    # Test basic endpoints
    if not test_health():
        print("Health check failed - is the server running?")
        sys.exit(1)

    if not test_status():
        print("Status check failed - authentication issue?")
        sys.exit(1)

    # Sample document and questions for testing
    # Replace with actual document URLs for real testing
    sample_document = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

    sample_questions = [
        "What is the grace period for premium payment?",
        "What are the exclusions in this policy?",
        "What is the waiting period for pre-existing diseases?"
    ]

    # Test document processing
    print("\n" + "="*60)
    print("Testing Document Processing")
    print("="*60)

    success = test_query_processing(sample_document, sample_questions)

    if success:
        print("\n" + "="*60)
        print("Testing Advanced Processing")
        print("="*60)
        test_advanced_processing(sample_document, sample_questions[:2])  # Test with fewer questions

    print("\n" + "="*60)
    print("Testing Complete")
    print("="*60)

if __name__ == "__main__":
    main()
