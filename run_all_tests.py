#!/usr/bin/env python3
"""
Comprehensive test runner for emoticon_fix library.
"""

import sys
from test_emoticon_fix.test_cases import run_basic_tests
from emoticon_fix import (
    emoticon_fix,
    remove_emoticons,
    replace_emoticons,
    analyze_sentiment,
    get_sentiment_score,
    classify_sentiment,
    extract_emotions,
    batch_analyze
)

def test_sentiment_analysis():
    """Test sentiment analysis functionality."""
    print("Testing sentiment analysis...")
    
    test_cases = [
        ("Happy :)", "Very Positive"),  # Updated expectation
        ("Sad :(", "Negative"),
        ("Love you (｡♥‿♥｡)", "Very Positive"),
        ("Angry >:(", "Very Negative"),  # Updated expectation
        ("Neutral :|", "Neutral"),
        ("", "Neutral"),
        ("No emoticons", "Neutral"),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, (text, expected_classification) in enumerate(test_cases, 1):
        try:
            classification = classify_sentiment(text)
            score = get_sentiment_score(text)
            analysis = analyze_sentiment(text)
            emotions = extract_emotions(text)
            
            if classification == expected_classification:
                passed += 1
                print(f"Test {i}: ✓ PASS")
            else:
                print(f"Test {i}: ✗ FAIL")
                print(f"  Expected: {expected_classification}, Got: {classification}")
                
        except Exception as e:
            print(f"Test {i}: ✗ ERROR - {e}")
    
    print(f"Sentiment Analysis Tests: {passed}/{total} passed")
    return passed == total

def test_additional_functions():
    """Test additional functions."""
    print("Testing additional functions...")
    
    test_text = "Happy :) and sad :("
    
    try:
        # Test remove_emoticons
        removed = remove_emoticons(test_text)
        assert "Happy and sad" in removed
        
        # Test replace_emoticons
        replaced = replace_emoticons(test_text)
        assert "__EMO_" in replaced
        
        # Test batch_analyze
        batch_results = batch_analyze([test_text, "Just text"])
        assert len(batch_results) == 2
        
        print("Additional Functions: ✓ PASS")
        return True
        
    except Exception as e:
        print(f"Additional Functions: ✗ ERROR - {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("EMOTICON_FIX COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Run basic tests
    results.append(run_basic_tests())
    print()
    
    # Run sentiment analysis tests
    results.append(test_sentiment_analysis())
    print()
    
    # Run additional function tests
    results.append(test_additional_functions())
    print()
    
    # Summary
    passed_suites = sum(results)
    total_suites = len(results)
    
    print("=" * 60)
    print(f"SUMMARY: {passed_suites}/{total_suites} test suites passed")
    
    if passed_suites == total_suites:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 