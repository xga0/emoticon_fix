#!/usr/bin/env python3
"""
Basic test cases for the emoticon_fix core functionality.
"""

from emoticon_fix import emoticon_fix

def run_basic_tests():
    """Run basic test cases for emoticon_fix functionality."""
    test_cases = [
        ("Hello :) World", "Hello Smile World"),
        ("Hello:) World", "Hello Smile World"),
        ("Hello :)! World", "Hello Smile ! World"),  # Updated expectation
        ("Hello :) :D World", "Hello Smile Laugh World"),
        ("Hello, :) World!", "Hello , Smile World !"),  # Updated expectation
        ("Hello:)! Nice:D.", "Hello Smile ! Nice Laugh ."),  # Updated expectation
        ("Hello:)!:D", "Hello Smile ! Laugh"),  # Updated expectation
        (":) Hello", "Smile Hello"),
        ("Hello :)", "Hello Smile"),
        ("Hello:):)", "Hello Smile Smile"),
        ("Hello:)World:D!", "Hello Smile World Laugh !"),  # Updated expectation
        ("Hi!:)Bye", "Hi ! Smile Bye"),  # Updated expectation
        ("Test,:),test", "Test , Smile , test")  # Updated expectation
    ]
    
    print("Running basic emoticon_fix tests...")
    passed = 0
    total = len(test_cases)
    
    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = emoticon_fix(input_text)
        success = result == expected
        
        if success:
            passed += 1
            print(f"Test {i}: ✓ PASS")
        else:
            print(f"Test {i}: ✗ FAIL")
            print(f"  Input:    '{input_text}'")
            print(f"  Expected: '{expected}'")
            print(f"  Got:      '{result}'")
    
    print(f"\nBasic Tests: {passed}/{total} passed")
    return passed == total

if __name__ == "__main__":
    run_basic_tests() 