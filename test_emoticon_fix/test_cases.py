from emoticon_fix import emoticon_fix

# Test cases
test_cases = [
    # Basic cases
    ("Hello :) World", "Hello Smile World"),
    ("Hello:) World", "Hello Smile World"),
    ("Hello :)! World", "Hello Smile! World"),
    
    # Multiple emoticons
    ("Hello :) :D World", "Hello Smile Laugh World"),
    
    # Punctuation cases
    ("Hello, :) World!", "Hello, Smile World!"),
    ("Hello:)! Nice:D.", "Hello Smile! Nice Laugh."),
    ("Hello:)!:D", "Hello Smile! Laugh"),
    
    # Edge cases
    (":) Hello", "Smile Hello"),
    ("Hello :)", "Hello Smile"),
    ("Hello:):)", "Hello Smile Smile"),
    
    # Mixed cases
    ("Hello:)World:D!", "Hello Smile World Laugh!"),
    ("Hi!:)Bye", "Hi! Smile Bye"),
    ("Test,:),test", "Test, Smile, test")
]

# Run tests
for i, (input_text, expected) in enumerate(test_cases, 1):
    result = emoticon_fix(input_text)
    success = result == expected
    print(f"\nTest {i}:")
    print(f"Input:    '{input_text}'")
    print(f"Expected: '{expected}'")
    print(f"Got:      '{result}'")
    print(f"Status:   {'✓ PASS' if success else '✗ FAIL'}")
    
    if not success:
        print("Difference analysis:")
        print(f"Expected length: {len(expected)}, Got length: {len(result)}")
        for j, (e, g) in enumerate(zip(expected, result)):
            if e != g:
                print(f"Mismatch at position {j}: Expected '{e}', Got '{g}'") 