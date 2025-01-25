import re


def clean_custom_patterns(text):

    # Replace: email, phone, youtube link, regular link  with [email], [phone], [youtube], [link]
    clean_text = re.sub(  # email
        r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", "[email]", text
    )
    clean_text = re.sub(  # phone
        r"(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})",
        "[phone]",
        clean_text,
    )
    clean_text = re.sub(  # youtube link
        r"(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+", "[youtube]", clean_text
    )
    clean_text = re.sub(  # regular link
        r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)",
        "[link]",
        clean_text,
    )

    return clean_text


def test_clean_custom_patterns():
    test_cases = [
        # Email tests
        {
            "input": "Contact me at john.doe@example.com",
            "expected": "Contact me at [email]",
        },
        {
            "input": "Multiple emails: a@b.com and x.y@z.co.uk",
            "expected": "Multiple emails: [email] and [email]",
        },
        {
            "input": "Complex email: user+tag@subdomain.example.com",
            "expected": "Complex email: [email]",
        },
        # Phone tests
        {"input": "Call me at 123-456-7890", "expected": "Call me at [phone]"},
        {
            "input": "Phone formats: (123) 456-7890, 123.456.7890, 123 456 7890",
            "expected": "Phone formats: [phone], [phone], [phone]",
        },
        {"input": "Short number: 123-4567", "expected": "Short number: [phone]"},
        # YouTube link tests
        {
            "input": "Watch: https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "expected": "Watch: [youtube]",
        },
        {
            "input": "Short YouTube: https://youtu.be/dQw4w9WgXcQ",
            "expected": "Short YouTube: [youtube]",
        },
        {
            "input": "YouTube without protocol: www.youtube.com/watch?v=dQw4w9WgXcQ",
            "expected": "YouTube without protocol: [youtube]",
        },
        # Regular link tests
        {"input": "Visit http://example.com", "expected": "Visit [link]"},
        {
            "input": "Secure link: https://secure.example.com/path?query=value",
            "expected": "Secure link: [link]",
        },
        {
            "input": "Link with port: http://localhost:8080/test",
            "expected": "Link with port: [link]",
        },
        # Mixed pattern tests
        {
            "input": "Contact: john@example.com, 123-456-7890, https://youtu.be/abc123, http://example.com",
            "expected": "Contact: [email], [phone], [youtube], [link]",
        },
        {"input": "No patterns here!", "expected": "No patterns here!"},
        # Edge cases
        {
            "input": "Email in parentheses (john.doe@example.com)",
            "expected": "Email in parentheses ([email])",
        },
        {
            "input": "Phone with extension: 123-456-7890 ext. 123",
            "expected": "Phone with extension: [phone] ext. 123",
        },
        {
            "input": "YouTube channel: https://www.youtube.com/channel/UC-lHJZR3Gqxm24_Vd_AJ5Yw",
            "expected": "YouTube channel: [youtube]",
        },
        {
            "input": "Link with fragment: https://example.com/page#section",
            "expected": "Link with fragment: [link]",
        },
        # Repeated patterns
        {
            "input": "Repeat: john@example.com john@example.com",
            "expected": "Repeat: [email] [email]",
        },
        {
            "input": "Repeat: 123-456-7890 123-456-7890",
            "expected": "Repeat: [phone] [phone]",
        },
        {
            "input": "Repeat: https://youtube.com/watch?v=123 https://youtube.com/watch?v=456",
            "expected": "Repeat: [youtube] [youtube]",
        },
        {
            "input": "Repeat: http://example.com http://example.org",
            "expected": "Repeat: [link] [link]",
        },
    ]

    results = []
    for i, test_case in enumerate(test_cases):
        output = clean_custom_patterns(test_case["input"])
        results.append(
            {
                "test_case": i + 1,
                "input": test_case["input"],
                "output": output,
                "expected": test_case["expected"],
                "passed": output == test_case["expected"],
            }
        )

    return results


# Run the test cases
test_results = test_clean_custom_patterns()

# Print results
for result in test_results:
    print(f"Test Case {result['test_case']}:")
    print(f"Input: {result['input']}")
    print(f"Output: {result['output']}")
    print(f"Expected: {result['expected']}")
    print(f"Passed: {result['passed']}")
    print()

# Summary
total_tests = len(test_results)
passed_tests = sum(1 for result in test_results if result["passed"])
print(f"Total tests: {total_tests}")
print(f"Passed tests: {passed_tests}")
print(f"Failed tests: {total_tests - passed_tests}")
