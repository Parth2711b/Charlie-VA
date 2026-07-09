import asyncio
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from handlers.web import _extract_url, IFRAME_BLOCKERS

logging.basicConfig(level=logging.INFO)

def run_tests():
    print("--- Testing Web Browser Intent ---")
    
    test_cases = [
        ("open github", "https://www.github.com", True),
        ("go to wikipedia", "https://www.wikipedia.org", True),
        ("visit example.com", "https://example.com", True),
        ("open netflix", "https://www.netflix.com", False),
        ("open youtube", "https://www.youtube.com", False),
        ("go to google", "https://www.google.com", False),
    ]
    
    passed = 0
    for text, expected_url, expected_iframe in test_cases:
        url, use_iframe = _extract_url(text)
        if url == expected_url and use_iframe == expected_iframe:
            print(f"[PASS] '{text}' -> {url} (Iframe: {use_iframe})")
            passed += 1
        else:
            print(f"[FAIL] '{text}' -> Expected {expected_url} ({expected_iframe}), got {url} ({use_iframe})")
            
    print(f"\nPassed {passed}/{len(test_cases)} tests.")
    
if __name__ == "__main__":
    run_tests()
