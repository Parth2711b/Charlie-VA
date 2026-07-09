"""
handlers/calculator.py — Evaluate simple math expressions offline.
No LLM needed — pure Python eval in a sandboxed context.
"""

import re
import math
import logging

logger = logging.getLogger("Charlie.handler.calculator")

# Safe math namespace — no builtins that could be dangerous
SAFE_MATH = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
SAFE_MATH.update({"abs": abs, "round": round, "min": min, "max": max})


def _extract_expression(text: str) -> str:
    """Pull a math expression from spoken text."""
    text_lower = text.lower()

    # Remove filler words
    cleaned = re.sub(
        r"\b(what is|calculate|compute|evaluate|solve|whats|equals?|the answer to)\b",
        "", text_lower, flags=re.IGNORECASE
    ).strip()

    # Convert spoken words to operators
    replacements = {
        r"\bplus\b":           "+",
        r"\bminus\b":          "-",
        r"\btimes\b":          "*",
        r"\bmultiplied by\b":  "*",
        r"\bdivided by\b":     "/",
        r"\bover\b":           "/",
        r"\bto the power of\b": "**",
        r"\bsquared\b":        "**2",
        r"\bcubed\b":          "**3",
        r"\bsquare root of\b": "math.sqrt",
        r"\bsqrt\b":           "math.sqrt",
        r"\bpi\b":             "math.pi",
    }
    for pattern, replacement in replacements.items():
        cleaned = re.sub(pattern, replacement, cleaned)

    # Keep only safe characters
    cleaned = re.sub(r"[^0-9+\-*/().% \t]", "", cleaned).strip()
    return cleaned


def handle(text: str) -> str:
    expr = _extract_expression(text)
    if not expr:
        return "I didn't catch a math expression. Try saying 'what is 12 times 8'."

    try:
        result = eval(expr, {"__builtins__": {}}, SAFE_MATH)  # noqa: S307
        # Format nicely
        if isinstance(result, float):
            if result == int(result):
                result = int(result)
            else:
                result = round(result, 6)
        logger.info("Calc: %s = %s", expr, result)
        return f"{expr} equals {result}."
    except ZeroDivisionError:
        return "Division by zero is undefined."
    except Exception as e:
        logger.warning("Calc eval failed for '%s': %s", expr, e)
        return f"I couldn't calculate that. Try a simpler expression."
