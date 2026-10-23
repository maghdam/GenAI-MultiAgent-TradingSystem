"""Non-overridable activation gate for the unapproved bracket runtime.

No environment variable, saved protection observation or UI toggle grants
approval. A separately reviewed activation implementation is required.
"""


def paper_bracket_automation_readiness():
    return {
        "automatic_trading_allowed": False,
        "activation_available": False,
        "blockers": [
            "Exact-head local regression and required GitHub CI/review approval",
            "Explicit operator approval of automatic Paper trading",
            "Authorized broker field validation of partial fills, OCO and cancellation",
            "Continuous protection monitoring and bounded failure escalation",
            "Exclusive account ownership during cancellation-to-close interval",
            "Supervised recovery of missing/replaced legs and ambiguous failed mutations",
        ],
        "paper_only": True,
        "crypto_entry_allowed": False,
        "live_allowed": False,
    }
