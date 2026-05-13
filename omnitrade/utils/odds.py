"""American odds to decimal odds conversion."""


def american_to_decimal(american_odds: float) -> float:
    if american_odds > 0:
        return 1.0 + american_odds / 100.0
    elif american_odds < 0:
        return 1.0 + 100.0 / abs(american_odds)
    return 2.0
