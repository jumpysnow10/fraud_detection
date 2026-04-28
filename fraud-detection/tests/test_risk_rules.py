import pytest
import pandas as pd

from risk_rules import label_risk, score_transaction
from analyze_fraud import summarize_results
from features import build_model_frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def base_tx(**overrides):
    """Minimal transaction with every signal in its lowest tier (score = 0)."""
    tx = {
        "device_risk_score": 10,
        "is_international": 0,
        "amount_usd": 100.0,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


# ---------------------------------------------------------------------------
# label_risk
# ---------------------------------------------------------------------------

def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


def test_label_risk_boundaries():
    assert label_risk(29) == "low"
    assert label_risk(30) == "medium"
    assert label_risk(59) == "medium"
    assert label_risk(60) == "high"


# ---------------------------------------------------------------------------
# score_transaction — each signal in isolation
# ---------------------------------------------------------------------------

def test_baseline_score_is_zero():
    assert score_transaction(base_tx()) == 0


def test_high_device_risk_adds_25():
    assert score_transaction(base_tx(device_risk_score=70)) == 25
    assert score_transaction(base_tx(device_risk_score=85)) == 25


def test_medium_device_risk_adds_10():
    assert score_transaction(base_tx(device_risk_score=40)) == 10
    assert score_transaction(base_tx(device_risk_score=69)) == 10


def test_low_device_risk_adds_nothing():
    assert score_transaction(base_tx(device_risk_score=39)) == 0


def test_international_adds_15():
    assert score_transaction(base_tx(is_international=1)) == 15


def test_domestic_adds_nothing():
    assert score_transaction(base_tx(is_international=0)) == 0


def test_large_amount_adds_25():
    assert score_transaction(base_tx(amount_usd=1000)) == 25
    assert score_transaction(base_tx(amount_usd=1500)) == 25


def test_medium_amount_adds_10():
    assert score_transaction(base_tx(amount_usd=500)) == 10
    assert score_transaction(base_tx(amount_usd=999)) == 10


def test_small_amount_adds_nothing():
    assert score_transaction(base_tx(amount_usd=499)) == 0


def test_high_velocity_adds_20():
    assert score_transaction(base_tx(velocity_24h=6)) == 20
    assert score_transaction(base_tx(velocity_24h=10)) == 20


def test_medium_velocity_adds_5():
    assert score_transaction(base_tx(velocity_24h=3)) == 5
    assert score_transaction(base_tx(velocity_24h=5)) == 5


def test_low_velocity_adds_nothing():
    assert score_transaction(base_tx(velocity_24h=2)) == 0


def test_high_failed_logins_adds_20():
    assert score_transaction(base_tx(failed_logins_24h=5)) == 20
    assert score_transaction(base_tx(failed_logins_24h=9)) == 20


def test_medium_failed_logins_adds_10():
    assert score_transaction(base_tx(failed_logins_24h=2)) == 10
    assert score_transaction(base_tx(failed_logins_24h=4)) == 10


def test_no_failed_logins_adds_nothing():
    assert score_transaction(base_tx(failed_logins_24h=0)) == 0


def test_multiple_prior_chargebacks_add_20():
    assert score_transaction(base_tx(prior_chargebacks=2)) == 20
    assert score_transaction(base_tx(prior_chargebacks=5)) == 20


def test_single_prior_chargeback_adds_5():
    assert score_transaction(base_tx(prior_chargebacks=1)) == 5


def test_no_prior_chargebacks_adds_nothing():
    assert score_transaction(base_tx(prior_chargebacks=0)) == 0


# ---------------------------------------------------------------------------
# score_transaction — clamping and combined profiles
# ---------------------------------------------------------------------------

def test_score_minimum_is_zero():
    assert score_transaction(base_tx()) == 0


def test_score_clamped_at_100():
    tx = base_tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=2000,
        velocity_24h=8,
        failed_logins_24h=6,
        prior_chargebacks=3,
    )
    assert score_transaction(tx) == 100


def test_high_risk_profile_scores_high():
    # Mirrors transaction 50003 from the dataset (confirmed chargeback, $1,250)
    # device(+25) + international(+15) + amount(+25) + velocity(+20) + logins(+20) = 105 → capped 100
    tx = base_tx(
        device_risk_score=81,
        is_international=1,
        amount_usd=1250,
        velocity_24h=6,
        failed_logins_24h=5,
        prior_chargebacks=0,
    )
    assert score_transaction(tx) == 100
    assert label_risk(score_transaction(tx)) == "high"


def test_low_risk_profile_scores_low():
    tx = base_tx(device_risk_score=8, amount_usd=45)
    assert label_risk(score_transaction(tx)) == "low"


def test_signals_are_additive():
    # medium device(+10) + medium amount(+10) = 20
    tx = base_tx(device_risk_score=50, amount_usd=750)
    assert score_transaction(tx) == 20


# ---------------------------------------------------------------------------
# summarize_results
# ---------------------------------------------------------------------------

def _make_scored(*rows):
    return pd.DataFrame(rows, columns=["transaction_id", "amount_usd", "risk_label"])


def _make_chargebacks(*ids):
    return pd.DataFrame({"transaction_id": list(ids)})


def test_summarize_transaction_counts():
    scored = _make_scored(
        (1, 100.0, "low"),
        (2, 200.0, "low"),
        (3, 500.0, "high"),
    )
    result = summarize_results(scored, _make_chargebacks(3))
    assert result.loc[result["risk_label"] == "high", "transactions"].iloc[0] == 1
    assert result.loc[result["risk_label"] == "low", "transactions"].iloc[0] == 2


def test_summarize_total_and_avg_amount():
    scored = _make_scored(
        (1, 100.0, "low"),
        (2, 300.0, "low"),
    )
    result = summarize_results(scored, _make_chargebacks())
    row = result[result["risk_label"] == "low"].iloc[0]
    assert row["total_amount_usd"] == pytest.approx(400.0)
    assert row["avg_amount_usd"] == pytest.approx(200.0)


def test_summarize_chargeback_rate():
    scored = _make_scored(
        (1, 100.0, "high"),
        (2, 200.0, "high"),
        (3, 300.0, "high"),
        (4, 50.0, "low"),
    )
    result = summarize_results(scored, _make_chargebacks(1, 2))
    assert result.loc[result["risk_label"] == "high", "chargeback_rate"].iloc[0] == pytest.approx(2 / 3)
    assert result.loc[result["risk_label"] == "low", "chargeback_rate"].iloc[0] == 0.0


def test_summarize_no_chargebacks_gives_zero_rate():
    scored = _make_scored(
        (1, 100.0, "low"),
        (2, 200.0, "medium"),
    )
    result = summarize_results(scored, _make_chargebacks())
    assert (result["chargeback_rate"] == 0.0).all()


def test_summarize_all_transactions_are_fraud():
    scored = _make_scored(
        (1, 500.0, "high"),
        (2, 600.0, "high"),
    )
    result = summarize_results(scored, _make_chargebacks(1, 2))
    assert result.loc[result["risk_label"] == "high", "chargeback_rate"].iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_model_frame
# ---------------------------------------------------------------------------

def test_build_model_frame_joins_prior_chargebacks():
    transactions = pd.DataFrame({"transaction_id": [1], "account_id": [42], "amount_usd": [100.0]})
    accounts = pd.DataFrame({"account_id": [42], "prior_chargebacks": [3]})
    result = build_model_frame(transactions, accounts)
    assert result.loc[0, "prior_chargebacks"] == 3


def test_build_model_frame_preserves_all_transactions():
    # Left join: transactions with no matching account row must still appear
    transactions = pd.DataFrame({
        "transaction_id": [1, 2],
        "account_id": [10, 99],
        "amount_usd": [50.0, 75.0],
    })
    accounts = pd.DataFrame({"account_id": [10], "prior_chargebacks": [1]})
    result = build_model_frame(transactions, accounts)
    assert len(result) == 2
