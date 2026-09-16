"""Tester för src/dashboard.py. Syntetisk data, inget beroende på den sparade modellen."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.dashboard import (
    GROUP_COLUMNS,
    active_customers,
    filter_customers,
    group_breakdown,
    kpis,
    revenue_forecast,
)
from src.data import TARGET_COLUMN, TENURE_LABELS
from tests.conftest import make_raw_df


@pytest.fixture
def customers():
    """Rådata plus en påhittad churn-sannolikhet i [0, 1], deterministisk."""
    df = make_raw_df()
    df["churn_probability"] = np.linspace(0.0, 1.0, len(df))
    return df


@pytest.fixture
def three_customers():
    """Tre aktiva kunder med handvalda värden så att (1 - p)^m kan räknas för hand."""
    return pd.DataFrame(
        {
            "MonthlyCharges": [100.0, 50.0, 20.0],
            "churn_probability": [0.5, 0.1, 0.0],
            TARGET_COLUMN: ["No", "No", "No"],
        }
    )


START = date(2026, 9, 16)


def test_filter_customers_tom_dict_ger_alla_rader(customers):
    result = filter_customers(customers, {})
    assert len(result) == len(customers)


def test_filter_customers_tom_lista_ger_inget_filter(customers):
    result = filter_customers(customers, {"Contract": []})
    assert len(result) == len(customers)


def test_filter_customers_kombinerar_och_eller(customers):
    filters = {"Contract": ["One year", "Two year"], "InternetService": ["Fiber optic"]}
    result = filter_customers(customers, filters)
    expected = customers["Contract"].isin(["One year", "Two year"]) & (
        customers["InternetService"] == "Fiber optic"
    )
    assert len(result) == expected.sum()
    assert set(result["Contract"]) <= {"One year", "Two year"}
    assert (result["InternetService"] == "Fiber optic").all()


def test_filter_customers_fungerar_pa_heltalskolumn(customers):
    result = filter_customers(customers, {"SeniorCitizen": [1]})
    assert (result["SeniorCitizen"] == 1).all()


def test_filter_customers_returnerar_kopia(customers):
    result = filter_customers(customers, {"Contract": ["Two year"]})
    result["ny_kolumn"] = 1  # skulle varna om result vore en vy
    assert "ny_kolumn" not in customers.columns


def test_filter_customers_okand_kolumn_ger_keyerror(customers):
    with pytest.raises(KeyError, match="FinnsInte"):
        filter_customers(customers, {"FinnsInte": ["x"]})


def test_filter_customers_noll_rader_ger_valueerror(customers):
    with pytest.raises(ValueError):
        filter_customers(customers, {"Contract": ["Finns inte"]})


def test_active_customers_ger_bara_churn_no(customers):
    result = active_customers(customers)
    assert (result[TARGET_COLUMN] == "No").all()
    assert len(result) == len(customers) // 2  # make_raw_df: "No" på jämna index


def test_active_customers_utan_churn_kolumn_ger_keyerror(customers):
    with pytest.raises(KeyError, match=TARGET_COLUMN):
        active_customers(customers.drop(columns=[TARGET_COLUMN]))


def test_revenue_forecast_ger_en_rad_per_manad_plus_start(customers):
    result = revenue_forecast(customers, months=24, start=START)
    assert len(result) == 25
    assert list(result["months_ahead"]) == list(range(25))


def test_revenue_forecast_rad_noll_ar_dagens_mrr(three_customers):
    result = revenue_forecast(three_customers, months=3, start=START)
    first = result.iloc[0]
    assert first["expected_mrr"] == 170.0
    assert first["expected_customers"] == 3.0
    assert first["expected_loss"] == 0.0
    assert first["cumulative_loss"] == 0.0


def test_revenue_forecast_handraknade_varden(three_customers):
    result = revenue_forecast(three_customers, months=2, start=START)
    # m = 1: 100*0.5 + 50*0.9 + 20*1.0 = 115, kunder 0.5 + 0.9 + 1.0 = 2.4
    assert result.loc[1, "expected_mrr"] == pytest.approx(115.0)
    assert result.loc[1, "expected_customers"] == pytest.approx(2.4)
    assert result.loc[1, "expected_loss"] == pytest.approx(170.0 - 115.0)
    # m = 2: 100*0.25 + 50*0.81 + 20*1.0 = 85.5, kunder 0.25 + 0.81 + 1.0 = 2.06
    assert result.loc[2, "expected_mrr"] == pytest.approx(85.5)
    assert result.loc[2, "expected_customers"] == pytest.approx(2.06)
    assert result.loc[2, "expected_loss"] == pytest.approx(115.0 - 85.5)
    assert result.loc[2, "cumulative_loss"] == pytest.approx(170.0 - 85.5)


def test_revenue_forecast_ignorerar_churnade_kunder(three_customers):
    churned = pd.DataFrame(
        {"MonthlyCharges": [9999.0], "churn_probability": [0.0], TARGET_COLUMN: ["Yes"]}
    )
    with_churned = pd.concat([three_customers, churned], ignore_index=True)
    expected = revenue_forecast(three_customers, months=6, start=START)
    result = revenue_forecast(with_churned, months=6, start=START)
    pd.testing.assert_frame_equal(result, expected)


def test_revenue_forecast_identiteter(customers):
    result = revenue_forecast(customers, months=12, start=START)
    mrr = result["expected_mrr"]
    loss = result["expected_loss"]
    for m in range(1, 13):
        assert loss[m] == pytest.approx(mrr[m - 1] - mrr[m])
        assert result.loc[m, "cumulative_loss"] == pytest.approx(loss[1 : m + 1].sum())


def test_revenue_forecast_manader_pa_svenska_fran_start(three_customers):
    result = revenue_forecast(three_customers, months=4, start=START)
    assert result.loc[0, "month"] == pd.Timestamp("2026-09-01")
    assert result.loc[1, "month"] == pd.Timestamp("2026-10-01")
    assert list(result["month_label"]) == [
        "Sep 2026",
        "Okt 2026",
        "Nov 2026",
        "Dec 2026",
        "Jan 2027",
    ]


@pytest.mark.parametrize("column", ["churn_probability", "MonthlyCharges"])
def test_revenue_forecast_saknad_kolumn_ger_keyerror(three_customers, column):
    with pytest.raises(KeyError, match=column):
        revenue_forecast(three_customers.drop(columns=[column]), start=START)


def test_revenue_forecast_sannolikhet_utanfor_intervall_ger_valueerror(three_customers):
    three_customers.loc[0, "churn_probability"] = 1.5
    with pytest.raises(ValueError, match="churn_probability"):
        revenue_forecast(three_customers, start=START)


def test_revenue_forecast_months_under_ett_ger_valueerror(three_customers):
    with pytest.raises(ValueError, match="months"):
        revenue_forecast(three_customers, months=0, start=START)


def test_revenue_forecast_utan_aktiva_kunder_ger_valueerror(three_customers):
    three_customers[TARGET_COLUMN] = "Yes"
    with pytest.raises(ValueError, match="aktiva"):
        revenue_forecast(three_customers, start=START)


def test_kpis_stammer_med_revenue_forecast(customers):
    result = kpis(customers)
    forecast = revenue_forecast(customers, start=START)
    assert result["mrr_today"] == forecast.loc[0, "expected_mrr"]
    assert result["expected_loss_next_month"] == forecast.loc[1, "expected_loss"]


def test_kpis_returnerar_vanliga_python_typer(customers):
    result = kpis(customers)
    assert set(result) == {
        "mrr_today",
        "expected_loss_next_month",
        "actual_loss_last_month",
        "predicted_churn_rate",
        "n_customers",
    }
    assert type(result["n_customers"]) is int
    for key in result.keys() - {"n_customers"}:
        assert type(result[key]) is float


def test_kpis_faktisk_forlust_bara_pa_churnade(three_customers):
    churned = pd.DataFrame(
        {"MonthlyCharges": [9999.0], "churn_probability": [0.0], TARGET_COLUMN: ["Yes"]}
    )
    result = kpis(pd.concat([three_customers, churned], ignore_index=True))
    assert result["actual_loss_last_month"] == 9999.0
    assert result["mrr_today"] == 170.0  # den churnade kunden ingår inte
    assert result["n_customers"] == 3
    assert result["predicted_churn_rate"] == pytest.approx(0.2)  # (0.5 + 0.1 + 0.0) / 3
    assert result["expected_loss_next_month"] == pytest.approx(55.0)  # 100*0.5 + 50*0.1


def test_kpis_utan_aktiva_kunder_ger_valueerror(three_customers):
    three_customers[TARGET_COLUMN] = "Yes"
    with pytest.raises(ValueError, match="aktiva"):
        kpis(three_customers)


def test_group_breakdown_kolumner_och_sortering(customers):
    result = group_breakdown(customers, "Contract")
    assert list(result.columns) == [
        "Contract",
        "antal_kunder",
        "mrr",
        "predikterad_churn",
        "forvantad_forlust_nasta_manad",
    ]
    loss = result["forvantad_forlust_nasta_manad"]
    assert loss.is_monotonic_decreasing
    assert list(result.index) == list(range(len(result)))


def test_group_breakdown_summerar_till_kpis(customers):
    result = group_breakdown(customers, "InternetService")
    totals = kpis(customers)
    assert result["antal_kunder"].sum() == totals["n_customers"]
    assert result["mrr"].sum() == pytest.approx(totals["mrr_today"])
    assert result["forvantad_forlust_nasta_manad"].sum() == pytest.approx(
        totals["expected_loss_next_month"]
    )


def test_group_breakdown_predikterad_churn_ar_gruppens_medel(customers):
    result = group_breakdown(customers, "Contract").set_index("Contract")
    active = active_customers(customers)
    for contract in active["Contract"].unique():
        expected = active.loc[active["Contract"] == contract, "churn_probability"].mean()
        assert result.loc[contract, "predikterad_churn"] == pytest.approx(expected)


def test_group_breakdown_fungerar_pa_tenure_group(customers):
    result = group_breakdown(customers, "tenure_group")
    assert set(result["tenure_group"]) <= set(TENURE_LABELS)
    assert result["antal_kunder"].sum() == len(active_customers(customers))


def test_group_breakdown_otillaten_kolumn_ger_keyerror(customers):
    assert "gender" in customers.columns and "gender" not in GROUP_COLUMNS
    with pytest.raises(KeyError, match="gender"):
        group_breakdown(customers, "gender")


def test_group_breakdown_utan_aktiva_kunder_ger_valueerror(customers):
    customers[TARGET_COLUMN] = "Yes"
    with pytest.raises(ValueError, match="aktiva"):
        group_breakdown(customers, "Contract")
