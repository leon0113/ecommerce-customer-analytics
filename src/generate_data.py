"""Generate the synthetic e-commerce dataset described in config.yaml.

Customers with hidden behaviour types place orders month by month until they churn.
Order lines, payments and returns follow from those orders. Data quality issues are
then injected and logged as ground truth. Run with: python -m src.generate_data
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from src.common import load_config, project_path, setup_logging
from src.messiness import IssueInjector

logger = logging.getLogger(__name__)

CUSTOMER_COLUMNS = [
    "customer_id", "full_name", "gender", "phone", "division", "district",
    "acquisition_channel", "signup_date",
]
PRODUCT_COLUMNS = ["product_id", "product_name", "category_name", "brand", "list_price"]
ORDER_COLUMNS = [
    "order_id", "customer_id", "order_date", "order_status", "discount_pct", "delivery_district",
]


def type_param(cfg: dict, key: str, default: float | None = None) -> np.ndarray:
    """One value per customer type, in config order."""
    return np.array([t.get(key, default) for t in cfg["customer_types"].values()], dtype=float)


def category_param(cfg: dict, key: str) -> np.ndarray:
    """One value per category, in config order."""
    return np.array([c[key] for c in cfg["categories"].values()])


def build_calendar(cfg: dict) -> pd.DataFrame:
    """One row per day with the weekday x event weight that decides when orders happen."""
    season = cfg["seasonality"]
    dates = pd.date_range(cfg["dates"]["start"], cfg["dates"]["end"], freq="D")
    weekday_weight = np.array([season["weekday_weights"][day] for day in range(7)])
    event_lift = np.ones(len(dates))
    for event in season["events"]:
        for event_date in pd.to_datetime(event["dates"]):
            window_start = event_date - pd.Timedelta(days=event["days_before"])
            in_window = (dates >= window_start) & (dates <= event_date)
            event_lift[in_window] = np.maximum(event_lift[in_window], event["lift"])
    return pd.DataFrame({
        "date": dates,
        "month_idx": (dates.year - dates[0].year) * 12 + dates.month - dates[0].month,
        "month": dates.month,
        "event_lift": event_lift,
        "weight": weekday_weight[dates.dayofweek] * event_lift,
    })


def build_type_intensity(calendar: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Relative order intensity per (customer type, month).

    Every type follows the seasonal curve; a type with `event_focus` moves that share of
    its activity into the months that contain Eid and sale events.
    """
    monthly = (
        calendar.assign(event_extra=calendar["event_lift"] - 1)
        .groupby("month_idx")
        .agg(month=("month", "first"), mean_weight=("weight", "mean"), event_extra=("event_extra", "sum"))
    )
    base = monthly["month"].map(cfg["seasonality"]["month_weights"]).to_numpy() * monthly["mean_weight"].to_numpy()
    event = monthly["event_extra"].to_numpy()
    event = event * base.mean() / event.mean()
    focus = type_param(cfg, "event_focus", 0.0)
    return (1 - focus)[:, None] * base + focus[:, None] * event


def build_name_pools(cfg: dict) -> dict[str, np.ndarray]:
    """Draw pools of Bengali names from Faker once, so customers can be named vectorized."""
    fake = Faker(cfg["generation"]["faker_locale"])
    fake.seed_instance(cfg["seed"])
    size = cfg["generation"]["name_pool_size"]
    return {
        "Male": np.array([fake.first_name_male() for _ in range(size)]),
        "Female": np.array([fake.first_name_female() for _ in range(size)]),
        "last": np.array([fake.last_name() for _ in range(size)]),
    }


def sample_locations(cfg: dict, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample a division by weight, then a district in it; the first-listed district is the largest."""
    geo = cfg["geography"]
    divisions = geo["divisions"]
    names = list(divisions)
    division = rng.choice(names, size=n, p=[divisions[d]["weight"] for d in names])
    district = np.empty(n, dtype=object)
    for name in names:
        in_division = division == name
        districts = divisions[name]["districts"]
        probs = np.full(len(districts), (1 - geo["hq_district_share"]) / (len(districts) - 1))
        probs[0] = geo["hq_district_share"]
        district[in_division] = rng.choice(districts, size=in_division.sum(), p=probs)
    return division, district


def sample_signup_dates(
    cfg: dict, calendar: pd.DataFrame, n: int, rng: np.random.Generator
) -> pd.Series:
    """Existing customers joined before the window; new ones arrive at a steadily growing rate."""
    growth = cfg["growth"]
    start = calendar["date"].iloc[0]
    n_existing = round(n * growth["existing_customer_share"])
    days_before = rng.integers(1, growth["existing_signup_years"] * 365 + 1, size=n_existing)
    existing = (start - pd.to_timedelta(days_before, unit="D")).to_numpy()

    acquisition = (1 + growth["yoy_acquisition_growth"]) ** (np.arange(len(calendar)) / 365)
    new_days = rng.choice(len(calendar), size=n - n_existing, p=acquisition / acquisition.sum())
    new = calendar["date"].to_numpy()[new_days]

    signup = pd.Series(np.concatenate([existing, new]))
    return signup + pd.to_timedelta(rng.integers(0, 86400, size=n), unit="s")


def generate_customers(cfg: dict, calendar: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Customers with public attributes plus hidden behaviour columns used by the simulation."""
    n = cfg["volume"]["customers"]
    noise = cfg["customer_noise"]
    channels = cfg["acquisition_channels"]
    type_names = np.array(list(cfg["customer_types"]))
    type_idx = rng.choice(len(type_names), size=n, p=type_param(cfg, "share"))

    genders = cfg["gender_split"]
    gender = rng.choice(list(genders), size=n, p=list(genders.values()))
    names = build_name_pools(cfg)
    first_name = np.where(gender == "Male", rng.choice(names["Male"], size=n), rng.choice(names["Female"], size=n))
    division, district = sample_locations(cfg, n, rng)
    operator_digit = pd.Series(rng.integers(3, 10, size=n)).astype(str)
    subscriber = pd.Series(rng.integers(0, 10**8, size=n)).astype(str).str.zfill(8)

    customers = pd.DataFrame({
        "full_name": pd.Series(first_name) + " " + pd.Series(rng.choice(names["last"], size=n)),
        "gender": gender,
        "phone": "01" + operator_digit + subscriber,
        "division": division,
        "district": district,
        "acquisition_channel": rng.choice(list(channels), size=n, p=[c["share"] for c in channels.values()]),
        "signup_date": sample_signup_dates(cfg, calendar, n, rng),
        "customer_type": type_names[type_idx],
        "type_idx": type_idx,
        "activity": type_param(cfg, "order_weight")[type_idx]
        * rng.lognormal(0, cfg["customer_activity_sigma"], size=n),
        "discount_propensity": np.clip(
            type_param(cfg, "discount_prob")[type_idx] + rng.normal(0, noise["discount_prop_sd"], size=n),
            0.01, 0.99,
        ),
        "return_propensity": type_param(cfg, "return_mult")[type_idx]
        * rng.lognormal(0, noise["return_prop_sigma"], size=n),
    })
    customers = customers.sort_values("signup_date", kind="mergesort").reset_index(drop=True)
    customers.insert(0, "customer_id", np.arange(1, n + 1))

    start = calendar["date"].iloc[0]
    signup = customers["signup_date"]
    customers["signup_month_idx"] = (signup.dt.year - start.year) * 12 + signup.dt.month - start.month
    customers["signup_day_idx"] = (signup.dt.normalize() - start).dt.days
    return customers


def generate_categories(cfg: dict) -> pd.DataFrame:
    """Category lookup table with canonical names."""
    names = list(cfg["categories"])
    return pd.DataFrame({"category_id": np.arange(1, len(names) + 1), "category_name": names})


def generate_products(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Products with log-normal prices per category and a hidden Zipf-like popularity."""
    n = cfg["volume"]["products"]
    categories = cfg["categories"]
    cat_idx = rng.choice(len(categories), size=n, p=category_param(cfg, "product_share"))

    price = rng.lognormal(np.log(category_param(cfg, "price_median")[cat_idx]), category_param(cfg, "price_sigma")[cat_idx])
    price = np.maximum(np.round(price, -1), cfg["products"]["min_price"])

    noun = np.empty(n, dtype=object)
    for i, category in enumerate(categories.values()):
        in_category = cat_idx == i
        noun[in_category] = rng.choice(category["nouns"], size=in_category.sum())
    model_number = pd.Series(rng.integers(100, 1000, size=n)).astype(str)
    popularity_rank = rng.permutation(n) + 1

    return pd.DataFrame({
        "product_id": np.arange(1, n + 1),
        "product_name": pd.Series(noun) + " " + model_number,
        "category_name": np.array(list(categories))[cat_idx],
        "brand": rng.choice(cfg["products"]["brands"], size=n),
        "list_price": price,
        "popularity": 1.0 / popularity_rank ** cfg["product_popularity_zipf_a"],
        "return_rate": category_param(cfg, "return_rate")[cat_idx],
        "extra_qty_mean": category_param(cfg, "extra_qty_mean")[cat_idx],
    })


def product_choice_probs(products: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Per customer type, a probability over products: popularity tilted by price_preference."""
    relative_price = (products["list_price"] / products["list_price"].median()).to_numpy()
    preference = type_param(cfg, "price_preference")
    weights = products["popularity"].to_numpy() * relative_price ** preference[:, None]
    return weights / weights.sum(axis=1, keepdims=True)


def simulate_order_counts(
    customers: pd.DataFrame, intensity: np.ndarray, cfg: dict, scale: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate orders per (customer, month) with monthly churn.

    Most new customers order in their signup month (accounts are usually created at
    checkout); after that, orders follow the customer's intensity. Churn odds rise with months since the last order, discount dependence and return
    tendency, and are shifted by customer type and acquisition channel.
    Returns the order count matrix and each customer's churn month index (-1 = still active).
    """
    churn = cfg["churn"]
    n, n_months = len(customers), intensity.shape[1]
    type_idx = customers["type_idx"].to_numpy()
    retention = customers["acquisition_channel"].map(
        {name: c["retention_mult"] for name, c in cfg["acquisition_channels"].items()}
    ).to_numpy()
    mean_return_rate = category_param(cfg, "return_rate").mean()
    base_logit = (
        churn["intercept"]
        + type_param(cfg, "churn_offset")[type_idx]
        - np.log(retention)
        + churn["coef_discount_share"] * customers["discount_propensity"].to_numpy()
        + churn["coef_return_rate"] * mean_return_rate * customers["return_propensity"].to_numpy()
    )
    order_rate = scale * customers["activity"].to_numpy()[:, None] * intensity[type_idx]
    max_orders = type_param(cfg, "max_orders", np.inf)[type_idx]
    signup_month = customers["signup_month_idx"].to_numpy()
    first_order_prob = cfg["orders"]["first_order_prob"]

    counts = np.zeros((n, n_months), dtype=int)
    total_orders = np.zeros(n)
    months_since_order = np.zeros(n)
    active = np.ones(n, dtype=bool)
    churn_month = np.full(n, -1)
    for month in range(n_months):
        live = active & (signup_month <= month)
        first_order = (signup_month == month) & (rng.random(n) < first_order_prob)
        month_counts = np.maximum(rng.poisson(order_rate[:, month]), first_order) * live
        month_counts = np.minimum(month_counts, max_orders - total_orders).astype(int)
        counts[:, month] = month_counts
        total_orders += month_counts
        months_since_order = np.where(month_counts > 0, 0, np.where(live, months_since_order + 1, 0))
        logit = base_logit + churn["coef_months_since_last_order"] * months_since_order
        churned = live & ((rng.random(n) < 1 / (1 + np.exp(-logit))) | (total_orders >= max_orders))
        active &= ~churned
        churn_month[churned] = month
    return counts, churn_month


def sample_order_datetimes(
    customer_idx: np.ndarray,
    month_idx: np.ndarray,
    customers: pd.DataFrame,
    calendar: pd.DataFrame,
    cfg: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick a day in each order's month by daily weight (not before signup), then an hour by hour weight.

    Days are drawn by inverting the cumulative daily weight inside the allowed range.
    """
    cumulative = np.concatenate([[0.0], np.cumsum(calendar["weight"].to_numpy())])
    day_numbers = pd.Series(np.arange(len(calendar))).groupby(calendar["month_idx"])
    month_first = day_numbers.min().to_numpy()
    month_last = day_numbers.max().to_numpy()

    low = np.maximum(month_first[month_idx], customers["signup_day_idx"].to_numpy()[customer_idx])
    high = month_last[month_idx] + 1
    point = cumulative[low] + rng.random(len(low)) * (cumulative[high] - cumulative[low])
    day = np.clip(np.searchsorted(cumulative, point, side="right") - 1, low, high - 1)

    hour_weights = np.array(cfg["seasonality"]["hour_weights"])
    hour = rng.choice(24, size=len(day), p=hour_weights / hour_weights.sum())
    seconds = hour * 3600 + rng.integers(0, 3600, size=len(day))
    order_date = calendar["date"].to_numpy()[day] + pd.to_timedelta(seconds, unit="s").to_numpy()
    # An account can be created at checkout, so an order may share its signup timestamp.
    return np.maximum(order_date, customers["signup_date"].to_numpy()[customer_idx])


def generate_orders(
    customers: pd.DataFrame, calendar: pd.DataFrame, cfg: dict, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    """Simulate who orders when, calibrated and trimmed to the configured order volume.

    Fewer orders mean longer gaps and therefore more churn, so volume is not linear in the
    intensity scale; the scale is re-estimated over a few rounds until volume lands just
    above the target.
    """
    gen = cfg["generation"]
    target = cfg["volume"]["orders"]
    intensity = build_type_intensity(calendar, cfg)
    scale = 1.0
    for _ in range(gen["calibration_rounds"]):
        counts, churn_month = simulate_order_counts(customers, intensity, cfg, scale, rng)
        total = counts.sum()
        if target <= total <= target * gen["calibration_max_overshoot"]:
            break
        scale *= gen["calibration_headroom"] * target / total

    customer_idx, month_idx = np.nonzero(counts)
    repeats = counts[customer_idx, month_idx]
    customer_idx, month_idx = np.repeat(customer_idx, repeats), np.repeat(month_idx, repeats)
    if len(customer_idx) > target:
        keep = np.sort(rng.choice(len(customer_idx), size=target, replace=False))
        customer_idx, month_idx = customer_idx[keep], month_idx[keep]
    else:
        logger.warning("Simulation produced %d orders, below the target of %d", len(customer_idx), target)

    n = len(customer_idx)
    uses_discount = rng.random(n) < customers["discount_propensity"].to_numpy()[customer_idx]
    elsewhere = rng.random(n) < cfg["orders"]["delivery_elsewhere_rate"]
    districts = customers["district"].to_numpy()
    orders = pd.DataFrame({
        "customer_id": customers["customer_id"].to_numpy()[customer_idx],
        "order_date": sample_order_datetimes(customer_idx, month_idx, customers, calendar, cfg, rng),
        "discount_pct": np.where(uses_discount, rng.choice(cfg["discounts"]["levels"], size=n), 0.0),
        "delivery_district": np.where(elsewhere, rng.choice(districts, size=n), districts[customer_idx]),
    })
    orders = orders.sort_values("order_date", kind="mergesort").reset_index(drop=True)
    orders.insert(0, "order_id", np.arange(1, n + 1))
    logger.info("Simulated %d orders (intensity scale %.4f)", n, scale)
    return orders, churn_month


def build_order_items(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    choice_probs: np.ndarray,
    cfg: dict,
    rng: np.random.Generator,
    first_item_id: int,
) -> pd.DataFrame:
    """Order lines: a random line count per order, products chosen by the customer type's preferences."""
    n_lines = 1 + rng.poisson(cfg["volume"]["mean_items_per_order"] - 1, size=len(orders))
    order_pos = np.repeat(np.arange(len(orders)), n_lines)
    line_type = customers["type_idx"].to_numpy()[orders["customer_id"].to_numpy() - 1][order_pos]
    product_idx = np.empty(len(order_pos), dtype=int)
    for type_i, probs in enumerate(choice_probs):
        is_type = line_type == type_i
        product_idx[is_type] = rng.choice(len(products), size=is_type.sum(), p=probs)

    return pd.DataFrame({
        "order_item_id": np.arange(first_item_id, first_item_id + len(order_pos)),
        "order_id": orders["order_id"].to_numpy()[order_pos],
        "product_id": product_idx + 1,
        "quantity": 1 + rng.poisson(products["extra_qty_mean"].to_numpy()[product_idx]),
        "unit_price": products["list_price"].to_numpy()[product_idx],
    })


def build_payments(
    orders: pd.DataFrame, items: pd.DataFrame, cfg: dict, rng: np.random.Generator
) -> pd.DataFrame:
    """One payment per order (so payment_id equals order_id); COD is collected on delivery."""
    pay = cfg["payments"]
    n = len(orders)
    gross = (items["quantity"] * items["unit_price"]).groupby(items["order_id"]).sum()
    gross = gross.reindex(orders["order_id"]).to_numpy()
    method = rng.choice(list(pay["methods"]), size=n, p=list(pay["methods"].values()))
    low, high = pay["cod_delivery_days"]
    delay_days = np.where(method == "cash_on_delivery", rng.integers(low, high + 1, size=n), 0)
    delay = pd.to_timedelta(delay_days, unit="D") + pd.to_timedelta(rng.integers(30, 900, size=n), unit="s")

    return pd.DataFrame({
        "payment_id": orders["order_id"].to_numpy(),
        "order_id": orders["order_id"].to_numpy(),
        "payment_method": method,
        "amount": np.round(gross * (1 - orders["discount_pct"].to_numpy()), 2),
        "payment_status": np.where(rng.random(n) < pay["failure_rate"], "failed", "completed"),
        "payment_date": orders["order_date"].to_numpy() + delay.to_numpy(),
    })


def build_returns(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    cfg: dict,
    rng: np.random.Generator,
    first_return_id: int,
) -> pd.DataFrame:
    """Delivered lines are returned with probability category rate x customer return propensity."""
    ret = cfg["returns"]
    line_order = orders.set_index("order_id").loc[items["order_id"]]
    product_idx = items["product_id"].to_numpy() - 1
    customer_idx = line_order["customer_id"].to_numpy() - 1
    prob = np.minimum(
        products["return_rate"].to_numpy()[product_idx] * customers["return_propensity"].to_numpy()[customer_idx],
        ret["max_probability"],
    )
    returned = (line_order["order_status"].to_numpy() == "delivered") & (rng.random(len(items)) < prob)
    lines, line_order = items[returned], line_order[returned]

    n = len(lines)
    low, high = ret["days_after_order"]
    delay = pd.to_timedelta(rng.integers(low, high + 1, size=n), unit="D") + pd.to_timedelta(
        rng.integers(0, 86400, size=n), unit="s"
    )
    return pd.DataFrame({
        "return_id": np.arange(first_return_id, first_return_id + n),
        "order_item_id": lines["order_item_id"].to_numpy(),
        "return_date": line_order["order_date"].to_numpy() + delay.to_numpy(),
        "return_reason": rng.choice(list(ret["reasons"]), size=n, p=list(ret["reasons"].values())),
        "refund_amount": np.round(
            lines["quantity"].to_numpy() * lines["unit_price"].to_numpy() * (1 - line_order["discount_pct"].to_numpy()),
            2,
        ),
    })


def add_text_issues(df: pd.DataFrame, injector: IssueInjector, table: str, key: str, column: str) -> None:
    """Text noise and typos on separate rows, so each cell carries at most one text issue."""
    noisy = injector.text_noise(df, table, key, column)
    injector.typos(df, table, key, column, eligible=~df.index.isin(noisy))


def make_raw_customers(customers: pd.DataFrame, cfg: dict, injector: IssueInjector) -> pd.DataFrame:
    """Public customer columns with missing values (likelier for one-time buyers), text issues and mixed dates."""
    mess = cfg["messiness"]
    raw = customers[CUSTOMER_COLUMNS].copy()
    raw["signup_date"] = raw["signup_date"].dt.strftime(cfg["date_formats"]["canonical"])
    is_one_time = customers["customer_type"] == "one_time"
    weights = pd.Series(np.where(is_one_time, mess["missing_bias"]["one_time_customer_mult"], 1.0), index=raw.index)
    for column in mess["missing_columns"]["customers"]:
        injector.missing(raw, "customers", "customer_id", column, weights)
    for column in ["division", "district"]:
        add_text_issues(raw, injector, "customers", "customer_id", column)
    injector.date_formats(raw, "customers", "customer_id", "signup_date")
    return raw


def make_raw_products(products: pd.DataFrame, cfg: dict, injector: IssueInjector) -> pd.DataFrame:
    """Public product columns with missing brands (likelier for groceries) and category name issues."""
    bias = cfg["messiness"]["missing_bias"]
    raw = products[PRODUCT_COLUMNS].copy()
    is_unbranded = raw["category_name"] == bias["unbranded_category"]
    weights = pd.Series(np.where(is_unbranded, bias["unbranded_mult"], 1.0), index=raw.index)
    for column in cfg["messiness"]["missing_columns"]["products"]:
        injector.missing(raw, "products", "product_id", column, weights)
    add_text_issues(raw, injector, "products", "product_id", "category_name")
    return raw


def make_raw_orders(orders: pd.DataFrame, cfg: dict, injector: IssueInjector, n_customers: int) -> pd.DataFrame:
    """Orders with missing districts (likelier before the legacy cutoff), orphans, bad dates and duplicates."""
    mess = cfg["messiness"]
    bias = mess["missing_bias"]
    raw = orders[ORDER_COLUMNS].copy()
    is_legacy = orders["order_date"] < pd.Timestamp(bias["legacy_cutoff"])
    weights = pd.Series(np.where(is_legacy, bias["legacy_order_mult"], 1.0), index=raw.index)
    for column in mess["missing_columns"]["orders"]:
        injector.missing(raw, "orders", "order_id", column, weights)
    add_text_issues(raw, injector, "orders", "order_id", "delivery_district")
    injector.orphan_keys(raw, "orders", "order_id", "customer_id", n_customers)

    raw["order_date"] = raw["order_date"].dt.strftime(cfg["date_formats"]["canonical"])
    impossible = injector.impossible_dates(raw, "orders", "order_id", "order_date")
    injector.date_formats(raw, "orders", "order_id", "order_date", eligible=~raw.index.isin(impossible))
    return injector.duplicates(raw, "orders", "order_id")


def make_raw_order_items(items: pd.DataFrame, injector: IssueInjector, n_products: int) -> pd.DataFrame:
    """Order lines with negative quantities, price outliers and orphan product ids."""
    raw = items.copy()
    injector.negative_quantities(raw, "order_items", "order_item_id", "quantity")
    injector.price_outliers(raw, "order_items", "order_item_id", "unit_price")
    injector.orphan_keys(raw, "order_items", "order_item_id", "product_id", n_products)
    return raw


def make_raw_payments(payments: pd.DataFrame, cfg: dict, injector: IssueInjector) -> pd.DataFrame:
    """Payments with inconsistent method casing and mixed date formats."""
    raw = payments.copy()
    raw["payment_date"] = raw["payment_date"].dt.strftime(cfg["date_formats"]["canonical"])
    injector.text_noise(raw, "payments", "payment_id", "payment_method")
    injector.date_formats(raw, "payments", "payment_id", "payment_date")
    return raw


def make_raw_returns(returns: pd.DataFrame, cfg: dict, injector: IssueInjector) -> pd.DataFrame:
    """Returns with mixed date formats."""
    raw = returns.copy()
    raw["return_date"] = raw["return_date"].dt.strftime(cfg["date_formats"]["canonical"])
    injector.date_formats(raw, "returns", "return_id", "return_date")
    return raw


def append_csv(df: pd.DataFrame, path: Path, first_chunk: bool) -> None:
    """Write the first chunk with a header, append later chunks without one."""
    df.to_csv(path, mode="w" if first_chunk else "a", header=first_chunk, index=False, encoding="utf-8")


def write_order_tables(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    cfg: dict,
    rng: np.random.Generator,
    injector: IssueInjector,
    raw_dir: Path,
) -> dict[str, int]:
    """Build, mess up and write order-level tables chunk by chunk; return raw row counts."""
    choice_probs = product_choice_probs(products, cfg)
    chunk_size = cfg["generation"]["chunk_size"]
    row_counts = {"orders": 0, "order_items": 0, "payments": 0, "returns": 0}
    next_item_id, next_return_id = 1, 1

    for start in range(0, len(orders), chunk_size):
        chunk = orders.iloc[start:start + chunk_size].copy()
        items = build_order_items(chunk, customers, products, choice_probs, cfg, rng, next_item_id)
        payments = build_payments(chunk, items, cfg, rng)
        chunk["order_status"] = np.where(payments["payment_status"].to_numpy() == "failed", "cancelled", "delivered")
        returns = build_returns(chunk, items, customers, products, cfg, rng, next_return_id)
        next_item_id += len(items)
        next_return_id += len(returns)

        raw_tables = {
            "orders": make_raw_orders(chunk, cfg, injector, len(customers)),
            "order_items": make_raw_order_items(items, injector, len(products)),
            "payments": make_raw_payments(payments, cfg, injector),
            "returns": make_raw_returns(returns, cfg, injector),
        }
        for name, table in raw_tables.items():
            append_csv(table, raw_dir / f"{name}.csv", first_chunk=start == 0)
            row_counts[name] += len(table)
        logger.info("Wrote order chunk %d-%d", start + 1, start + len(chunk))
    return row_counts


def write_customer_truth(customers: pd.DataFrame, churn_month: np.ndarray, calendar: pd.DataFrame, path: Path) -> None:
    """Save hidden customer attributes so analysis results can be checked against the truth."""
    months = pd.period_range(calendar["date"].iloc[0], calendar["date"].iloc[-1], freq="M").astype(str).to_numpy()
    truth = customers[["customer_id", "customer_type", "acquisition_channel", "discount_propensity", "return_propensity"]].copy()
    truth["churn_month"] = np.where(churn_month >= 0, months[np.maximum(churn_month, 0)], None)
    truth.to_csv(path, index=False)


def main() -> None:
    """Generate every raw table plus the two ground-truth files."""
    setup_logging()
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])
    raw_dir = project_path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    calendar = build_calendar(cfg)
    customers = generate_customers(cfg, calendar, rng)
    products = generate_products(cfg, rng)
    categories = generate_categories(cfg)
    orders, churn_month = generate_orders(customers, calendar, cfg, rng)

    injector = IssueInjector(cfg["messiness"], cfg["date_formats"], rng)
    row_counts = {
        "customers": len(customers),
        "products": len(products),
        "categories": len(categories),
    }
    make_raw_customers(customers, cfg, injector).to_csv(raw_dir / "customers.csv", index=False, encoding="utf-8")
    make_raw_products(products, cfg, injector).to_csv(raw_dir / "products.csv", index=False, encoding="utf-8")
    categories.to_csv(raw_dir / "categories.csv", index=False)
    row_counts |= write_order_tables(orders, customers, products, cfg, rng, injector, raw_dir)

    issues = injector.issue_log()
    issues.to_csv(project_path(cfg["paths"]["issue_log"]), index=False, encoding="utf-8")
    write_customer_truth(customers, churn_month, calendar, project_path(cfg["paths"]["customer_truth"]))

    for table, count in row_counts.items():
        logger.info("%-12s %6d rows", table, count)
    logger.info("Injected %d issues; ground truth written to %s", len(issues), cfg["paths"]["issue_log"])


if __name__ == "__main__":
    main()
