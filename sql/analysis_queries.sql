-- Analysis queries for the cleaned, synthetic e-commerce database.
-- Each query starts with a "-- name: <query_name>" line; src/metrics.py loads them by name.
-- Net revenue = quantity x unit price x (1 - order discount), on delivered orders only.
-- Refunds are reported separately, not subtracted from revenue.


-- name: kpi_summary
-- Headline numbers for the whole period.
WITH order_totals AS (
    SELECT
        o.order_id,
        o.customer_id,
        SUM(oi.quantity * oi.unit_price) * (1 - o.discount_pct) AS net_revenue
    FROM orders AS o
    JOIN order_items AS oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY o.order_id, o.customer_id, o.discount_pct
),
refunds AS (
    SELECT SUM(refund_amount) AS total_refunds, COUNT(*) AS returned_lines
    FROM returns
)
SELECT
    ROUND(SUM(ot.net_revenue), 2)                          AS net_revenue_bdt,
    COUNT(*)                                               AS delivered_orders,
    COUNT(DISTINCT ot.customer_id)                         AS buying_customers,
    ROUND(AVG(ot.net_revenue), 2)                          AS avg_order_value_bdt,
    ROUND(r.total_refunds, 2)                              AS refunds_bdt,
    ROUND(r.total_refunds / SUM(ot.net_revenue) * 100, 2)  AS refunds_pct_of_revenue,
    (SELECT ROUND(AVG(order_status = 'cancelled') * 100, 2) FROM orders) AS cancelled_orders_pct
FROM order_totals AS ot
CROSS JOIN refunds AS r;


-- name: monthly_revenue
-- Revenue per month with a running total, month-over-month and year-over-year change.
WITH order_totals AS (
    SELECT
        o.order_id,
        strftime('%Y-%m', o.order_date) AS month,
        SUM(oi.quantity * oi.unit_price) * (1 - o.discount_pct) AS net_revenue
    FROM orders AS o
    JOIN order_items AS oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY o.order_id, month, o.discount_pct
),
monthly AS (
    SELECT month, SUM(net_revenue) AS revenue, COUNT(*) AS orders
    FROM order_totals
    GROUP BY month
)
SELECT
    month,
    ROUND(revenue, 2)                                              AS revenue,
    orders,
    ROUND(revenue / orders, 2)                                     AS avg_order_value,
    ROUND(SUM(revenue) OVER (ORDER BY month), 2)                   AS running_revenue,
    ROUND((revenue / LAG(revenue, 1) OVER (ORDER BY month) - 1) * 100, 1)  AS mom_growth_pct,
    ROUND((revenue / LAG(revenue, 12) OVER (ORDER BY month) - 1) * 100, 1) AS yoy_growth_pct
FROM monthly
ORDER BY month;


-- name: yearly_revenue
-- Revenue, orders, buyers and AOV per year with year-over-year growth.
WITH order_totals AS (
    SELECT
        o.order_id,
        o.customer_id,
        CAST(strftime('%Y', o.order_date) AS INTEGER) AS year,
        SUM(oi.quantity * oi.unit_price) * (1 - o.discount_pct) AS net_revenue
    FROM orders AS o
    JOIN order_items AS oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY o.order_id, o.customer_id, year, o.discount_pct
),
yearly AS (
    SELECT
        year,
        SUM(net_revenue)            AS revenue,
        COUNT(*)                    AS orders,
        COUNT(DISTINCT customer_id) AS buyers
    FROM order_totals
    GROUP BY year
)
SELECT
    year,
    ROUND(revenue, 2)                                                   AS revenue,
    orders,
    buyers,
    ROUND(revenue / orders, 2)                                          AS avg_order_value,
    ROUND((revenue / LAG(revenue) OVER (ORDER BY year) - 1) * 100, 1)   AS yoy_growth_pct
FROM yearly
ORDER BY year;


-- name: category_performance
-- Revenue, units and return rate per category, ranked by revenue.
WITH delivered_lines AS (
    SELECT
        c.category_name,
        oi.quantity,
        oi.quantity * oi.unit_price * (1 - o.discount_pct) AS net_revenue,
        r.return_id,
        r.refund_amount
    FROM order_items AS oi
    JOIN orders AS o      ON o.order_id = oi.order_id
    JOIN products AS p    ON p.product_id = oi.product_id
    JOIN categories AS c  ON c.category_id = p.category_id
    LEFT JOIN returns AS r ON r.order_item_id = oi.order_item_id
    WHERE o.order_status = 'delivered'
),
by_category AS (
    SELECT
        category_name,
        SUM(net_revenue)           AS revenue,
        SUM(quantity)              AS units,
        COUNT(*)                   AS lines,
        COUNT(return_id)           AS returned_lines,
        COALESCE(SUM(refund_amount), 0) AS refunds
    FROM delivered_lines
    GROUP BY category_name
)
SELECT
    RANK() OVER (ORDER BY revenue DESC)                     AS revenue_rank,
    category_name,
    ROUND(revenue, 2)                                       AS revenue,
    ROUND(revenue * 100.0 / SUM(revenue) OVER (), 1)        AS revenue_share_pct,
    units,
    lines,
    ROUND(returned_lines * 100.0 / lines, 1)                AS return_rate_pct,
    ROUND(refunds, 2)                                       AS refunds
FROM by_category
ORDER BY revenue_rank;


-- name: category_rank_by_year
-- Category revenue rank within each year, to see whether the leaders change.
WITH yearly AS (
    SELECT
        CAST(strftime('%Y', o.order_date) AS INTEGER) AS year,
        c.category_name,
        SUM(oi.quantity * oi.unit_price * (1 - o.discount_pct)) AS revenue
    FROM order_items AS oi
    JOIN orders AS o     ON o.order_id = oi.order_id
    JOIN products AS p   ON p.product_id = oi.product_id
    JOIN categories AS c ON c.category_id = p.category_id
    WHERE o.order_status = 'delivered'
    GROUP BY year, c.category_name
)
SELECT
    year,
    category_name,
    ROUND(revenue, 2)                                          AS revenue,
    RANK() OVER (PARTITION BY year ORDER BY revenue DESC)      AS rank_in_year
FROM yearly
ORDER BY year, rank_in_year;


-- name: division_performance
-- Revenue and customer value per customer division.
WITH order_totals AS (
    SELECT
        o.order_id,
        o.customer_id,
        SUM(oi.quantity * oi.unit_price) * (1 - o.discount_pct) AS net_revenue
    FROM orders AS o
    JOIN order_items AS oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY o.order_id, o.customer_id, o.discount_pct
)
SELECT
    c.division,
    COUNT(DISTINCT c.customer_id)                                   AS buyers,
    COUNT(ot.order_id)                                              AS orders,
    ROUND(SUM(ot.net_revenue), 2)                                   AS revenue,
    ROUND(SUM(ot.net_revenue) * 100.0 / SUM(SUM(ot.net_revenue)) OVER (), 1) AS revenue_share_pct,
    ROUND(SUM(ot.net_revenue) / COUNT(ot.order_id), 2)              AS avg_order_value,
    ROUND(SUM(ot.net_revenue) / COUNT(DISTINCT c.customer_id), 2)   AS revenue_per_buyer
FROM order_totals AS ot
JOIN customers AS c ON c.customer_id = ot.customer_id
GROUP BY c.division
ORDER BY revenue DESC;


-- name: weekday_hour_orders
-- Delivered and cancelled order counts by weekday (0 = Sunday in SQLite) and hour.
SELECT
    CAST(strftime('%w', order_date) AS INTEGER) AS weekday,
    CAST(strftime('%H', order_date) AS INTEGER) AS hour,
    COUNT(*)                                    AS orders
FROM orders
GROUP BY weekday, hour
ORDER BY weekday, hour;


-- name: payment_method_mix
-- Share of orders and failure rate by payment method.
SELECT
    payment_method,
    COUNT(*)                                                    AS orders,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)          AS order_share_pct,
    ROUND(AVG(payment_status = 'failed') * 100, 2)              AS failure_rate_pct,
    ROUND(SUM(CASE WHEN payment_status = 'completed' THEN amount END), 2) AS collected_bdt
FROM payments
GROUP BY payment_method
ORDER BY orders DESC;


-- name: order_revenue
-- One row per delivered order, numbered within each customer, with the gap since the previous order.
WITH order_totals AS (
    SELECT
        o.order_id,
        o.customer_id,
        o.order_date,
        o.discount_pct,
        SUM(oi.quantity * oi.unit_price) * (1 - o.discount_pct) AS net_revenue
    FROM orders AS o
    JOIN order_items AS oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY o.order_id, o.customer_id, o.order_date, o.discount_pct
)
SELECT
    order_id,
    customer_id,
    order_date,
    discount_pct,
    ROUND(net_revenue, 2) AS net_revenue,
    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date, order_id) AS order_number,
    ROUND(julianday(order_date)
          - julianday(LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date, order_id)), 1)
          AS days_since_previous_order
FROM order_totals
ORDER BY customer_id, order_number;


-- name: customer_summary
-- One row per buying customer: profile, order history, discount dependence and return rate.
WITH order_totals AS (
    SELECT
        o.order_id,
        o.customer_id,
        o.order_date,
        o.discount_pct,
        SUM(oi.quantity * oi.unit_price) * (1 - o.discount_pct) AS net_revenue
    FROM orders AS o
    JOIN order_items AS oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY o.order_id, o.customer_id, o.order_date, o.discount_pct
),
customer_orders AS (
    SELECT
        customer_id,
        COUNT(*)                                             AS orders,
        SUM(net_revenue)                                     AS revenue,
        MIN(order_date)                                      AS first_order_date,
        MAX(order_date)                                      AS last_order_date,
        AVG(CASE WHEN discount_pct > 0 THEN 1.0 ELSE 0.0 END) AS discount_share
    FROM order_totals
    GROUP BY customer_id
),
customer_returns AS (
    SELECT
        o.customer_id,
        COUNT(*)           AS delivered_lines,
        COUNT(r.return_id) AS returned_lines
    FROM order_items AS oi
    JOIN orders AS o       ON o.order_id = oi.order_id
    LEFT JOIN returns AS r ON r.order_item_id = oi.order_item_id
    WHERE o.order_status = 'delivered'
    GROUP BY o.customer_id
)
SELECT
    c.customer_id,
    c.gender,
    c.division,
    c.district,
    c.acquisition_channel,
    c.signup_date,
    co.orders,
    ROUND(co.revenue, 2)                                     AS revenue,
    co.first_order_date,
    co.last_order_date,
    ROUND(co.discount_share, 3)                              AS discount_share,
    ROUND(cr.returned_lines * 1.0 / cr.delivered_lines, 3)   AS return_rate
FROM customers AS c
JOIN customer_orders AS co  ON co.customer_id = c.customer_id
JOIN customer_returns AS cr ON cr.customer_id = c.customer_id
ORDER BY c.customer_id;


-- name: customer_quarter_activity
-- One row per customer and quarter in which they placed a delivered order, with their
-- first-order quarter (cohort) and how many quarters after it the activity happened.
WITH quarterly AS (
    SELECT DISTINCT
        o.customer_id,
        CAST(strftime('%Y', o.order_date) AS INTEGER) * 4
            + (CAST(strftime('%m', o.order_date) AS INTEGER) - 1) / 3 AS quarter_index
    FROM orders AS o
    WHERE o.order_status = 'delivered'
),
with_cohort AS (
    SELECT
        customer_id,
        quarter_index,
        MIN(quarter_index) OVER (PARTITION BY customer_id) AS cohort_index
    FROM quarterly
)
SELECT
    w.customer_id,
    c.acquisition_channel,
    w.cohort_index,
    (w.cohort_index / 4) || '-Q' || (w.cohort_index % 4 + 1) AS cohort_quarter,
    w.quarter_index - w.cohort_index                         AS quarters_since_first
FROM with_cohort AS w
JOIN customers AS c ON c.customer_id = w.customer_id
ORDER BY w.customer_id, quarters_since_first;
