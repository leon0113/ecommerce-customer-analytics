-- Schema for the cleaned, synthetic e-commerce dataset.
-- Dates are stored as ISO-8601 text ('YYYY-MM-DD HH:MM:SS'), the SQLite convention.
-- Tables are dropped child-first and created parent-first so foreign keys always resolve.

DROP TABLE IF EXISTS returns;
DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS categories;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id         INTEGER PRIMARY KEY,
    full_name           TEXT    NOT NULL,
    gender              TEXT    NOT NULL CHECK (gender IN ('Male', 'Female', 'Unknown')),
    phone               TEXT,
    division            TEXT    NOT NULL,
    district            TEXT    NOT NULL,
    acquisition_channel TEXT    NOT NULL,
    signup_date         TEXT    NOT NULL
);

CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT    NOT NULL UNIQUE
);

CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    product_name TEXT    NOT NULL,
    category_id  INTEGER NOT NULL REFERENCES categories (category_id),
    brand        TEXT    NOT NULL,
    list_price   REAL    NOT NULL CHECK (list_price > 0)
);

CREATE TABLE orders (
    order_id          INTEGER PRIMARY KEY,
    customer_id       INTEGER NOT NULL REFERENCES customers (customer_id),
    order_date        TEXT    NOT NULL,
    order_status      TEXT    NOT NULL CHECK (order_status IN ('delivered', 'cancelled')),
    discount_pct      REAL    NOT NULL CHECK (discount_pct >= 0 AND discount_pct < 1),
    delivery_district TEXT    NOT NULL
);

CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders (order_id),
    product_id    INTEGER NOT NULL REFERENCES products (product_id),
    quantity      INTEGER NOT NULL CHECK (quantity > 0),
    unit_price    REAL    NOT NULL CHECK (unit_price > 0)
);

CREATE TABLE payments (
    payment_id     INTEGER PRIMARY KEY,
    order_id       INTEGER NOT NULL UNIQUE REFERENCES orders (order_id),
    payment_method TEXT    NOT NULL,
    amount         REAL    NOT NULL CHECK (amount >= 0),
    payment_status TEXT    NOT NULL CHECK (payment_status IN ('completed', 'failed')),
    payment_date   TEXT    NOT NULL
);

CREATE TABLE returns (
    return_id     INTEGER PRIMARY KEY,
    order_item_id INTEGER NOT NULL UNIQUE REFERENCES order_items (order_item_id),
    return_date   TEXT    NOT NULL,
    return_reason TEXT    NOT NULL,
    refund_amount REAL    NOT NULL CHECK (refund_amount >= 0)
);

-- Indexes on the columns used for joins and date filtering.
CREATE INDEX idx_orders_customer   ON orders (customer_id);
CREATE INDEX idx_orders_date       ON orders (order_date);
CREATE INDEX idx_items_order       ON order_items (order_id);
CREATE INDEX idx_items_product     ON order_items (product_id);
CREATE INDEX idx_products_category ON products (category_id);
