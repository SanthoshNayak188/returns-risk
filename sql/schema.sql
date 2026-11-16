CREATE TABLE IF NOT EXISTS dim_customer (
  customer_key INTEGER PRIMARY KEY, customer_id TEXT UNIQUE, country TEXT, is_unknown_member INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS dim_product (
  product_key INTEGER PRIMARY KEY, stock_code TEXT UNIQUE, description_clean TEXT, is_product_code INTEGER);
CREATE TABLE IF NOT EXISTS dim_date (
  date_key INTEGER PRIMARY KEY, full_date TEXT UNIQUE, day_of_week INTEGER, month INTEGER, year INTEGER, is_weekend INTEGER);
CREATE TABLE IF NOT EXISTS fact_sales_line (
  line_key INTEGER PRIMARY KEY, source_row INTEGER UNIQUE, invoice_no TEXT, invoice_ts TEXT,
  date_key INTEGER REFERENCES dim_date, customer_key INTEGER REFERENCES dim_customer, product_key INTEGER REFERENCES dim_product,
  quantity INTEGER, unit_price REAL, line_amount REAL, n_lines_in_invoice INTEGER, invoice_total REAL);
CREATE TABLE IF NOT EXISTS fact_return_line (
  credit_line_key INTEGER PRIMARY KEY, source_row INTEGER UNIQUE, credit_invoice_no TEXT, credit_ts TEXT,
  date_key INTEGER REFERENCES dim_date, customer_key INTEGER REFERENCES dim_customer, product_key INTEGER REFERENCES dim_product,
  quantity_returned INTEGER);
CREATE TABLE IF NOT EXISTS bridge_return_match (
  sales_line_key INTEGER REFERENCES fact_sales_line, credit_line_key INTEGER REFERENCES fact_return_line,
  matched_qty INTEGER, days_to_return INTEGER);
CREATE TABLE IF NOT EXISTS etl_batch_log (
  batch_id INTEGER PRIMARY KEY, load_date TEXT, rows_in INTEGER, rows_loaded INTEGER, rows_rejected INTEGER, status TEXT);
CREATE TABLE IF NOT EXISTS recon_results (
  batch_id INTEGER REFERENCES etl_batch_log, check_name TEXT, source_value REAL, warehouse_value REAL, passed INTEGER);
CREATE TABLE IF NOT EXISTS reject_rows (source_row INTEGER PRIMARY KEY, day TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS stg_files (name TEXT, sha256 TEXT, rows INTEGER);
CREATE INDEX IF NOT EXISTS ix_bridge_sale ON bridge_return_match(sales_line_key);
CREATE INDEX IF NOT EXISTS ix_bridge_credit ON bridge_return_match(credit_line_key);
CREATE INDEX IF NOT EXISTS ix_sales_customer ON fact_sales_line(customer_key, product_key);
