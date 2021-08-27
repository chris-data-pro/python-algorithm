-- V1 If I can use the column names:
SELECT 'Home_Page' AS Page, SUM(Home_Page) AS sums, COUNT(*) AS counts
FROM input_table
UNION
SELECT 'Product_Page' AS Page, SUM(Product_Page) AS sums, COUNT(*) AS counts
FROM input_table
UNION
SELECT 'Warranty_Page' AS Page, SUM(Warranty_Page) AS sums, COUNT(*) AS counts
FROM input_table;


-- V2 If I don't use the column names (dynamic):
with col_name_table as (
 SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'input_table'
)
SELECT column_name AS Page, COUNT(*) AS counts FROM input_table
CROSS JOIN col_name_table
GROUP BY 1;


-- Test:
-- Create development.input_table
begin;
DROP TABLE IF EXISTS development.input_table;
commit;

begin;
create table development.input_table (
	Home_Page                        int
	, Product_Page                   int
	, Warranty_Page                  int
);
commit;

begin;
INSERT INTO development.input_table
(Home_Page, Product_Page, Warranty_Page) VALUES
(1, 1, 1),
(1, 1, 0),
(1, 0, 1),
(1, 0, 0),
(0, 1, 1),
(0, 1, 0),
(0, 0, 1),
(0, 0, 0);
commit;

--SELECT * FROM development.input_table;

-- Create development.input_table
-- because redshift doesn't support mix of INFORMATION_SCHEMA and normal table selects
-- This generates an equivalent table as
-- SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'development' AND TABLE_NAME = 'input_table'
begin;
DROP TABLE IF EXISTS development.col_name_table;
commit;

begin;
create table development.col_name_table (
	column_name                      char varying(50)
);
commit;

begin;
INSERT INTO development.col_name_table
(column_name) VALUES
('Home_Page'),
('Product_Page'),
('Warranty_Page');
commit;

SELECT * FROM development.col_name_table;


-- Test V1
SELECT 'Home_Page' AS Page, SUM(Home_Page) AS sums, COUNT(*) AS counts
FROM development.input_table
UNION
SELECT 'Product_Page' AS Page, SUM(Product_Page) AS sums, COUNT(*) AS counts
FROM development.input_table
UNION
SELECT 'Warranty_Page' AS Page, SUM(Warranty_Page) AS sums, COUNT(*) AS counts
FROM development.input_table;

-- output table
Page                  sums        counts
Product_Page            4           8
Home_Page               4           8
Warranty_Page           4           8


-- Test V2
with col_name_table as (
-- SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'development' AND TABLE_NAME = 'input_table'
 SELECT column_name FROM development.col_name_table
)
SELECT column_name AS Page, COUNT(*) AS counts FROM development.input_table
CROSS JOIN col_name_table
GROUP BY 1;

-- output table
Page            counts
Product_Page        8
Warranty_Page       8
Home_Page           8