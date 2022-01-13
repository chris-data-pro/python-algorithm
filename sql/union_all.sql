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

-- Use UNION ALL instead of UNION
SELECT 'Home_Page' AS Page, SUM(Home_Page) AS sums, COUNT(*) AS counts
FROM development.input_table
UNION ALL
SELECT 'Product_Page' AS Page, SUM(Product_Page) AS sums, COUNT(*) AS counts
FROM development.input_table
UNION ALL
SELECT 'Warranty_Page' AS Page, SUM(Warranty_Page) AS sums, COUNT(*) AS counts
FROM development.input_table;

-- output table
Page                  sums        counts
Product_Page            4           8
Home_Page               4           8
Warranty_Page           4           8

