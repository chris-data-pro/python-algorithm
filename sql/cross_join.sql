-- CROSS JOIN 2 TABLES: 左边每一行对应右边所有行

with col_name_table as (
-- SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'development' AND TABLE_NAME = 'input_table'
 SELECT column_name FROM development.col_name_table
)
SELECT column_name AS Page, COUNT(*) AS counts
FROM development.input_table
CROSS JOIN col_name_table
GROUP BY 1;

-- output
Page            counts
Product_Page        8
Warranty_Page       8
Home_Page           8


-- this is a cross join and it's inefficient
select * from development.children, development.parents;


SELECT Customers.CustomerID, Customers.Name, Sales.LastSaleDate
FROM Customers, Sales
WHERE Customers.CustomerID = Sales.CustomerID


-- instead, use inner join
SELECT Customers.CustomerID, Customers.Name, Sales.LastSaleDate
FROM Customers
   INNER JOIN Sales
   ON Customers.CustomerID = Sales.CustomerID
