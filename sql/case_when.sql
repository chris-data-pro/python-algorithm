begin;
DROP TABLE IF EXISTS development.product_sales;
commit;

begin;
create table development.product_sales (
	order_day                        char varying(50)
	, order_id                       char varying(5)
	, product_id                     char varying(5)
	, quantity                       int
	, price                          int
);
commit;

SELECT * FROM development.product_sales;

begin;
INSERT INTO development.product_sales
(order_day, order_id, product_id, quantity, price) VALUES
('01-JUL-2011', 'O1', 'P1', 5, 5),
('01-JUL-2011', 'O2', 'P2', 2, 10),
('01-JUL-2011', 'O3', 'P3', 10, 25),
('01-JUL-2011', 'O4', 'P1', 20, 5),
('02-JUL-2011', 'O5', 'P3', 5, 25),
('02-JUL-2011', 'O6', 'P4', 6, 20),
('02-JUL-2011', 'O7', 'P1', 2, 5),
('02-JUL-2011', 'O8', 'P5', 1, 50),
('02-JUL-2011', 'O9', 'P6', 2, 50),
('02-JUL-2011', 'O10', 'P2', 4, 10);
commit;
-- order fact table


-- 4 get all products and the total sales on day 1 and day 2
SELECT product_id,
       SUM(CASE WHEN order_day = '01-JUL-2011' THEN quantity * price ELSE 0 END) as tot_sales_01,
       SUM(CASE WHEN order_day = '02-JUL-2011' THEN quantity * price ELSE 0 END) as tot_sales_02
FROM development.product_sales
GROUP BY 1;


-- 5 get all order_day product_id vis, that was ordered more than once
select order_day, product_id
from development.product_sales
group by 1, 2
having count(*) > 1  -- This is an order table, count(*) is same as count(DISTINCT order_id)




