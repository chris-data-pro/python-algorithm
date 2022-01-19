--SQL Server: returns any distinct values from the query left of EXCEPT that aren't found on the right query.
SELECT ProductID
FROM Production.Product
EXCEPT
SELECT ProductID
FROM Production.WorkOrder ;


--Oracle: the same
SELECT ProductID
FROM Production.Product
MINUS
SELECT ProductID
FROM Production.WorkOrder ;


--Both can be done by left join
SELECT l.ProductID
FROM Production.Product p
LEFT JOIN Production.WorkOrder w
ON p.ProductID = w.ProductID
WHERE w.ProductID IS NULL;


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


-- 2 get products that was ordered on '02-JUL-2011' but not on '01-JUL-2011'
SELECT product_id from development.product_sales where order_day = '02-JUL-2011'
MINUS
SELECT product_id from development.product_sales where order_day = '01-JUL-2011'


SELECT * FROM
(select product_id, lag(order_day, 1) over (partition by product_id order by order_day) as lag_day, order_day
from development.product_sales)
where lag_day ISNULL and order_day = '02-JUL-2011'


SELECT * --t.product_id
from (select product_id, order_day from development.product_sales where order_day = '02-JUL-2011') t  --left join时左边的table提前filter好
left join (select product_id, order_day from development.product_sales where order_day = '01-JUL-2011') o
on t.product_id = o.product_id
where o.product_id IS NULL


select product_id from (
select product_id, order_day, order_id, dense_rank() over (partition by product_id order by order_day) as rk
from development.product_sales
order by 1,2,3) foo
where rk = 1 and order_day = '02-JUL-2011';


-- 3 get the highest sold $ amount products () on these 2 days
select order_day, product_id, sale_amount
from
(select order_day, product_id, sum(quantity * price) as sale_amount, row_number() over (partition by order_day order by sale_amount desc) as rk
from development.product_sales
group by 1, 2) foo
where rk = 1
order by 1;

with day_rank AS(
SELECT order_day, product_id, SUM(quantity * price) as sale_amount, rank() over (partition by order_day order by sale_amount DESC) as rk
FROM development.product_sales
GROUP BY 1, 2)
select order_day, product_id, sale_amount
FROM day_rank
where rk = 1


--order_day  product_id  sale_amount
--02-JUL-2011     P3       125
--01-JUL-2011     P3       250



-- 4 get all products and the total sales on day 1 and day 2
SELECT product_id,
       SUM(CASE WHEN order_day = '01-JUL-2011' THEN quantity * price ELSE 0 END) as tot_sales_01,
       SUM(CASE WHEN order_day = '02-JUL-2011' THEN quantity * price ELSE 0 END) as tot_sales_02
FROM development.product_sales
GROUP BY 1


select
nvl(d1.product_id, d2.product_id) as product_id,
nvl(d1.tot_sales_01, 0) as tot_sales_01,
nvl(d2.tot_sales_02, 0) as tot_sales_02
from
(select product_id, sum(quantity * price) as tot_sales_01
from development.product_sales
where order_day = '01-JUL-2011'
group by 1) d1 full outer join
(select product_id, sum(quantity * price) as tot_sales_02
from development.product_sales
where order_day = '02-JUL-2011'
group by 1) d2 on d1.product_id = d2.product_id
order by 1;

--product_id  tot_sales_01  tot_sales_02
-- P1            125             10
-- P2            20              40
-- P3            250             125
-- P4            0               120
-- P5            0               50
-- P6            0               100


