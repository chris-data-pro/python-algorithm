-- 2) identify customers (customer_id) who placed more than 3 orders in both year 2014 and year 2015.
begin;
create table development.cust_orders (
  order_id                        int,
  customer_id                     int,
  order_date                      date
);
commit;

SELECT * FROM development.cust_orders;

begin;
INSERT INTO development.cust_orders
(order_id, customer_id, order_date) VALUES
(101, 201, date('2014-01-01')),
(102, 202, date('2014-02-12')),
(103, 203, date('2014-03-05')),
(104, 204, date('2014-04-15')),
(105, 202, date('2014-05-15')),
(106, 202, date('2014-07-10')),
(107, 203, date('2014-09-15')),
(108, 202, date('2014-12-15')),
(109, 201, date('2015-01-01')),
(110, 202, date('2015-02-12')),
(111, 202, date('2015-03-05')),
(112, 201, date('2015-04-15')),
(113, 202, date('2015-05-15')),
(114, 202, date('2015-07-10')),
(115, 205, date('2015-09-15'));
commit;


SELECT customer_id
FROM development.cust_orders
WHERE DATE_PART(y, order_date) = '2014'
GROUP BY 1
HAVING COUNT(*) > 3
INTERSECT
SELECT customer_id
FROM development.cust_orders
WHERE DATE_PART(y, order_date) = '2015'
GROUP BY 1
HAVING COUNT(*) > 3


SELECT customer_id,
       SUM(CASE WHEN DATE_PART(y, order_date) = '2014' THEN 1 ELSE 0 END) AS ct2014,
       SUM(CASE WHEN DATE_PART(y, order_date) = '2015' THEN 1 ELSE 0 END) AS ct2015
FROM development.cust_orders
GROUP BY 1
HAVING ct2014 > 3 AND ct2015 > 3


select a.customer_id
FROM
(
select DATE_PART(y, order_date) as year, customer_id, count(distinct order_id) as cnt
from development.cust_orders
where DATE_PART(y, order_date) = '2014'
group by 1, 2
HAVING count(distinct order_id) > 3
) a
inner join
(
select DATE_PART(y, order_date) as year, customer_id, count(distinct order_id) as cnt
from development.cust_orders
where DATE_PART(y, order_date) = '2015'
group by 1, 2
HAVING count(distinct order_id) > 3
) b
on a.customer_id = b.customer_id
order by 1;