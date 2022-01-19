--given 2 tables parents and children,
--give me all parents that have at least one male child and one female child

begin;
CREATE TABLE development.children (
   child_id int,
   child_name varchar(50),
   Child_Gender varchar(50),
   parent_id int
);

INSERT INTO development.children VALUES
(1, 'David','Male', 6),
(2, 'Emma','Male', 4),
(3, 'Alice','Female', 3),
(4, 'David','Male', 3),
(5, 'Tom','Female', 1),
(6, 'Nathen','Male', 2),
(7, 'Akira','Male', 5),
(8, 'Kate','Male', 1),
(9, 'Lily','Female', 3);
commit;


begin;
CREATE TABLE development.parents (
   parent_id int,
   parent_name varchar(50)
);

INSERT INTO development.parents VALUES
(1, 'p1'),
(2, 'p2'),
(3, 'p3'),
(5, 'p5'),
(6, 'p6');
commit;


select * from development.children;
select * from development.parents;







--The SQL NATURAL JOIN is a type of JOIN, it's like INNER JOIN
--It is structured in such a way that, columns with same name of associate tables will appear once only.
--The associated tables have one or more pairs of identically named columns.
--The columns must be the same data type.
--Don’t use ON clause in a natural join.
select * from development.children NATURAL JOIN development.parents order by 1;






select parent_id,
       SUM(CASE WHEN child_gender='Male' THEN 1 ELSE 0 END) as male_ct,
       SUM(CASE WHEN child_gender='Female' THEN 1 ELSE 0 END) as female_ct
from development.children
group by 1
having female_ct >= 1 and male_ct >= 1
order by 1;

select parent_name
from development.children c join development.parents p
on c.parent_id = p.parent_id
group by 1
having SUM(CASE WHEN child_gender='Male' THEN 1 ELSE 0 END) >= 1
and SUM(CASE WHEN child_gender='Female' THEN 1 ELSE 0 END) >= 1;


select parent_name
from development.children c, development.parents p
where c.parent_id = p.parent_id
group by 1
having SUM(CASE WHEN child_gender='Male' THEN 1 ELSE 0 END) >= 1
and SUM(CASE WHEN child_gender='Female' THEN 1 ELSE 0 END) >= 1;


select *
from development.parents p
join (
 select distinct parent_id from development.children where child_gender = 'Male'
) a on p.parent_id = a.parent_id
join (
 select distinct parent_id from development.children where child_gender = 'Female'
) b on p.parent_id = b.parent_id;

select *
from development.parents p
join development.children a on p.parent_id = a.parent_id
join development.children b on p.parent_id = b.parent_id
where a.child_gender = 'Male' and b.child_gender = 'Female'

select parent_id from development.children where child_gender = 'Male'
INTERSECT
select parent_id from development.children where child_gender = 'Female'



select parent_id, count(distinct child_id) as mct
from development.children
where child_gender = 'Male'
group by 1
having count(distinct child_id) > 0

begin;
drop table IF EXISTS development.parents;
commit;


---------------------------------------------------------------------------------
--Summarize each group of tests.

--The table of results should contain four columns:
--name (name of the group),
--all_test_cases (number of tests in the group),
--passed_test_cases (number of test cases with the status OK),
--and total_value (total value of passed tests in this group).

--Rows should be ordered by decreasing total_value.
--In the case of a tie, rows should be sorted lexicographically by name.
---------------------------------------------------------------------------------
begin;
CREATE TABLE development.test_groups (
    name          VARCHAR(40)   NOT NULL,
    test_value    INT   NOT NULL,
    unique(name)
);
commit;

begin;
INSERT INTO development.test_groups VALUES
('performance', 15),
('corner cases', 10),
('numerical stability', 20),
('memory usage', 10);
commit;


begin;
CREATE TABLE development.test_cases (
    id  INTEGER   NOT NULL,
    group_name    VARCHAR(40)   NOT NULL,
    status        VARCHAR(5)    NOT NULL,
    unique(id)
);
commit;

begin;
INSERT INTO development.test_cases VALUES
(13, 'memory usage', 'OK'),
(14, 'numerical stability', 'OK'),
(15, 'memory usage', 'ERROR'),
(16, 'numerical stability', 'OK'),
(17, 'numerical stability', 'OK'),
(18, 'performance', 'ERROR'),
(19, 'performance', 'ERROR'),
(20, 'memory usage', 'OK'),
(21, 'numerical stability', 'OK');
commit;


SELECT tg.name,
       COUNT(tc.id) as all_test_cases,
       SUM(CASE WHEN tc.status = 'OK' THEN 1 ELSE 0 END) as passed_test_cases,
       SUM(CASE WHEN tc.status = 'OK' THEN tg.test_value ELSE 0 END) as total_value
FROM development.test_groups tg
LEFT JOIN development.test_cases tc
ON tg.name = tc.group_name
GROUP BY 1
ORDER BY 4 DESC, 1;





begin;
drop table IF EXISTS development.test_groups;
commit;




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


-- sum without group by
--find orders with quantity, plus the total quantity sold for the day
select
order_day,
order_id,
quantity,
sum(quantity) over (partition by order_day) as total_quantity
from development.product_sales
order by 1;

--01-JUL-2011	O1	5	37
--01-JUL-2011	O2	2	37
--01-JUL-2011	O3	10	37
--01-JUL-2011	O4	20	37
--02-JUL-2011	O10	4	20
--02-JUL-2011	O5	5	20
--02-JUL-2011	O6	6	20
--02-JUL-2011	O7	2	20
--02-JUL-2011	O8	1	20
--02-JUL-2011	O9	2	20


--with group by
--get all products that got sold both the days, and the number of times sold
SELECT product_id, COUNT(order_id) as time_sold
FROM development.product_sales
GROUP BY 1
HAVING COUNT(DISTINCT order_day) > 1

--or

select product_id, count(distinct order_id) as time_sold
from development.product_sales
where product_id in (
select distinct d1.product_id --, d1.order_id as day1order, d1.order_day as day1, d2.order_id as day2order, d2.order_day as day2
from development.product_sales d1
join development.product_sales d2
on d1.product_id = d2.product_id
and d1.order_day = '01-JUL-2011'
and d2.order_day = '02-JUL-2011'
)
group by 1;


--product_id  time_sold
--P1              3
--P2              2
--P3              2


-- 3) get a list of customers (cust_id) who have placed less than 2 orders or have ordered for less than $100
begin;
DROP TABLE IF EXISTS development.orders;
commit;

begin;
create table development.orders (
  order_id                        int,
  cust_id                         int,
  order_date                      date,
  product                         VARCHAR(20),
  order_amount                    int

);
commit;

SELECT * FROM development.orders;

begin;
INSERT INTO development.orders
(order_id, cust_id, order_date, product, order_amount) VALUES
(50001, 101, date('2015-08-29'), 'Camera', 100),
(50002, 102, date('2015-08-30'), 'Shoes', 90),
(50003, 103, date('2015-05-31'), 'Laptop', 400),
(50004, 101, date('2015-08-29'), 'Mobile', 100),
(50005, 104, date('2015-08-29'), 'FrozenMeals', 30),
(50006, 104, date('2015-08-30'), 'Cloths', 65);
commit;


--select foo.cust_id
--FROM
--(
--select cust_id, count(distinct order_id) as no_orders, sum(order_amount) as total_amt
--from development.orders
--group by 1
--HAVING count(distinct order_id) < 2 or sum(order_amount) < 100
--) foo
--order by 1;


select cust_id
from development.orders
group by 1
HAVING count(distinct order_id) < 2 or sum(order_amount) < 100
order by 1;


