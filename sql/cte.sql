-- common table expression
begin;
DROP TABLE IF EXISTS development.order_table;
commit;

begin;
create table development.order_table (
	order_id                   char varying(5)
	, item                     char varying(5)
	, qty                      int
);
commit;

SELECT * FROM development.order_table;

begin;
INSERT INTO development.order_table
(order_id, item, qty) VALUES
('O1', 'A1', 5),
('O2', 'A2', 1),
('O3', 'A3', 3);
commit;

WITH RECURSIVE RCTE (ORDER_ID, ITEM, QTY, EXPLODEROWS) AS
( SELECT ORDER_ID, ITEM, QTY, 1 FROM development.order_table  -- EXPLODEROWS 1只是initialize第一行
  UNION ALL
  SELECT ORDER_ID, ITEM, QTY, EXPLODEROWS + 1 FROM RCTE WHERE EXPLODEROWS < QTY)  -- 第二行：1+1=2 where 1 < 5, ... 第五行：4+1=5 where 4 < 5
SELECT *
FROM RCTE
ORDER BY ORDER_ID;

--ORDER_ID ITEM QTY EXPLODEROWS
--O1        A1   5         1
--O1        A1   5         2
--O1        A1   5         3
--O1        A1   5         4
--O1        A1   5         5
--O2        A2   1         1
--O3        A3   3         1
--O3        A3   3         2
--O3        A3   3         3


WITH RECURSIVE RCTE (ORDER_ID,ITEM,QTY,EXPLODEROWS) AS
( SELECT ORDER_ID, ITEM, QTY, 1 FROM development.order_table  -- EXPLODEROWS 1只是initialize第一行
  UNION ALL
  SELECT ORDER_ID, ITEM, QTY, EXPLODEROWS + 1 FROM RCTE WHERE EXPLODEROWS < QTY)  --O2没有第二行：1+1=2 where 1 < 1不成立
SELECT ORDER_ID , ITEM , 1 AS CNT
FROM RCTE
ORDER BY ORDER_ID;

--ORDER_ID ITEM CNT
--O1        A1   1
--O1        A1   1
--O1        A1   1
--O1        A1   1
--O1        A1   1
--O2        A2   1
--O3        A3   1
--O3        A3   1
--O3        A3   1


-- common table expression
WITH RECURSIVE cte (n) AS
(
  SELECT 1
  UNION ALL
  SELECT n + 1 FROM cte WHERE n < 5  -- limit 4 最大到4
)
SELECT * FROM cte;

--n
--1
--2
--3
--4
--5


-- x是从0开始的fibonacci数列，y是从1开始的fibonacci数列
with recursive base (x, y) as
(
select 0, 1
union all
select y, x + y from base where y < 20
)
select * from base;


--求 从0开始的fibonacci数列 里的第50个数
WITH RECURSIVE fibonacci (n, fib_n, next_fib_n) AS
(
  SELECT cast(1 as int), cast(0 as BIGINT), cast(1 as BIGINT)
  UNION ALL
  SELECT n + 1, next_fib_n, fib_n + next_fib_n
    FROM fibonacci WHERE n < 50
)
SELECT fib_n FROM fibonacci where n = 50;



WITH RECURSIVE cte (n, str) AS
(
  SELECT 1 AS n, cast('abc' as char(20)) AS str
  UNION ALL
  SELECT n + 1, concat(str, str) FROM cte WHERE n < 3
)
SELECT * FROM cte;

--n  str
--1  abc
--2  abcabc
--3  abcabcabcabc


WITH RECURSIVE cte (n, p, q) AS
(
  SELECT 1 AS n, 1 AS p, -1 AS q
  UNION ALL
  SELECT n + 1, q * 2, p * 2 FROM cte WHERE n < 5  -- 每次recursive都从上一行取
)
SELECT * FROM cte;

--n   p  q
--1   1  -1
--2  -2  2
--3   4  -4
--4  -8  8
--5  16  -16


begin;
DROP TABLE IF EXISTS development.sales_table;
commit;

begin;
create table development.sales_table (
	sale_date                   date
	, price                     float
);
commit;

SELECT * FROM development.sales_table;

begin;
INSERT INTO development.sales_table
(sale_date, price) VALUES
(date('2017-01-03'), 100.0),
(date('2017-01-03'), 200.0),
(date('2017-01-06'), 50.0),
(date('2017-01-08'), 10.0),
(date('2017-01-08'), 20.0),
(date('2017-01-08'), 150.0),
(date('2017-01-10'), 5.0),
(date('2017-01-10'), 25.0);
commit;


with recursive base (sale_date) as (
select min(sale_date) from development.sales_table
union all
select date(sale_date + interval '1 day') from base where sale_date < (select max(sale_date) from development.sales_table)
)
select b.sale_date, isnull(sum(st.price), 0) as sum_price
from base b left join development.sales_table st on b.sale_date = st.sale_date
group by 1
order by 1;

--sale_date   sum_price
--2017-01-03    300.0
--2017-01-04    0.0
--2017-01-05    0.0
--2017-01-06    50.0
--2017-01-07    0.0
--2017-01-08    180.0
--2017-01-09    0.0
--2017-01-10    30.0


begin;
DROP TABLE IF EXISTS development.employees;
commit;

begin;
create table development.employees (
  id         INT PRIMARY KEY NOT NULL,
  name       VARCHAR(100) NOT NULL,
  manager_id INT NULL
);
commit;

SELECT * FROM development.employees;

begin;
INSERT INTO development.employees
(id, name, manager_id) VALUES
(333, 'Yasmina', NULL),
(198, 'John', 333),
(692, 'Tarek', 333),
(29, 'Pedro', 198),
(4610, 'Sarah', 29),
(72, 'Pierre', 29),
(123, 'Adil', 692);
commit;


WITH RECURSIVE employee_paths (id, name, path) AS
(
  SELECT id, name, CAST(id AS CHAR(200))
    FROM development.employees
    WHERE manager_id IS NULL
  UNION ALL
  SELECT e.id, e.name, ep.path||','||e.id
    FROM employee_paths AS ep JOIN development.employees AS e
      ON ep.id = e.manager_id
)
SELECT * FROM employee_paths;


--id     name       path
--333    Yasmina    333
--692    Tarek      333,692
--198    John       333,198
--123    Adil       333,692,123
--29     Pedro      333,198,29
--72     Pierre     333,198,29,72
--4610   Sarah      333,198,29,4610


-- 如果只看一层的team
-- 1) identify managers (name, not id) with the biggest team size
begin;
DROP TABLE IF EXISTS development.employee;
commit;

begin;
create table development.employee (
  emp_id         INT PRIMARY KEY NOT NULL,
  emp_name       VARCHAR(100) NOT NULL,
  manager_id INT NULL
);
commit;

SELECT * FROM development.employee;

begin;
INSERT INTO development.employee
(emp_id, emp_name, manager_id) VALUES
(101, 'John', 104),
(102, 'Mary', 104),
(103, 'Smith', 104),
(104, 'Bill', 105),
(105, 'Kelly', 106),
(106, 'Will', null);
commit;


select m.emp_id, m.emp_name, count(*) as ct
from development.employee m
join development.employee e
  on m.emp_id = e.manager_id
group by 1, 2
order by ct desc
limit 1


--or
select a.emp_name
from
(
SELECT foo.emp_name, rank() over (order by ct desc) as rk
from
(select a.emp_name, count(distinct b.emp_id) as ct
from development.employee a
join development.employee b
  on a.emp_id = b.manager_id
group by 1) foo
) a
where a.rk = 1;

