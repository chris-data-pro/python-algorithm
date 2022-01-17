-- common table expression
SELECT * FROM development.order_table;

begin;
INSERT INTO development.order_table
(order_id, item, qty) VALUES
('O1', 'A1', 5),
('O2', 'A2', 1),
('O3', 'A3', 3);
commit;


WITH RECURSIVE RCTE (ORDER_ID,ITEM,QTY,EXPLODEROWS) AS
( SELECT ORDER_ID,ITEM,QTY, 1 FROM development.order_table
  UNION ALL
  SELECT ORDER_ID,ITEM,QTY,EXPLODEROWS + 1 FROM RCTE WHERE EXPLODEROWS < QTY)
SELECT ORDER_ID , ITEM , 1 AS  CNT FROM RCTE
ORDER BY ORDER_ID;

-- common table expression
WITH RECURSIVE cte (n) AS
(
  SELECT 1
  UNION ALL
  SELECT n + 1 FROM cte WHERE n < 5  -- limit 4
)
SELECT * FROM cte;


with recursive base (x, y) as
(
select 0, 1
union all
select y, x + y from base where y < 20
)
select * from base;


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


WITH RECURSIVE cte (n, p, q) AS
(
  SELECT 1 AS n, 1 AS p, -1 AS q
  UNION ALL
  SELECT n + 1, q * 2, p * 2 FROM cte WHERE n < 5
)
SELECT * FROM cte;


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