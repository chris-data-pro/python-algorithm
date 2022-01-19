-- 7 give the top product by sales in each of. the group, and additionally gather glance views, inventory and ad spend
begin;
DROP TABLE IF EXISTS development.product_dimension;
commit;

begin;
create table development.product_dimension (
  product_id                 VARCHAR(10),
  product_group              VARCHAR(100),
  product_name               VARCHAR(100)
);
commit;

SELECT * FROM development.product_dimension;

begin;
INSERT INTO development.product_dimension
(product_id, product_group, product_name) VALUES
('P1', 'Book', 'Harry Potter 1'),
('P2', 'Book', 'Harry Potter 2'),
('P3', 'Electronics', 'Nikon 10 MPS'),
('P4', 'Electronics', 'Cannon 8 MPS'),
('P5', 'Electronics', 'Cannon 10 MPS'),
('P6', 'Video DVD', 'Pirates 1'),
('P7', 'Video DVD', 'Pirates 2'),
('P8', 'Video DVD', 'HP 1'),
('P9', 'Video DVD', 'HP 2'),
('P10', 'Shoes', 'Nike 10'),
('P11', 'Shoes', 'Nike 11'),
('P12', 'Shoes', 'Adidas 10'),
('P13', 'Shoes', 'Adidas 09'),
('P14', 'Book', 'God Father 1'),
('P15', 'Book', 'God Father 2');
commit;


begin;
DROP TABLE IF EXISTS development.sale_fact;
commit;

begin;
create table development.sale_fact (
  day                        date,
  product_id                 VARCHAR(10),
  sales_amt                  int
);
commit;

SELECT * FROM development.sale_fact;

begin;
INSERT INTO development.sale_fact
(day, product_id, sales_amt) VALUES
(date('2011-07-20'), 'P1', 10),
(date('2011-07-20'), 'P2', 5),
(date('2011-07-20'), 'P8', 100),
(date('2011-07-20'), 'P3', 5),
(date('2011-07-20'), 'P4', 25),
(date('2011-07-20'), 'P5', 15),
(date('2011-07-20'), 'P6', 35),
(date('2011-07-20'), 'P7', 5),
(date('2011-07-20'), 'P9', 30),
(date('2011-07-20'), 'P10', 8),
(date('2011-07-20'), 'P11', 45);
commit;


begin;
DROP TABLE IF EXISTS development.view_fact;
commit;

begin;
create table development.view_fact (
  day                        date,
  product_id                 VARCHAR(10),
  glance_views               int
);
commit;

SELECT * FROM development.view_fact;

begin;
INSERT INTO development.view_fact
(day, product_id, glance_views) VALUES
(date('2011-07-20'), 'P1', 1000),
(date('2011-07-20'), 'P2', 800),
(date('2011-07-20'), 'P8', 700),
(date('2011-07-20'), 'P3', 800),
(date('2011-07-20'), 'P4', 500),
(date('2011-07-20'), 'P5', 250),
(date('2011-07-20'), 'P6', 10),
(date('2011-07-20'), 'P7', 1000),
(date('2011-07-20'), 'P9', 1500),
(date('2011-07-20'), 'P10', 600),
(date('2011-07-20'), 'P12', 670),
(date('2011-07-20'), 'P13', 300),
(date('2011-07-20'), 'P14', 230);
commit;


begin;
DROP TABLE IF EXISTS development.inventory_fact;
commit;

begin;
create table development.inventory_fact (
  day                            date,
  product_id                     VARCHAR(10),
  on_hand_quantity               int
);
commit;

SELECT * FROM development.inventory_fact;

begin;
INSERT INTO development.inventory_fact
(day, product_id, on_hand_quantity) VALUES
(date('2011-07-20'), 'P1', 100),
(date('2011-07-20'), 'P2', 70),
(date('2011-07-20'), 'P8', 90),
(date('2011-07-20'), 'P3', 10),
(date('2011-07-20'), 'P4', 30),
(date('2011-07-20'), 'P5', 100),
(date('2011-07-20'), 'P6', 120),
(date('2011-07-20'), 'P7', 70),
(date('2011-07-20'), 'P9', 90);
commit;


begin;
DROP TABLE IF EXISTS development.ad_spend;
commit;

begin;
create table development.ad_spend (
  day                            date,
  product_id                     VARCHAR(10),
  ad_spend                       int
);
commit;

SELECT * FROM development.ad_spend;

begin;
INSERT INTO development.ad_spend
(day, product_id, ad_spend) VALUES
(date('2011-07-20'), 'P1', 10),
(date('2011-07-20'), 'P2', 5),
(date('2011-07-20'), 'P8', 100),
(date('2011-07-20'), 'P3', 5),
(date('2011-07-20'), 'P4', 25),
(date('2011-07-20'), 'P5', 15),
(date('2011-07-20'), 'P6', 35),
(date('2011-07-20'), 'P7', 5),
(date('2011-07-20'), 'P9', 30),
(date('2011-07-20'), 'P10', 8),
(date('2011-07-20'), 'P11', 45);
commit;


-- 7 give the top product by sales in each of. the group, and additionally gather glance views, inventory and ad spend
select foo.product_group, foo.product_id, foo.sales_amt, nvl(vf.glance_views, 0), nvl(inf.on_hand_quantity, 0), nvl(ads.ad_spend, 0)
from (
select
pd.product_group,
sf.product_id,
sf.sales_amt,
row_number() over (partition by product_group order by sales_amt desc) as rk
from development.product_dimension pd join development.sale_fact sf on pd.product_id = sf.product_id) foo
left join development.view_fact vf on foo.product_id = vf.product_id
left join development.inventory_fact inf on foo.product_id = inf.product_id
left join development.ad_spend ads on foo.product_id = ads.product_id
where foo.rk = 1
order by 1;


--or
SELECT pd.product_group,
       pd.product_id,
       sales_amt,
       rank() OVER (PARTITION BY pd.product_group ORDER BY sf.sales_amt DESC NULLS LAST) AS rk,
       NVL(vf.glance_views, 0),
       NVL(ift.on_hand_quantity, 0),
       NVL(ast.ad_spend, 0)
FROM development.product_dimension pd
LEFT JOIN development.sale_fact sf ON pd.product_id = sf.product_id
LEFT JOIN development.view_fact vf ON pd.product_id = vf.product_id
LEFT JOIN development.inventory_fact ift ON pd.product_id = ift.product_id
LEFT JOIN development.ad_spend ast ON pd.product_id = ast.product_id
QUALIFY rk = 1
order by 1;


-- 8 give all products that have glance views but no sales
select
vf.product_id
from development.sale_fact sf right join development.view_fact vf
on sf.product_id = vf.product_id
where sf.product_id is null --and vf.product_id is not null
;


--or
SELECT product_id FROM development.view_fact
MINUS
SELECT product_id FROM development.sale_fact


--product_id
--P12
--P13
--P14


-- 9 give the sales of electronics as a precentage of books
-- 不同行的数 算一个数
select 1 as id,
       sum(CASE WHEN pd.product_group = 'Electronics' THEN sf.sales_amt ELSE 0 END) as e_tot,
       sum(CASE WHEN pd.product_group = 'Book' THEN sf.sales_amt ELSE 0 END) as b_tot,
       e_tot / b_tot::float * 100
from development.product_dimension pd join development.sale_fact sf on pd.product_id = sf.product_id
group by 1;


--or
WITH e AS (
select pd.product_group, sum(sf.sales_amt) as e_tot
from development.product_dimension pd join development.sale_fact sf on pd.product_id = sf.product_id
where pd.product_group = 'Electronics'
group by 1),
b AS (
select pd.product_group, sum(sf.sales_amt) as b_tot
from development.product_dimension pd join development.sale_fact sf on pd.product_id = sf.product_id
where pd.product_group = 'Book'
group by 1)
SELECT e_tot / b_tot::float * 100
FROM e JOIN b ON true


--or
with a as (select 1 as id,
pd.product_group,
sum(sf.sales_amt) as sum_electronics
from development.product_dimension pd join development.sale_fact sf on pd.product_id = sf.product_id
where pd.product_group = 'Electronics'
GROUP by 1, 2),
b as (select 1 as id,
pd.product_group,
sum(sf.sales_amt) as sum_books
from development.product_dimension pd join development.sale_fact sf on pd.product_id = sf.product_id
where pd.product_group = 'Book'
GROUP by 1, 2)
select a.sum_electronics / b.sum_books::float * 100
from a join b on a.id = b.id;


-- 11
begin;
DROP TABLE IF EXISTS development.customer_orders;
commit;

begin;
create table development.customer_orders (
  order_day                       date,
  customer_id                     int
);
commit;

SELECT * FROM development.customer_orders;

begin;
INSERT INTO development.customer_orders
(order_day, customer_id) VALUES
(date('2020-07-20'), 1),
(date('2020-09-20'), 2),
(date('2021-07-20'), 3);
commit;

begin;
DROP TABLE IF EXISTS development.customer_orders_daily;
commit;

begin;
create table development.customer_orders_daily (
  snapshot_day                    date,
  order_day                       date,
  customer_id                     int
);
commit;

SELECT * FROM development.customer_orders_daily;

begin;
INSERT INTO development.customer_orders_daily
(snapshot_day, order_day, customer_id) VALUES
(date('2020-10-11'), date('2020-10-11'), 1),
(date('2020-10-11'), date('2020-10-11'), 2),
(date('2020-10-11'), date('2020-10-11'), 4),
(date('2020-10-11'), date('2020-10-11'), 4);
commit;

with customers as (select distinct customer_id from development.customer_orders),
ly_customers as (select distinct customer_id from development.customer_orders where order_day >= current_date - interval '1 year')
select
foo.snapshot_day,
sum(case when foo.customer_id not in (
select customer_id from customers
) then 1 else 0 end) as new_custormers,
sum(case when foo.customer_id in (
select customer_id from customers
) and foo.customer_id not in (
select customer_id from ly_customers
) then 1 else 0 end) as reactivated_customers
from (select distinct snapshot_day, customer_id from development.customer_orders_daily) foo
group by 1;


