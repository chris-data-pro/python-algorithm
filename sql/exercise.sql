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


-----------------------------------------
-- assignment
-----------------------------------------
ACCOUNTS (Columns)

ACCOUNT_ID
ACCOUNT_NAME
SUBSCRIPTION_ID


ACCOUNT_DETAILS (Columns)

ACCOUNT_ID
ACCOUNT_COUNTRY
CSM_MANAGER_NAME


USAGE_DETAILS (Columns)

SUBSCRIPTION_ID
MONTH_ID
LICENSE_QTY
ACTIVE_HOST
TOTAL_MEETINGS


SITE_DETAILS (Columns)

SUBSCRIPTION_ID
SITE_ID
SITE_NAME
AUDIO_TYPE



--1) Total number of accounts and subscriptions managed by CSM manager “Curtis” in USA
Select CSM_MANAGER_NAME, ACCOUNT_COUNTRY, count(distinct a.ACCOUNT_ID), count(distinct ad.SUBSCRIPTION_ID)
From ACCOUNTS a join ACCOUNT_DETAILS ad
on a.ACCOUNT_ID = ad.ACCOUNT_ID
Where CSM_MANAGER_NAME = 'Curtis' and ACCOUNT_COUNTRY = 'USA'
Group by 1,2;



--2) How many accounts have subscriptions with more than 1000 licenses in the month of October 2021 (current month)
Select count(distinct a.ACCOUNT_ID)
From ACCOUNTS a join USAGE_DETAILS ud
On a.SUBSCRIPTION_ID = ud.SUBSCRIPTION_ID
Join (Select max(month_id) as current_month
From USAGE_DETAILS) p
On ud.MONTH_ID = p.current_month
Where ud.LICENSE_QTY > 1000
;



--3) Show all the sites of accounts in Australia with Audio Type as “Webex Audio”
Select sd.SITE_ID
From SITE_DETAILS sd join ACCOUNTS a
On sd.SUBSCRIPTION_ID = a.SUBSCRIPTION_ID
Join ACCOUNT_DETAILS ad
On a.ACCOUNT_ID = ad.ACCOUNT_ID
Where sd.AUDIO_TYPE = 'Webex Audio' and ad.ACCOUNT_COUNTRY = 'Australia'
;


--4) Show the accounts with subscriptions that have number of “Active Hosts” more than License qty for previous month
Select acount_id
From ACCOUNTS a join USAGE_DETAILS ud On a.SUBSCRIPTION_ID = ud.SUBSCRIPTION_ID
join (
Select DISTINCT month_id as previous_month, dense_rank() over (order by month_id desc) as rk
From USAGE_DETAILS
Where rk = 2
) pm on ud.month_id = pm.previous_month
Where ud.ACTIVE_HOST > ud.LICENSE_QTY
;


--5) Show the subscriptions with decrease in TOTAL_MEETINGS from previous month to current month
--   with at least one site has Audio Type as “CCA SP”
With base as (
Select SUBSCRIPTION_ID
From SITE_DETAILS
Where  AUDIO_TYPE = 'CCA SP')
Select distinct ud.SUBSCRIPTION_ID
From USAGE_DETAILS ud join base sd on ud.SUBSCRIPTION_ID = sd.SUBSCRIPTION_ID
Where ud.SUBSCRIPTION_ID in (Select SUBSCRIPTION_ID
From USAGE_DETAILS a join USAGE_DETAILS b on a.SUBSCRIPTION_ID = b.SUBSCRIPTION_ID
Where a.month_id = lag( b.month_id )
And b.TOTAL_MEETINGS > a.TOTAL_MEETINGS);


SELECT SUBSCRIPTION_ID
FROM USAGE_DETAILS ud JOIN SITE_DETAILS sd ON ud.SUBSCRIPTION_ID = sd.SUBSCRIPTION_ID
WHERE sd.AUDIO_TYPE = 'CCA SP'
QUALIFY LAG(ud.TOT_MTGS, 1) OVER (PARTITION BY ud.SUBSCRIPTION_ID ORDER BY ud.MONTH_ID) > TOT_MTGS


-------------
-- codility--
-------------
-- 1 --
begin;
create table development.inputs (
  id  int
	,Name                        char(50)
	, AnnualSalary                   int
	, ManagerId                  int
);
commit;

SELECT * FROM development.inputs;

begin;
INSERT INTO development.inputs
(id, Name, AnnualSalary, ManagerId) VALUES
(1, 'Lisa Smith', 150000, NULL),
(2, 'Dan Bradley',110000,1),
(3, 'Oliver Queen',180000,1),
(4, 'Dave Dakota',100000,1),
(5 ,'Steve Carr',200000,NULL),
(6,'Alice Johnson',205000,5),
(7,'Damian Luther', 100000,5),
(8,'Avery Montgomery',210000,5),
(9,'Mark Spencer',140000,5),
(10,'Melanie Thorthon',200000 ,NULL),
(11,'Dana Parke',100000 ,10),
(12,'Antonio Maker' ,120000, 10),
(13,'Lucille Alvarez',140000,10 );
commit;

-- given the above table, ManagerId = NULL means this row is the manager.
-- return the 2nd largest AnnualSalary row for each team
with base as (
SELECT Name,
       AnnualSalary,
       NVL(ManagerId, ID) as ManagerId,
       ManagerId as omid
FROM development.inputs),
base2 as (
select Name , AnnualSalary, ManagerId, omid, row_number() over (partition by ManagerId order by AnnualSalary desc) as rk
from base)
select Name , AnnualSalary, omid as ManagerId FROM base2
where rk = 2
;


-- 2 --
begin;
create table development.inputss (
Source char(50),
Month int,
Year int,
NumberOfVisits int
);
commit;

begin;
INSERT INTO development.inputss
(Source, Month, Year, NumberOfVisits) VALUES
('Facebook',12,2019,17746),
('Twitter',12,2019,55559),
('Google',12,2019,98320),
('Partner blogs',12,2019,77008),
('YouTube',12,2019,50514),
('Organic',12,2019,60549),
('Direct traffic',12,2019,65269),
('Facebook',11,2019,4736),
('Twitter',11,2019,13997),
('Google',11,2019,50531),
('Partner blogs',11,2019,98214),
('YouTube',11,2019,68333),
('Organic',11,2019,6657),
('Direct traffic',11,2019,16722),
('Facebook',10,2019,24828),
('Twitter',10,2019,51223),
('Google',10,2019,90950),
('Partner blogs',10,2019,82740),
('YouTube',10,2019,47301),
('Organic',10,2019,95222),
('Direct traffic',10,2019,31202),
('Facebook',9,2019,371),
('Twitter',9,2019,49343),
('Google',9,2019,54496),
('Partner blogs',9,2019,64032),
('YouTube',9,2019,1082),
('Organic',9,2019,23057),
('Direct traffic',9,2019,14259),
('Facebook',8,2019,39160),
('Twitter',8,2019,44046),
('Google',8,2019,90683),
('Partner blogs',8,2019,11121),
('YouTube',8,2019,17437),
('Organic',8,2019,96938),
('Direct traffic',8,2019,35851),
('Facebook',7,2019,94566),
('Twitter',7,2019,61651),
('Google',7,2019,12448),
('Partner blogs',7,2019,22508),
('YouTube',7,2019,3182),
('Organic',7,2019,97344),
('Direct traffic',7,2019,65591),
('Facebook',6,2019,80091),
('Twitter',6,2019,99512),
('Google',6,2019,95462),
('Partner blogs',6,2019,13796),
('YouTube',6,2019,38291),
('Organic',6,2019,25956),
('Direct traffic',6,2019,58631),
('Facebook',5,2019,57206),
('Twitter',5,2019,44790),
('Google',5,2019,82806),
('Partner blogs',5,2019,3394),
('YouTube',5,2019,99838),
('Organic',5,2019,54843),
('Direct traffic',5,2019,43567),
('Facebook',4,2019,75688),
('Twitter',4,2019,99975),
('Google',4,2019,23695),
('Partner blogs',4,2019,59881),
('YouTube',4,2019,34196),
('Organic',4,2019,90551),
('Direct traffic',4,2019,35303),
('Facebook',3,2019,45507),
('Twitter',3,2019,32727),
('Google',3,2019,81718),
('Partner blogs',3,2019,62395),
('YouTube',3,2019,92016),
('Organic',3,2019,36045),
('Direct traffic',3,2019,22458),
('Facebook',2,2019,91787),
('Twitter',2,2019,58565),
('Google',2,2019,40815),
('Partner blogs',2,2019,45265),
('YouTube',2,2019,3140),
('Organic',2,2019,3721),
('Direct traffic',2,2019,75957),
('Facebook',1,2019,37738),
('Twitter',1,2019,96433),
('Google',1,2019,38371),
('Partner blogs',1,2019,10559),
('YouTube',1,2019,20127),
('Organic',1,2019,38770),
('Direct traffic',1,2019,65176),
('Facebook',12,2018,85293),
('Twitter',12,2018,93395),
('Google',12,2018,79139),
('Partner blogs',12,2018,60331),
('YouTube',12,2018,97491),
('Organic',12,2018,79703),
('Direct traffic',12,2018,12949),
('Facebook',11,2018,17284),
('Twitter',11,2018,17244),
('Google',11,2018,48557),
('Partner blogs',11,2018,75090),
('YouTube',11,2018,18629),
('Organic',11,2018,53198),
('Direct traffic',11,2018,70218),
('Facebook',10,2018,46042),
('Twitter',10,2018,41277),
('Google',10,2018,78005),
('Partner blogs',10,2018,45126),
('YouTube',10,2018,28210),
('Organic',10,2018,36809),
('Direct traffic',10,2018,86158),
('Facebook',9,2018,24201),
('Twitter',9,2018,76333),
('Google',9,2018,64317),
('Partner blogs',9,2018,4026),
('YouTube',9,2018,36363),
('Organic',9,2018,22198),
('Direct traffic',9,2018,81736),
('Facebook',8,2018,68566),
('Twitter',8,2018,81109),
('Google',8,2018,41651),
('Partner blogs',8,2018,33305),
('YouTube',8,2018,23628),
('Organic',8,2018,48051),
('Direct traffic',8,2018,11104),
('Facebook',7,2018,39492),
('Twitter',7,2018,87153),
('Google',7,2018,75735),
('Partner blogs',7,2018,47492),
('YouTube',7,2018,32978),
('Organic',7,2018,33077),
('Direct traffic',7,2018,62785),
('Facebook',6,2018,39499),
('Twitter',6,2018,99969),
('Google',6,2018,53298),
('Partner blogs',6,2018,7128),
('YouTube',6,2018,1534),
('Organic',6,2018,14739),
('Direct traffic',6,2018,49418),
('Facebook',5,2018,12844),
('Twitter',5,2018,18756),
('Google',5,2018,16417),
('Partner blogs',5,2018,45193),
('YouTube',5,2018,23538),
('Organic',5,2018,75795),
('Direct traffic',5,2018,49088),
('Facebook',4,2018,10116),
('Twitter',4,2018,76908),
('Google',4,2018,43579),
('Partner blogs',4,2018,72605),
('YouTube',4,2018,22806),
('Organic',4,2018,50523),
('Direct traffic',4,2018,14570),
('Facebook',3,2018,96297),
('Twitter',3,2018,2857),
('Google',3,2018,93920),
('Partner blogs',3,2018,55506),
('YouTube',3,2018,48027),
('Organic',3,2018,96478),
('Direct traffic',3,2018,97143),
('Facebook',2,2018,96143),
('Twitter',2,2018,11641),
('Google',2,2018,67057),
('Partner blogs',2,2018,41386),
('YouTube',2,2018,29417),
('Organic',2,2018,68571),
('Direct traffic',2,2018,22230),
('Facebook',1,2018,98446),
('Twitter',1,2018,35110),
('Google',1,2018,76194),
('Partner blogs',1,2018,43192),
('YouTube',1,2018,68404),
('Organic',1,2018,97860),
('Direct traffic',1,2018,68406);
commit;


with base as (
select source, month,
               year, cast(year as varchar(4))||'-'||cast(month as varchar(2)) as year_month, numberofvisits
from development.inputss)
select a.source,
       b.year_month as previous_month,
       b.numberofvisits as previous_visits,
       a.year_month as year_month,
       a.numberofvisits,
       cast((a.numberofvisits - b.numberofvisits) / b.numberofvisits::float * 100 as varchar(5))||'%' as percent
from base a
join base b
on (a.year = b.year and a.month = b.month + 1 and a.source = b.source) OR
   (a.year = b.year + 1 and a.month = 1 and b.month = 12 and a.source = b.source)
order by 1,2
;

--source       previous_month   previous_visits     year_month      numberofvisits      percent
--Facebook                                          	2018-1	98446	2018-2	96143	-2.33%
--Facebook                                          	2018-10	46042	2018-11	17284	-62.4%
--Facebook                                          	2018-11	17284	2018-12	85293	393.4%
--Facebook                                          	2018-12	85293	2019-1	37738	-55.7%
--Facebook                                          	2018-2	96143	2018-3	96297	0.160%
--Facebook                                          	2018-3	96297	2018-4	10116	-89.4%
--Facebook                                          	2018-4	10116	2018-5	12844	26.96%
--Facebook                                          	2018-5	12844	2018-6	39499	207.5%
--Facebook                                          	2018-6	39499	2018-7	39492	-0.01%
--Facebook                                          	2018-7	39492	2018-8	68566	73.61%
--Facebook                                          	2018-8	68566	2018-9	24201	-64.7%
--Facebook                                          	2018-9	24201	2018-10	46042	90.24%
--Facebook                                          	2019-1	37738	2019-2	91787	143.2%
--Facebook                                          	2019-10	24828	2019-11	4736	-80.9%
--Facebook                                          	2019-11	4736	2019-12	17746	274.7%
--Facebook                                          	2019-2	91787	2019-3	45507	-50.4%
--Facebook                                          	2019-3	45507	2019-4	75688	66.32%
--Facebook                                          	2019-4	75688	2019-5	57206	-24.4%
--Facebook                                          	2019-5	57206	2019-6	80091	40.00%
--Facebook                                          	2019-6	80091	2019-7	94566	18.07%
--Facebook                                          	2019-7	94566	2019-8	39160	-58.5%
--Facebook                                          	2019-8	39160	2019-9	371	    -99.0%
--Facebook                                          	2019-9	371	    2019-10	24828	6592.%



--
SELECT config FROM experiment WHERE id IN (SELECT MAX(id) FROM experiment GROUP BY hashId);


-- #1 --

with base as
(SELECT tg.name, tg.test_value, tc.status
FROM test_groups AS tg
LEFT JOIN test_cases AS tc
ON tg.name = tc.group_name)
select name,
            sum(case when status in ('OK', 'ERROR') then 1 else 0 end) as all_test_cases,
            sum(case when status in (‘OK’) then 1 else 0 end) as passed_test_cases,
            sum(case when status in (‘OK’) then test_value else 0 end) as total_value
from base
group by 1
ORDER BY 4 DESC, 1;


-- #2 --

SELECT DISTINCT name
FROM (
select p1.name, p1.year, p2.year, p3.year
from development.p p1
  inner join development.p p2
  on p1.year = p2.year + 1 and p1.name = p2.name
  inner join development.p p3
  on p1.year = p3.year + 2 and p1.name = p3.name
) foo
ORDER BY 1;


SELECT
name,
year,
first_value(year) over (partition by name order by year desc) - year + 1 as x,
row_number() over (partition by name order by year desc) as rn
FROM participation;


--// 拿一个experiment的最新状态和config by experimentId
select status, config
from experiment
where hashId = '7db478ed9e76d3d282d9' AND id IN (SELECT MAX(id) FROM experiment GROUP BY hashId);

--// 查看requestHashId by experimentId，如果是新upload的request必须run一次batch assignment才有requestHashId
select * from batch_assignment_request where experimentId = ‘7db478ed9ee3ad974e5ab376d3d282d9’

--// 看table schema
describe batch_assignment_request;

--// 查新加入的metric
SELECT * FROM phoenix_results_window
WHERE experimentId = 'f442313ae777d030414fbb92d6'
  and endDate = '2021-08-11'
  and metricName = 'l7_has_four_way_conversation'
LIMIT 3;

--// 改一行
UPDATE batch_assignment_process_log
SET createdAt = 333333
WHERE requestHashId = '456';

--// 加一列
ALTER TABLE batch_assignment_request ADD COLUMN processedCount bigint(20);

--// 加一行
insert ignore batch_assignment_process_log (requestHashId, backfillDays, experimentId, createdAt) values
(456, 0, 'asbdfdkjfldk', 333333);

INSERT INTO event (type, name, pod, source, effectiveTime, note, link, externalId, triggeredBy, createdAt)
VALUES ('github','new_exp_with_trial_1','other','Phoenix',16226161461,'Experiment started, triggered by scheduling',
        'https://tools.gotinder.com/#/phoenix/experiments/53c7176cc2d02c975c3cf64bae504680',
        '53c7176cc2d0f64bae504680','super.man@gmail.com',1622616146156);

--// 加多行
insert ignore phoenix_analytics_simplified_daily_results
(experimentId, gender, targetGender, platform, ageGroup, city, country, date,
dumpDate, entityCount, experimentBucket, metricName, metricValue)
values
('e2d4fd29b8e543c4f8042acf12d51701', 'all', 'all', 'all', 'all', 'all', 'all', '2021-03-01',
'2021-03-02', 1.0, 'enabled', 'Assignment.New', 35178.0),
('e2d4fd29b8e543c4f8042acf12d51701', 'all', 'all', 'all', 'all', 'all', 'all', '2021-03-01',
'2021-03-02', 1.0, 'control', 'Assignment.New', 35070.0);


-- 90 minutes SQL for data engineer
begin;
DROP TABLE IF EXISTS development.older_table;
commit;

begin;
create table development.older_table (
	order_day                        char varying(50)
	, order_id                       char varying(5)
	, product_id                     char varying(5)
	, quantity                       int
	, price                          int
);
commit;

SELECT * FROM development.older_table;

begin;
INSERT INTO development.older_table
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


select
order_day,
order_id,
quantity,
sum(quantity) over (partition by order_day) as total_quantity  -- or max, min
from development.older_table
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


select order_day, listagg(order_id::varchar, ';') within group (order by quantity desc) as order_list
from development.older_table
group by 1;

--01-JUL-2011	O4;O3;O1;O2
--02-JUL-2011	O6;O5;O10;O7;O9;O8


-- 1 get all products that got sold both the days, and the number of times sold
select product_id, count(distinct order_id) as time_sold
from development.older_table
where product_id in (
select distinct d1.product_id --, d1.order_id as day1order, d1.order_day as day1, d2.order_id as day2order, d2.order_day as day2
from development.older_table d1
join development.older_table d2
on d1.product_id = d2.product_id
and d1.order_day = '01-JUL-2011'
and d2.order_day = '02-JUL-2011'
)
group by 1;


select product_id, count(distinct order_id) as time_sold
from development.older_table
where product_id in (
select product_id from (
select product_id, order_day, order_id, dense_rank() over (partition by product_id order by order_day) as rk
from development.older_table
order by 1,2,3) foo
where rk = 2)
group by 1;


-- 2 get products that was ordered on '02-JUL-2011' but not on '01-JUL-2011'
select product_id from (
select product_id, order_day, order_id, dense_rank() over (partition by product_id order by order_day) as rk
from development.older_table
order by 1,2,3) foo
where rk = 1 and order_day = '02-JUL-2011';


-- 3 get the highest sold products () on these 2 days
select order_day, product_id, sold_amount
from
(select order_day, product_id, sum(quantity * price) as sold_amount, row_number() over (partition by order_day order by sold_amount desc) as rk
from development.older_table
group by 1, 2) foo
where rk = 1
order by 1;


-- 4 get all products and the total sales on day 1 and day 2
select
nvl(d1.product_id, d2.product_id) as product_id,
nvl(d1.total_sales_01, 0) as total_sales_01,
nvl(d2.total_sales_02, 0) as total_sales_02
from
(select product_id, sum(quantity * price) as total_sales_01
from development.older_table
where order_day = '01-JUL-2011'
group by 1) d1 full outer join
(select product_id, sum(quantity * price) as total_sales_02
from development.older_table
where order_day = '02-JUL-2011'
group by 1) d2 on d1.product_id = d2.product_id
order by 1;


-- 5 get all order_day product_id vis, that was ordered more than once
select order_day, product_id, count(*) as ct
from development.older_table
group by 1, 2
having ct > 1

select order_day, product_id
from
(select order_day, product_id, count(*)
from development.older_table
group by order_day,product_id
having count(*)>1
);


-- 6 explode the data into single unit level records
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


-- 8 give all products that have glance views but no sales
select
vf.product_id
from development.sale_fact sf right join development.view_fact vf
on sf.product_id = vf.product_id
where sf.product_id is null --and vf.product_id is not null
;


-- 9 give the sales of electronics as a precentage of books
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


-- 10
begin;
DROP TABLE IF EXISTS development.phone_dial;
commit;

begin;
create table development.phone_dial (
  source_number                  int,
  destination_number             int,
  call_start_date                datetime
);
commit;

SELECT * FROM development.phone_dial;

begin;
INSERT INTO development.phone_dial
(source_number, destination_number, call_start_date) VALUES
(1234, 4567, CONVERT(datetime, '2011-07-20 10:00:00')),
(1234, 2345, CONVERT(datetime, '2011-07-20 11:00:00')),
(1234, 3456, CONVERT(datetime, '2011-07-20 12:00:00')),
(1234, 3456, CONVERT(datetime, '2011-07-20 13:00:00')),
(1234, 4567, CONVERT(datetime, '2011-07-20 15:00:00')),
(1222, 7890, CONVERT(datetime, '2011-07-20 10:00:00')),
(1222, 7680, CONVERT(datetime, '2011-07-20 12:00:00')),
(1222, 2345, CONVERT(datetime, '2011-07-20 13:00:00'));
commit;

with asc_order as (
select source_number, destination_number as first_called from (
select source_number, destination_number,
rank() over (partition by source_number order by call_start_date) as rk
from development.phone_dial) a
where a.rk = 1),
desc_order as (
select source_number, destination_number as last_called from (
select source_number, destination_number,
rank() over (partition by source_number order by call_start_date desc) as rk
from development.phone_dial) b
where b.rk = 1)
select ao.source_number, (case when ao.first_called = deo.last_called then 'Y' else 'N' end) as flag
from asc_order ao join desc_order deo on ao.source_number = deo.source_number;


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


-- 3) get a list of customers (cust_id) who have placed less than 2 orders or have ordered for less than $100
begin;
DROP TABLE IF EXISTS development.cust_orders;
commit;

begin;
create table development.cust_orders (
  order_id                        int,
  cust_id                         int,
  order_date                      date,
  product                         VARCHAR(20),
  order_amount                    int

);
commit;

SELECT * FROM development.cust_orders;

begin;
INSERT INTO development.cust_orders
(order_id, cust_id, order_date, product, order_amount) VALUES
(50001, 101, date('2015-08-29'), 'Camera', 100),
(50002, 102, date('2015-08-30'), 'Shoes', 90),
(50003, 103, date('2015-05-31'), 'Laptop', 400),
(50004, 101, date('2015-08-29'), 'Mobile', 100),
(50005, 104, date('2015-08-29'), 'FrozenMeals', 30),
(50006, 104, date('2015-08-30'), 'Cloths', 65);
commit;


select foo.cust_id
FROM
(
select cust_id, count(distinct order_id) as no_orders, sum(order_amount) as total_amt
from development.cust_orders
group by 1
HAVING count(distinct order_id) < 2 or sum(order_amount) < 100
) foo
order by 1;


-- 5) identify callers who made their first call and the last call to the same recipient on a given day
begin;
DROP TABLE IF EXISTS development.phone_log;
commit;

begin;
create table development.phone_log (
  caller_id                  int,
  recipient_id               int,
  call_start_time            datetime
);
commit;

SELECT * FROM development.phone_log;

begin;
INSERT INTO development.phone_log
(caller_id, recipient_id, call_start_time) VALUES
(101, 201, CONVERT(datetime, '2015-08-31 02:35:00')),
(101, 301, CONVERT(datetime, '2015-08-31 02:37:00')),
(101, 501, CONVERT(datetime, '2015-08-31 02:39:00')),
(101, 201, CONVERT(datetime, '2015-09-01 02:41:00')),
(101, 201, CONVERT(datetime, '2015-09-01 05:00:00')),
(103, 401, CONVERT(datetime, '2015-08-31 05:35:00')),
(104, 501, CONVERT(datetime, '2015-08-31 06:35:00')),
(104, 601, CONVERT(datetime, '2015-08-31 07:35:00'));
commit;


with asc_order as (
select day, caller_id, recipient_id
from
(
select cast(call_start_time as date) as day, caller_id, recipient_id, rank() over (partition by day, caller_id order by call_start_time) as rk
from development.phone_log
) foo
where foo.rk = 1),
desc_order as (
select day, caller_id, recipient_id
from
(
select cast(call_start_time as date) as day, caller_id, recipient_id, rank() over (partition by day, caller_id order by call_start_time desc) as rk
from development.phone_log
) b
where b.rk = 1)
select ao.caller_id, ao.recipient_id, ao.day as call_date
from asc_order ao
join desc_order deo
on ao.day = deo.day and ao.caller_id = deo.caller_id and ao.recipient_id = deo.recipient_id
;