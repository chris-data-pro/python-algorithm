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


--// 查看date
SELECT metricId,
       CAST(date AS CHAR) AS day,
       COUNT(*)
FROM KpiMetricValue
WHERE metricId IN (29, 30, 31) GROUP BY 1,2;

SELECT id, experimentName, DATE_FORMAT(dumpDate, '%Y-%m-%d'), DATE_FORMAT(endDate, '%Y-%m-%d')
FROM phoenix_results_window_v2
ORDER BY RAND()
limit 10;

SELECT metricId, dimensionId, CAST(date AS CHAR) AS day, state, createdAt, debugInfo
FROM KpiMetricAnomaly
WHERE metricId = 6;

SELECT DISTINCT experimentName, DATE_FORMAT(endDate,'%d/%m/%Y') as data_through_date
FROM phoenix_results_window_v2
where endDate > '2021-09-09';

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
