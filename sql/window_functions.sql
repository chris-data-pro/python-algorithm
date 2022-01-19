select
parent_id, parent_name,
row_number() over (order by parent_id, parent_name) as row_number, -- 1,2,3,4,...
rank() over (order by parent_id, parent_name) as rank, -- 1,1,3,4,5,5,5,8,9,...
dense_rank() over (order by parent_id, parent_name) as dense_rank -- 1,1,2,3,4,4,4,5,6,...
from development.parents
order by 1;

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

--4) Show the accounts with subscriptions that have number of “Active Hosts” more than License qty for previous month
with base as (Select DISTINCT month_id, dense_rank() over (order by month_id DESC) as rk
              From "COLLAB_DB"."COLLAB_DS2_CSSTG"."PM_WBX_SITE_USAGE_MLY_WITH_PREDICTIVE_FEATURES")
select a.account_id, b.month_id
from "COLLAB_DB"."COLLAB_DS2_CSWI"."CS_SKU_BLIS_ANNUITY" as a
join "COLLAB_DB"."COLLAB_DS2_CSSTG"."PM_WBX_SITE_USAGE_MLY_WITH_PREDICTIVE_FEATURES" as b
on a.SUBSCRIPTION_ID = b.SUBSCRIPTION_ID
join base
on base.month_id = b.month_id
where base.rk = 2

-- or
with base as (Select DISTINCT SUBSCRIPTION_ID, month_id, dense_rank() over (order by month_id DESC) as rk
              From "COLLAB_DB"."COLLAB_DS2_CSSTG"."PM_WBX_SITE_USAGE_MLY_WITH_PREDICTIVE_FEATURES")
select a.account_id, b.month_id
from "COLLAB_DB"."COLLAB_DS2_CSWI"."CS_SKU_BLIS_ANNUITY" as a
join base b
on a.SUBSCRIPTION_ID = b.SUBSCRIPTION_ID
where b.rk = 2

-- or
select a.account_id, b.month_id, dense_rank() over (order by b.month_id DESC) as rk
from "COLLAB_DB"."COLLAB_DS2_CSWI"."CS_SKU_BLIS_ANNUITY" as a
join "COLLAB_DB"."COLLAB_DS2_CSSTG"."PM_WBX_SITE_USAGE_MLY_WITH_PREDICTIVE_FEATURES" as b
on a.SUBSCRIPTION_ID = b.SUBSCRIPTION_ID
qualify rk = 2


SELECT site_name,
       lag(month_id, 1) over (partition by site_name order by month_id) as last_month,  -- assuming continuous months
       lag(TOT_MTGS, 1) over (partition by site_name order by month_id) as last_tot,
       month_id as current_month, TOT_MTGS as current_tot,
       lag(month_id, -1) over (partition by site_name order by month_id) as next_month,
       lag(TOT_MTGS, -1) over (partition by site_name order by month_id) as next_tot
FROM "COLLAB_DB"."COLLAB_DS2_CSSTG"."PM_WBX_SITE_USAGE_MLY_WITH_PREDICTIVE_FEATURES"
WHERE site_name = 'uson'


-- 选比前月增长的
SELECT site_name,
       lag(TOT_MTGS, 1) over (partition by site_name order by month_id) as last_tot,
       month_id as current_month,
       TOT_MTGS as current_tot
FROM "COLLAB_DB"."COLLAB_DS2_CSSTG"."PM_WBX_SITE_USAGE_MLY_WITH_PREDICTIVE_FEATURES"
WHERE site_name = 'uson'
QUALIFY current_tot > last_tot  -- window function can only be inside SELECT, QUALIFY or ORDER BY clauses


--5) Show the subscriptions with decrease in TOTAL_MEETINGS from previous month to current month
--   with at least one site has Audio Type as “CCA SP”
SELECT SUBSCRIPTION_ID
FROM USAGE_DETAILS ud JOIN SITE_DETAILS sd ON ud.SUBSCRIPTION_ID = sd.SUBSCRIPTION_ID
WHERE sd.AUDIO_TYPE = 'CCA SP'
QUALIFY LAG(ud.TOT_MTGS, 1) OVER (PARTITION BY ud.SUBSCRIPTION_ID ORDER BY ud.MONTH_ID) > TOT_MTGS



--------------
--Codelity 1
--------------
begin;
create table development.salary (
  employee_id  int
	,name                        varchar(50)
	, annual_salary                   int
	, manager_id                  int
);
commit;

SELECT * FROM development.salary;

begin;
INSERT INTO development.salary
(employee_id, name, annual_salary, manager_id) VALUES
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

-- given the above table, manager_id = NULL means this row is the manager.
-- return the 2nd largest annual_salary row for each team
SELECT employee_id, name, annual_salary, manager_id, NVL(manager_id, employee_id) as team,
       dense_rank() over (partition by team order by annual_salary DESC) as rk
FROM development.salary
qualify rk = 2


--------------
--Codelity 2
--------------

begin;
create table development.page_visits (
source varchar(50),
month int,
year int,
number_of_visits int
);
commit;

begin;
INSERT INTO development.page_visits
(source, month, year, number_of_visits) VALUES
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


-- this has rows with empty previous_month & previous_visits & percentage & p
SELECT source,
       CASE WHEN month < 10 THEN '0' || cast(month as varchar(1)) ELSE cast(month as varchar(2)) END as str_month,
       cast(year as varchar(4)) || '-' || str_month as year_month,
       number_of_visits,
       lag(year_month, 1) over (partition by source order by year_month) as previous_month,
       lag(number_of_visits, 1) over (partition by source order by year_month) as previous_visits,
       round((number_of_visits - previous_visits) / previous_visits::float * 100, 4) as percentage,
       cast(percentage as varchar) || '%' as p
FROM development.page_visits
ORDER BY 1, 3


-- or, no rows with empty previous_month & previous_visits & percentage & p
SELECT c.source,
       CASE WHEN c.month < 10 THEN '0' || cast(c.month as varchar(1)) ELSE cast(c.month as varchar(2)) END as current_str_month,
       cast(c.year as varchar(4)) || '-' || current_str_month as year_month,
       c.number_of_visits,
       CASE WHEN p.month < 10 THEN '0' || cast(p.month as varchar(1)) ELSE cast(p.month as varchar(2)) END as previous_str_month,
       cast(p.year as varchar(4)) || '-' || previous_str_month as previous_month,
       p.number_of_visits as previous_visits,
       round((c.number_of_visits - previous_visits) / previous_visits::float * 100, 4) as percentage,
       (cast(percentage as varchar) || '%') as p
FROM development.page_visits c
JOIN development.page_visits p
ON c.source = p.source AND
   (c.year = p.year AND c.month = p.month + 1) OR
   (c.year = p.year + 1 AND c.month = 1 AND p.month = 12)
ORDER BY 1, 3


