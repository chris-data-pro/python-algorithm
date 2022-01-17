--n consecutive， n is a constant
begin;
drop table IF EXISTS development.participation;
commit;

begin;
CREATE TABLE development.participation (
   name varchar(50),
   year int
);
commit;

begin;
INSERT INTO development.participation VALUES
('John',2003),
('Lyla',1994),
('Faith',1996),
('John',2002),
('Carol',2000),
('Carol',1999),
('John',2001),
('Carol',2002),
('Lyla',1996),
('Lyla',1997),
('Carol',2001),
('John',2009);
commit;


SELECT * FROM development.participation

--Show only the Names in the table where the users have participated for 3 consecutive years or more
SELECT name,
       LAG(year, 2) OVER (PARTITION BY name ORDER BY year) as year_lag2,
       LAG(year, 1) OVER (PARTITION BY name ORDER BY year) as year_lag1,
       year
FROM development.participation
QULIFY year_lag2 = year - 2 AND year_lag1 = year - 1;


with base as (
SELECT name,
       LAG(year, 2) OVER (PARTITION BY name ORDER BY year) as year_lag2,
       LAG(year, 1) OVER (PARTITION BY name ORDER BY year) as year_lag1,
       year
FROM development.participation)
select *
from base
where year_lag2 = year - 2 AND year_lag1 = year - 1;

-- name  year_lag2 year_lag1 year
-- Carol 2000      2001  2002
-- Carol 1999      2000  2001
-- John   2001      2002  2003


SELECT c.name, l2.year as year_lag2, l1.year as year_lag1, c.year
FROM development.participation c
JOIN development.participation l1
ON c.name = l1.name AND c.year = l1.year + 1
JOIN development.participation l2
ON c.name = l2.name AND c.year = l2.year + 2


--or generic way
select name, flag, count(*) as ct
from
(SELECT name,
       year,
       row_number() over (partition by name order by year) as rn,
       year - rn as flag
FROM development.participation)
group by name, flag
having ct >= 3


--# Logs table:
--# +------------+
--# | log_id     |
--# +------------+
--# | 1          |
--# | 2          |
--# | 3          |
--# | 7          |
--# | 8          |
--# | 10         |
--# +------------+
--#
--# Result table:
--# +------------+--------------+
--# | start_id   | end_id       |
--# +------------+--------------+
--# | 1          | 3            |
--# | 7          | 8            |
--# | 10         | 10           |
--# +------------+--------------+
--# The result table should contain all ranges in table Logs.
--# From 1 to 3 is contained in the table.
--# From 4 to 6 is missing in the table
--# From 7 to 8 is contained in the table.
--# Number 9 is missing in the table.
--# Number 10 is contained in the table.
--#
--#
--# 1  1  0
--# 2  2  0
--# 3  3  0
--# 7  4  3
--# 8  5  3
--# 10 6  4


-- x consecutive
select distinct min(log_id) over (partition by flag) as start_id, max(log_id) over (partition by flag) as end_id
from
(select log_id, row_number() over (order by log_id) as rk, log_id - rk as flag
from development.log_ids)
order by 1;

--or
select min(log_id) as start_id, max(log_id) as end_id
from
(select log_id, row_number() over (order by log_id) as rk, log_id - rk as flag
from development.log_ids)
group by flag
order by 1;


-- x consecutive with group_id
begin;
drop table IF EXISTS development.logs;
commit;

begin;
CREATE TABLE development.logs (
   group_id varchar(1),
   log_id int
);
commit;

begin;
INSERT INTO development.logs VALUES
('A', 1),
('A', 2),
('A', 3),
('A', 5),
('A', 6),
('A', 8),
('A', 9),
('B', 11),
('C', 1),
('C', 2),
('C', 3);
commit;


SELECT * FROM development.logs;

with base as (
SELECT group_id, log_id, row_number() over (partition by group_id order by log_id) as rk, log_id - rk as flag
FROM development.logs)
select distinct group_id,
                min(log_id) over (partition by group_id, flag) as start_id,
                max(log_id) over (partition by group_id, flag) as end_id
from base
order by 1,2

--or

with base as (
SELECT group_id, log_id, row_number() over (partition by group_id order by log_id) as rk, log_id - rk as flag
FROM development.logs)
select group_id, min(log_id), max(log_id)
from base
group by group_id, flag
order by 1,2

--group_id min max
--A         1   3
--A         5   6
--A         8   9
--B        11   11
--C         1   3



-- 找到所有team的连胜场次
begin;
drop table IF EXISTS development.games;
commit;

begin;
CREATE TABLE development.games (
   home_team varchar(1),
   home_score int,
   away_team varchar(1),
   away_score int,
   game_date date
);
commit;

begin;
INSERT INTO development.games VALUES
('A', 3, 'B', 2, date('2021-08-01')),
('C', 0, 'D', 0, date('2021-08-02')),
('A', 3, 'D', 1, date('2021-08-12')),
('C', 1, 'B', 2, date('2021-08-12')),
('A', 0, 'C', 2, date('2021-08-22')),
('B', 4, 'D', 1, date('2021-08-23')),
('B', 0, 'A', 5, date('2021-09-01')),
('D', 0, 'C', 3, date('2021-09-02')),
('D', 0, 'A', 1, date('2021-09-12')),
('B', 0, 'C', 2, date('2021-09-12')),
('C', 0, 'A', 1, date('2021-09-22')),
('D', 2, 'B', 2, date('2021-09-23'));
commit;


SELECT * FROM development.games;


with wins AS (
SELECT home_team as team, home_score > away_score AS win, CAST(game_date AS VARCHAR) as game_date
FROM development.games
UNION ALL
SELECT away_team as team, home_score < away_score AS win, CAST(game_date AS VARCHAR) as game_date
FROM development.games),
ranks AS (
select team, win, rank() OVER (PARTITION BY team ORDER BY game_date) AS round
from wins),
flags AS (
SELECT team, round - row_number() over (partition by team order by round) as flag
FROM ranks
where win)
select team, count(*)
from flags
GROUP by team, flag


--team count
--A     2
--A     3
--B     2
--C     3


--or create a mid table
begin;

CREATE TABLE development.team_logs AS
with results AS (
SELECT home_team as team,
       CASE WHEN home_score > away_score THEN 'win' WHEN home_score < away_score THEN 'lose' ELSE 'draw' END AS result,
       CAST(game_date AS VARCHAR) as game_date
FROM development.games
UNION ALL
SELECT away_team as team,
       CASE WHEN home_score < away_score THEN 'win' WHEN home_score > away_score THEN 'lose' ELSE 'draw' END AS result,
       CAST(game_date AS VARCHAR) as game_date
FROM development.games)
select team, result, game_date, rank() OVER (PARTITION BY team ORDER BY game_date) AS round
from results;

commit;

SELECT * FROM development.team_logs

-- 找出有过3连胜的team
select team, listagg(result, '-') within group (order by game_date) as all_results
from development.team_logs
group by 1
having all_results like '%win-win-win%'

--team         all_results
--B   lose-win-win-lose-lose-draw
--C   draw-lose-win-win-win-lose
--D   draw-lose-lose-lose-lose-draw
--A   win-win-lose-win-win-win

--or

WITH flags AS (
SELECT team, round - row_number() over (partition by team order by round) as flag
FROM development.team_logs
where win)
select team
from flags
GROUP by team, flag
HAVING count(*) >= 3


