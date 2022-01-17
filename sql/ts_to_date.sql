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


-- PostgreSQL
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


SELECT game_date，date(game_date + interval '1 day') as next_date,
       EXTRACT(YEAR FROM game_date), EXTRACT(MONTH FROM game_date), EXTRACT(DAY FROM game_date),
       EXTRACT(DOW FROM game_date), EXTRACT(DOY FROM game_date), EXTRACT(QUARTER FROM game_date)
FROM development.games;


-- assignment_st type: timestamp
SELECT assignment_st,
       EXTRACT(YEAR FROM assignment_st), EXTRACT(MONTH FROM assignment_st), EXTRACT(DAY FROM assignment_st),
       EXTRACT(DOW FROM assignment_st), EXTRACT(DOY FROM assignment_st), EXTRACT(QUARTER FROM assignment_st),
       EXTRACT(HOUR FROM assignment_st), EXTRACT(MINUTE FROM assignment_st), EXTRACT(SECOND FROM assignment_st)
FROM analytics_rollup.experiment_users
LIMIT 3


