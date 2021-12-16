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