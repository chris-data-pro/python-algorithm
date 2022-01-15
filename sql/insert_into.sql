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


INSERT INTO subscribers (email)
VALUES('john.doe@gmail.com'),
      ('jane.smith@ibm.com');


INSERT IGNORE INTO subscribers(email)
VALUES('john.doe@gmail.com'),
      ('jane.smith@ibm.com');




