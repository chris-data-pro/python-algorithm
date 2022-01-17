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


select order_day, listagg(order_id::varchar, ';') within group (order by quantity desc) as order_list
from development.older_table
group by 1;

--01-JUL-2011	O4;O3;O1;O2
--02-JUL-2011	O6;O5;O10;O7;O9;O8