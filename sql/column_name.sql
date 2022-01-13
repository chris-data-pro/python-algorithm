-- Create development.input_table
-- because redshift doesn't support mix of INFORMATION_SCHEMA and normal table selects
-- This generates an equivalent table as

-- SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'development' AND TABLE_NAME = 'input_table'

SELECT COLUMN_NAME FROM "COLLAB_DB"."INFORMATION_SCHEMA"."COLUMNS"
WHERE TABLE_NAME = 'PM_WBX_SITE_USAGE_MLY_WITH_PREDICTIVE_FEATURES';

-- Output
COLUMN_NAME

MONTH_ID
PSTN_MEETING_PER_AH
SITE_MONTH_RANK
AGV_3M_CCA_MEETING_PER_AH
AUDIO_MEETING_SIZE
AUDIO_MEETING_PER_AH
AGV_3M_CMR_MEETING_PER_CMR_AU
AGV_3M_AUDIO_MEETING_SIZE

begin;
DROP TABLE IF EXISTS development.col_name_table;
commit;

begin;
create table development.col_name_table (
	column_name  char varying(50)
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

-- output table
column_name

Product_Page
Warranty_Page
Home_Page
