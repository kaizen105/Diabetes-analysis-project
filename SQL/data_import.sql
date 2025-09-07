SET GLOBAL local_infile = 'ON';
SHOW VARIABLES LIKE 'secure_file_priv';
USE diabetes_readmission_db;
LOAD DATA LOCAL INFILE
'C:\\diabetes_analysis_project\\datasets\\diabetic_data_clean.csv' 
INTO TABLE patient_data
CHARACTER SET 'utf8'
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES;