CREATE DATABASE diabetes_readmission_db;
USE diabetes_readmission_db;
DROP TABLE patient_data;

CREATE TABLE patient_data (
    encounter_id INT PRIMARY KEY,
    patient_nbr INT,
    race VARCHAR(50),
    gender VARCHAR(20),
    age VARCHAR(20),
    admission_type_id VARCHAR(50),
    discharge_disposition_id VARCHAR(100),
    admission_source_id VARCHAR(50),
    time_in_hospital INT,
    payer_code VARCHAR(50),
    num_lab_procedures INT,
    num_procedures INT,
    num_medications INT,
    diag_1 VARCHAR(50),
    diag_2 VARCHAR(50),
    diag_3 VARCHAR(50),
    number_diagnoses DECIMAL(3,1),
    insulin VARCHAR(20),
    change_meds VARCHAR(10),
    diabetesMed VARCHAR(10),
    readmitted VARCHAR(10),
    total_visits INT
);
