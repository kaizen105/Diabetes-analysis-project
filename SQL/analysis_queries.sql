USE diabetes_readmission_db;
SHOW TABLES;
DESCRIBE patient_data;
select * from patient_data limit 10;
SELECT COUNT(*) FROM patient_data;
SELECT DISTINCT gender FROM patient_data;

#1. How many unique patients and total encounters are in the dataset?
SELECT count(distinct encounter_id) AS unique_encounters,count(distinct patient_nbr) AS unique_patients from patient_data;

#2. What is the overall readmission rate (any readmit vs NO)?
select readmitted , count(*) AS readmitted_count,
concat(round(count(*)*100/(select count(*) from patient_data),2),"%") as readmitted_percentage
from patient_data 
group by readmitted;

#3. How does readmission rate vary by age group?
SELECT age,
concat(round(SUM(CASE WHEN readmitted='<30' THEN 1 ELSE 0 END)*100/count(*),2),"%") AS readmit_rate_under_30,
concat(round(SUM(CASE WHEN readmitted='>30' THEN 1 ELSE 0 END)*100/count(*),2),"%") AS readmit_rate_over_30,
concat(round(SUM(CASE WHEN readmitted='NO' THEN 1 ELSE 0 END)*100/count(*),2),"%") AS no_readmit_rate,
COUNT(*) AS total_encounters
FROM patient_data
GROUP BY age
ORDER BY CAST(SUBSTRING_INDEX(SUBSTRING_INDEX(age,'[',-1), '-', 1) AS UNSIGNED) ASC;

#4. How does readmission rate vary by gender?
SELECT gender,
concat(round(SUM(CASE WHEN readmitted='<30' THEN 1 ELSE 0 END)*100/count(*),2),"%") AS readmit_rate_under_30,
concat(round(SUM(CASE WHEN readmitted='>30' THEN 1 ELSE 0 END)*100/count(*),2),"%") AS readmit_rate_over_30,
concat(round(SUM(CASE WHEN readmitted='NO' THEN 1 ELSE 0 END)*100/count(*),2),"%") AS no_readmit_rate,
COUNT(*) AS total_encounters
FROM patient_data
GROUP BY gender;

#5. Are patients using insulin more likely to be readmitted?
select insulin,
SUM(CASE WHEN readmitted='>30' or readmitted="NO" THEN 1 ELSE 0 END) AS readmitted_not,
SUM(CASE WHEN readmitted='<30' THEN 1 ELSE 0 END) AS readmitted,
count(*) AS total_encounters,
concat(round(SUM(CASE WHEN readmitted='>30' or readmitted="NO" THEN 1 ELSE 0 END)*100/count(*),2),"%") AS readmitted_not_rate,
concat(round(SUM(CASE WHEN readmitted='<30' THEN 1 ELSE 0 END)*100/count(*),2),"%") AS readmitted_rate
from patient_data
group by insulin;

#6. What are the top 10 primary diagnoses (`diag_1`) by frequency?
select diag_1 , count(*) AS total_encounters
from patient_data
group by diag_1
order by total_encounters desc limit 10; 

#7.  Which diagnoses are most associated with readmission?
SELECT diag_code,
COUNT(CASE WHEN readmitted='<30' THEN encounter_id END) AS total_readmitted,
COUNT(DISTINCT encounter_id) AS total_patients,
CONCAT(ROUND(COUNT(CASE WHEN readmitted='<30' THEN encounter_id END)*100.0 / COUNT(DISTINCT encounter_id), 2), '%') AS readmit_rate
FROM (SELECT encounter_id, readmitted, diag_1 AS diag_code FROM patient_data
UNION ALL SELECT encounter_id, readmitted, diag_2 AS diag_code FROM patient_data
UNION ALL SELECT encounter_id, readmitted, diag_3 AS diag_code FROM patient_data) t
WHERE diag_code IS NOT NULL
GROUP BY diag_code
ORDER BY readmit_rate DESC;

#8. What is the average time in hospital by primary diagnosis (`diag_1`)?
SELECT diag_1,ROUND(AVG(time_in_hospital), 2) AS avg_time_in_hospital_days,COUNT(*) AS total_encounters
FROM patient_data
GROUP BY diag_1
ORDER BY total_encounters DESC
LIMIT 10;

#9. How do admission sources (`admission_source_id`) relate to discharge disposition?
SELECT admission_source_id,
SUM(CASE WHEN discharge_disposition_id = 'Discharged to home' THEN 1 ELSE 0 END) AS discharged_home,
SUM(CASE WHEN discharge_disposition_id = 'Transferred to another facility' THEN 1 ELSE 0 END) AS transferred_facility,
SUM(CASE WHEN discharge_disposition_id = 'Left AMA' THEN 1 ELSE 0 END) AS left_AMA,
SUM(CASE WHEN discharge_disposition_id = 'Still patient/referred to this institution' THEN 1 ELSE 0 END) AS still_patient,
SUM(CASE WHEN discharge_disposition_id = 'Not Available' THEN 1 ELSE 0 END) AS not_available,
COUNT(*) AS total_encounters
FROM patient_data
GROUP BY admission_source_id
ORDER BY total_encounters DESC;

#10. What are the most common diagnoses (`diag_1`) per race group?
WITH diag_counts AS (
    SELECT 
        race,
        diag_1,
        COUNT(*) AS total_encounters
    FROM patient_data
    WHERE diag_1 IS NOT NULL
    GROUP BY race, diag_1
),
total_per_race AS (
    SELECT race, SUM(total_encounters) AS race_total
    FROM diag_counts
    GROUP BY race
),
ranked_diags AS (
    SELECT
        d.race,
        d.diag_1,
        d.total_encounters,
        t.race_total,
        RANK() OVER (PARTITION BY d.race ORDER BY d.total_encounters DESC) AS rnk
    FROM diag_counts d
    JOIN total_per_race t ON d.race = t.race
)
SELECT 
    race,
    diag_1 AS diag_1_most_common,
    total_encounters,
    CONCAT(ROUND(total_encounters*100.0 / race_total,2), '%') AS percentage
FROM ranked_diags
WHERE rnk = 1
ORDER BY race;
