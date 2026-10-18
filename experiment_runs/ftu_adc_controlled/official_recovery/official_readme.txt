Title of the dataset: mm-Wave FMCW Radar Vital Sign Monitoring Dataset: Diverse Physiological Scenarios 



General Introduction:

This dataset originates from a study aimed at capturing vital signs, specifically heart rate and respiratory rate, using mm-Wave Frequency-Modulated Continuous Wave (FMCW) radar under various conditions. The study, conducted by the Faculty of Electrical Engineering, Mathematics, and Computer Science at the University of Twente, includes data from ten participants with diverse physiological profiles. The purpose is to provide a significant resource for developing and testing algorithms aimed at non-invasive health monitoring.

Methodological Information:

Data Collection Method: The dataset was generated through experiments using the Texas Instruments AWR1642BOOST radar evaluation module and Polar H10 heart rate sensors as a reference. Participants were involved in various scenarios, including distance, orientation, angle scenarios, and elevated heart rate conditions, to emulate real-world applications.

Instrumentation: mm-Wave FMCW radar (Texas Instruments AWR1642BOOST), Polar H10 heart rate sensor.

Software: Data recorded via the TI mmWave Studio for radar configuration and data acquisition. Matlab Runtime Engine V8.5.1 or higher and Code Composer Studio V7.1 or higher were used for data processing.

Quality Assurance: Data validation was performed against measurements from the Polar H10 sensor. The dataset underwent a preliminary validation to ensure reliability for scientific research.

Data Specific Information:

File Naming Convention: ParticipantID_ScenarioID_MeasurementID, where ParticipantID ranges from 1 to 10, ScenarioID refers to one of the four scenarios (Distance, Orientation, Angle, Elevated HR), and MeasurementID is a unique identifier for each measurement.

File Format: Raw data files are provided in binary format (.bin) for radar data and CSV format for heart rate reference values.

Units of Measurement: Heart rate in beats per minute (bpm), respiratory rate in breaths per minute (bpm).

Sharing and Access Information:

License: CC BY-NC-SA. This dataset is shared under the Creative Commons Attribution-NonCommercial-ShareAlike license, allowing for adaptation and sharing under similar terms, without commercial use.

Contact Information: For inquiries regarding the dataset, please contact:
Ehsan Sadeghi, e.sadeghi@utwente.nl
Faculty of Electrical Engineering, Mathematics, and Computer Science
University of Twente, 7522 NB Enschede, The Netherlands

Additional Notes:
This dataset accompanies the research article titled "Comprehensive mm-Wave FMCW Radar Dataset for Vital Sign Monitoring: Embracing Extreme Physiological Scenarios" and is intended for use in academic and research settings. Users are encouraged to reference the original article when using this dataset in their work.

