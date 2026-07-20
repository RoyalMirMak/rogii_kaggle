Overview
Drilling a horizontal well is like navigating underground without a map. The path forward runs through layers of rock you can’t see.

Build models to predict the geology along a horizontal wellbore. Your work will help automate and improve drilling operations in the oil and gas industry.

Start

2 months ago
Close

17 days to go
Merger & Entry
Description
Roughly 10,000 horizontal wells are drilled worldwide every year, yet much of the drilling process still relies on manual interpretation by experts. These operations require immense technical precision, where even small deviations from the target zone can lead to significant resource waste. If the well drifts into less favorable geology, it results in inefficient energy recovery and may require additional corrective measures that increase the overall environmental footprint of the site.

Interpreting the subsurface is challenging because direct measurements are inherently limited. Data from wells, seismic surveys, and logging tools only show part of the picture. Rock layers start out stacked like a layer cake, but can bend or break along faults, making it hard to know exactly where the drill bit sits within the formation. Geologists and engineers analyze incoming data to steer the well, but current analytical tools often struggle to match the nuance of expert interpretation.

In this competition, you’ll develop machine learning models that predict the geology encountered along a horizontal wellbore. Your models should identify favorable layers from drilling data and help guide well placement more accurately during operations.

Your solution could help reduce resource waste by minimizing redundant drilling, improve operational safety by better predicting geological hazards, and move the industry toward automated systems that make faster, more consistent, and data-driven decisions.

A clearer map beneath the surface could make every meter count.

Evaluation
Submissions are scored on the root mean squared error. RMSE is defined as:


where 
 is the predicted value, 
 is the original value, and 
 is the number of rows in the test data.

Submission File
For each row in the test set, you must predict the value of the target tvt as described on the data tab, each on a separate row in the submission file. The file should contain a header and have the following format:

id,tvt
000d7d20_1442,0.0
000d7d20_1443,0.0
000d7d20_1444,0.0
000d7d20_1445,0.0
...
Working Note Award (optional)
Eligibility: teams must be in the Medal Zone in the public leaderboard to be eligible for a Working Note Award.

The Working Note will be evaluated using the following criteria:

Criteria	Description
1. Breadth and Depth of Exploration	We value thorough exploration of genuinely different approaches, including both successful and unsuccessful ones. Approaches are considered distinct only when they differ in a meaningful way, such as the feature set, modeling strategy, method for handling spurious correlations, estimation of incidence angle, or other core methodological choices. Hyperparameter tuning, window-size adjustments, and minor variations of the same idea will be treated as a single approach.

Each approach should be documented with:
- The underlying idea and motivation;
- Validation results;
- Conclusions and lessons learned, including why the approach succeeded or failed.

A smaller number of deeply analyzed approaches is preferred over a long list of superficial experiments. Negative results are valued equally when accompanied by thoughtful analysis.
2. Insights About the Data and Wells	Participants should share the most important observations they made about the data throughout the competition. This may include differences in behavior across wells, expected and unexpected findings, insights gained from the public dataset, and the extent to which the public data helped guide development.

If different methods were applied to different wells, explain how those decisions were made and how you determined which approach was most appropriate for each well.
3. Physical Meaningfulness of the Solution	Describe the extent to which your final solution represents a physically meaningful interpretation of the underlying data rather than simply the highest-scoring combination of models, averaging schemes, or ensembles.

We encourage participants to reflect on where they draw the line between discovering genuine relationships in the data and optimizing specifically for the evaluation metric. Explain how your final method balances physical plausibility, robustness, and predictive performance.
4. Contribution of Individual Ideas	Clearly demonstrate how each major idea, feature, model component, or methodological decision contributed to the final result. Whenever possible, quantify the impact of individual components on validation performance and show how improvements accumulated throughout the development process.
5. Uncertainty Estimation	Describe whether and how your method estimates its own confidence. Strong submissions should discuss where predictions are likely to be reliable, where the method may be uncertain or prone to error, and how this uncertainty is quantified and communicated.
Timeline
May 5, 2026 - Start Date.
July 6, 2026 (optional) - Deadline to submit working notes to be considered for the Working Note Award.
July 29, 2026 - Entry Deadline. You must accept the competition rules before this date in order to compete.
July 29, 2026 - Team Merger Deadline. This is the last day participants may join or merge teams.
August 5, 2026 - Final Submission Deadline.
All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

Code Requirements


Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

CPU Notebook <= 9 hours run-time
GPU Notebook <= 9 hours run-time
Internet access disabled
Freely & publicly available external data is allowed, including pre-trained models
Submission file must be named submission.csv
Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.

Dataset Description
The competition data comprises horizontal well trajectories and vertical reference logs (Typewells) used for geological prediction. Your goal is to predict the TVT (True Vertical Thickness) for the evaluation zone of each horizontal well.

The data is organized into train/ and test/ directories, where each well is identified by a unique 8-character hash (e.g., 015fe0d2).

File and Field Information
train/ Contains the training data. Each well has three associated files:

{WELLNAME}__horizontal_well.csv - Trajectory, geological surfaces, and log data.
WELLNAME - unique identifier for the well.
MD - Measured Depth (ft): The total length of the wellbore from the surface.
X - Easting (ft): Spatial coordinate in the horizontal plane.
Y - Northing (ft): Spatial coordinate in the horizontal plane.
Z - True Vertical Depth (ft): The vertical distance below sea level.
ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA - Predicted depth of various geological formations (Training only).
TVT - True Vertical Thickness (ft): The manually interpreted geological position for each 1 ft of the lateral well. This is the target variable (Training only).
GR - Gamma Ray (API): Log measuring natural radioactivity of the rock.
TVT_input - Input Target (ft): A copy of TVT provided as a feature. This column contains NaN values for the evaluation zone.
{WELLNAME}__typewell.csv - Vertical reference log for geological correlation.
TVT - Vertical Depth Index (ft): Primary depth reference for the vertical log. Corresponds to TVT (geological position) of the associated horizontal well.
GR - Gamma Ray (API): The vertical Gamma Ray signature used for correlation.
Geology - Formation Label: Categorical label indicating the geological unit (e.g., EGFDL, BUDA).
{WELLNAME}.png - Visualization of the well path and geological cross-section.
test/ Contains the evaluation data for about 200 wells. Each well has two associated files:

{WELLNAME}__horizontal_well.csv - Trajectory and log data. In these files, the TVT target is hidden (replaced with NaN) in the evaluation zone.
{WELLNAME}__typewell.csv - Vertical reference log for the test well.
Note that the test/ folder visible here contains only a few instances from the training set as example data to help you author your submissions. When your submission is rerun on the hidden test set, these will be replaced with the actual test data.

sample_submission.csv - A sample submission file in the correct format.
id - A unique identifier for each prediction point, formatted as {WELLNAME}_{row_index} (e.g., 015fe0d2_1654).
tvt - Your predicted True Vertical Thickness (ft).