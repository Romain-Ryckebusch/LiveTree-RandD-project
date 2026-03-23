Problem : obtained data is sometimes discontinous, if it is the case the model cannot work
we need to emulate data when there is a blank so that the model can still work, for instance use historical data
The model makes predictions at midnight, so if "the consumption of last seven days" is not full it will simply stop working
144 data points per day, one point per 10 minutes, 5 years of historical data 
problem : consumption and production depends on temperature, holidays, time of the year, etc. So choosing of historical data is not easy


We also can check the system failures to see how to make it actually work

Goals : make model resilient and find ways to keep some precision in the continuously functionning model


3 servers in the livetree network, capable of doing long calculations, lots of computing power, so we will get to know the database so we can analyse and understand the data behavior and maybe even identify the failures that they meet, and maybe link them to communication periods, we can link the failures in other data tables, it will be part of the first part

part of the objective is to develop a way to determine quality of prediceted and real data. Basially be able to determine of the quality changes when data has to rely on the backup



student in AI will be able to work on the quality evaluation, they should have the skills used for that