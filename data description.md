external data description:

numer_sta: Station number (integer).
date: Date of the weather data (object, possibly needs to be converted to datetime).
pmer: Atmospheric pressure at sea level (integer).
tend: Atmospheric pressure tendency (integer).
cod_tend: Code for atmospheric pressure tendency (integer).
dd: Wind direction (integer).
ff: Wind speed (float).
t: Temperature (float).
td: Dew point temperature (float).
u: Humidity (integer).
vv: Horizontal visibility (integer).
ww: Weather phenomenon (integer).
w1 and w2: Additional weather phenomena (float).
n: Cloud cover (float).
nbas: Cloud base (float).
hbas: Cloud height (float).
cl, cm, ch: Cloud cover for different cloud layers (float).
pres: Atmospheric pressure (integer).
tend24: Atmospheric pressure tendency over 24 hours (float).
tn12, tn24, tx12, tx24: Minimum and maximum temperatures over 12 and 24 hours (float).
tminsol: Minimum soil temperature (float).
raf10, rafper: Wind gusts (float).
per: Precipitation type (integer).
etat_sol: Ground state (float).
ht_neige: Snow depth (float).
ssfrai: Freezing rain (float).
perssfrai: Freezing rain duration (float).
rr1, rr3, rr6, rr12, rr24: Precipitation over different time intervals (float).
phenspe1 to phenspe4: Phenomena species (all have zero non-null values).
nnuage1 to nnuage4: Cloud cover for different cloud layers (float).
ctype1 to ctype4: Cloud type for different cloud layers (float).
hnuage1 to hnuage4: Cloud height for different cloud layers (float).

train_parquest
counter_id: Category for the counter ID.
counter_name: Category for the counter name.
site_id: Integer representing the site ID.
site_name: Category for the site name.
bike_count: Float representing the count of cyclists.
date: Datetime column representing the date of the data.
counter_installation_date: Datetime column representing the installation date of the counter.
counter_technical_id: Category for the technical ID of the counter.
latitude: Float representing the latitude of the counter location.
longitude: Float representing the longitude of the counter location.
log_bike_count: Float representing the logarithm of the bike count.






w1, w2: replace nan with 3
n: NaN replace with 0
n: replace the 101. with 100.
nbas: replace nan with the mode of the day
hbas: replace with mode of the day
cl, cm, ch: replace with mode of the day
tend24: replace nan with mean of the day
raf10: replace with mean of the day
etat_sol: replace with mode of the day
ht_neige: replace negative with 0 and nan with mode of the day
ssfrai: replace nan with mode of the day 
persfrai: drop column
rr1 to rr24: replace nan with mean of the day
hnuage1: replace nan with mean of the day
nnuage1: replace nan with mode of the day
ctype1: replace nan with mode of the day

