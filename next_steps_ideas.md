Quick note: All values of listings and sales should be normalized by the number of houses in the zipcode 
(proxy is population of zipcode) so that the numbers are comparable across zipcodes. 
Let’s call this parameter theta, which can be interpreted as the rate at which homes are 
listed/sold in the given zipcode. This motivates the use of Poisson regression models to model
the number of listings/sales, as there are many homes
 in a given zipcode, each of which has a small and independent probability of 
 listing/selling – i.e. X ~ Pois(population * theta)
 
At this point, our goal is to identity features that have predictive power of future 
real estate market conditions. Once we identify all the relevant features, we can either try
to combine them all into one model or use an ensemble of models, each of which focuses on
a particular area.

Categories of potentially relevant features:

a. Google trends data:

build model to predict new listings (using seller terms) and home sales 
(using buyer terms and seller terms) for entire Boston market area

can also look at Google trends on other regions since patterns likely apply across market areas


b. auto-regressive and seasonal nature of time series:

Time series on market area level using standard time series models (SARIMA, etc.). 
Goal is to determine effect on previous months, seasonal component, etc. 
Can also see if a hotter than average start to the season means increases 
the likelihood that the rest of the season will be hotter than average as well.

Once analysis is done for the entire market area, the same analysis can perhaps be
done on the county level to see if the insights inferred on the market level hold on the county level.

c. static zipcode-level characteristics:

Influence of zipcode-level factors: 
1. income
2. demographics (age and racial breakdown)
3. distance from Boston
4. fraction of workforce than commutes to Boston
5. fraction of workforce that takes public transportation to work
6. median house valuation
7. fraction of listings with price reduction
8. population density
9. crime date
10. quality of schools

(https://www.neighborhoodscout.com/) <- need a subscription
(https://www.city-data.com/) <- free

We can include these as predictors in a Poisson regression model, where one observation
contains all the attributes of a given zipcode and the number of new listings/sales
in a given month. To account for differences between months of the year and between 
years themselves, we can include dummy variables for each year and each month. 

d. spatial correlation between zipcodes

Relationship in rate of listings and sales between nearby zipcodes: 
http://pysal.org/notebooks/viz/splot/esda_morans_viz.html

https://en.wikipedia.org/wiki/Spatial_analysis

https://en.wikipedia.org/wiki/Moran%27s_I

http://darribas.org/gds_scipy16/ipynb_md/05_spatial_dynamics.html

(I think REX has geographical coordinates of each zipcode)

e. temporal-spatial analysis 

If it turns out that there both temporal and spatial relationships, then we can try
to combine them but looks complicated and an active area of research

http://www.cse.msu.edu/~zhaoxi35/paper/cikm2017.pdf

