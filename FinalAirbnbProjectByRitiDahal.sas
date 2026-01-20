/*-------------------------------------------------------------*/
/* DSCI 519 – Advanced Business Analytics Modeling             */
/* Airbnb Price Prediction – Seattle, Washington               */
/* Submitted By: Riti Dahal                                   */
/*-------------------------------------------------------------*/


/* Step 1: Importing Data */

proc import datafile='/home/u64128366/Data Analysis Term Project/Seattle Data Airbnb.xlsx'
            dbms=xlsx 
            out=work.seattle
            replace;
    getnames=yes;
run;

/* Review data structure */
proc contents data=work.seattle;
run;


/* Step 2: Exploring Dependent Variable – Price */

proc univariate data=work.seattle;
    var price;
    histogram;
    title "Distribution of Price Variable";
run;

/* Observations:
   i.   There are 635 missing values in Price.
   ii.  The mean and median differ due to outliers.
   iii. The 99% cutoff = $1,295 and the maximum = $50,034.
   iv.  Histogram shows a highly right-skewed distribution.
*/


/* Step 3: Removing Extreme Outliers and Create Log Price */

data work.seattle_clean;
    set work.seattle;
    where 44 <= price <= 1295;
    price_log = log(price);
run;


/* Step 4: Checking Normality of Log Price Variable */

proc univariate data=work.seattle_clean;
    var price_log;
    histogram / normal;
    title "Distribution of Log Price";
run;


/* Step 5: Initial Numeric Data Exploration */

proc means data=work.seattle_clean (keep=_numeric_) 
            n nmiss min max mean median std;
    title "Summary Statistics for Numeric Variables";
run;

/* Selecting key numeric predictors for further review */
%let numvar = accommodates bedrooms beds bathrooms minimum_nights
              maximum_nights availability_30 availability_60 
              availability_90 availability_365 host_listings_count;

/*Exploring Selected Numeric Variables*/
proc means data=work.seattle_clean (keep=&numvar)
            n nmiss min max mean median std;
    title "Missing Value Check for Key Numeric Variables";
run;


/* Step 6: Handling Missing Values for host_listings_count */

proc stdize data=work.seattle_clean 
            out=work.seattle_clean 
            reponly
            method=mean;
    var host_listings_count;
run;

/* Step 7: Relationships Between accommodates, bedrooms, beds */

proc sgplot data=work.seattle_clean;
    scatter x=accommodates y=bedrooms 
            / markerattrs=(symbol=circlefilled color=blue);
    reg x=accommodates y=bedrooms 
        / lineattrs=(color=red thickness=2);
    title "Scatter Plot: Accommodates vs. Bedrooms";
run;

proc sgplot data=work.seattle_clean;
    scatter x=accommodates y=beds 
            / markerattrs=(symbol=circlefilled color=blue);
    reg x=accommodates y=beds 
        / lineattrs=(color=red thickness=2);
    title "Scatter Plot: Accommodates vs. Beds";
run;


/* Step 8: Group-Mean Imputation for Bedrooms & Beds */

proc means data=work.seattle_clean noprint;
    class accommodates;
    var bedrooms beds;
    output out=bed_bedroom_means (drop=_type_ _freq_) 
           mean=mean_bedrooms mean_beds;
run;

proc sort data=work.seattle_clean; by accommodates; run;
proc sort data=bed_bedroom_means; by accommodates; run;

data work.seattle_imputed;
    merge work.seattle_clean (in=a) bed_bedroom_means (in=b);
    by accommodates;
    if missing(bedrooms) then bedrooms = mean_bedrooms;
    if missing(beds) then beds = mean_beds;
    drop mean_bedrooms mean_beds;
run;

/*Checking for levels*/
proc freq data=work.seattle_imputed nlevels;
    tables beds bedrooms bathrooms / noprint;
run;


/* Step 9: Grouping as a part of Feature Engineering*/
/*Categorize Bathrooms */
data work.seattle_bathcat;
    set work.seattle_imputed;
	length bath_cat $20;
    if bathrooms <= 1 then bath_cat = "0-1 Bathroom";
    else if 1 < bathrooms <= 2 then bath_cat = "1-2 Bathroom";
    else if bathrooms > 2 then bath_cat = "More than 2 Bathroom";
run;

/* Checking new distribution for Bathrooms */
proc freq data=work.seattle_bathcat;
    tables bath_cat / nocum nopercent;
    title "Frequency Distribution of Bathroom Categories";
run;

/*Categorize Bedrooms*/
data work.seattle_bedcat;
    set work.seattle_bathcat; 
    length bed_cat $20;           

    if bedrooms <= 1 then bed_cat = "0-1 Bedroom";
    else if 1 < bedrooms <= 2 then bed_cat = "1-2 Bedroom";
    else if bedrooms > 2 then bed_cat = "More than 2 Bedroom";
run;

/* Checking new distribution for Bedrooms */
proc freq data=work.seattle_bedcat;
    tables bed_cat / nocum nopercent;
    title "Frequency Distribution of Bedroom Categories";
run;

/*Categorize Beds*/
data work.seattle_beds;
    set work.seattle_bedcat; 
    length beds_cat $20;           

    if beds <= 1 then beds_cat = "0-1 Bed";
    else if 1 < beds <= 2 then beds_cat = "1-2 Beds";
    else if beds > 2 then beds_cat = "More than 2 Beds";
run;


/* Checking new distribution for Beds */
proc freq data=work.seattle_beds;
    tables beds_cat / nocum nopercent;
    title "Frequency Distribution of Bed Categories";
run;


/* Step 9: Correlation Matrix for Numeric Variables */
%let numvar = accommodates minimum_nights maximum_nights 
               availability_30 availability_60 availability_90 availability_365 
               host_listings_count;

proc corr data=work.seattle_beds;
    var &numvar;
    title "Correlation Matrix of Numeric Variables";
run;

/*Checking for Multicollinearity Using VIF */
proc reg data=work.seattle_beds plots=all;
    model price_log = &numvar
                      /vif collin;
    title "VIF and Collinearity for Numeric Predictors";
run;
quit;


/* Step 10: Standardizing Numeric Variables */
proc stdize data=work.seattle_beds
            out=work.seattle_standardized
            method=std;               /* Standardizes to mean=0, std=1 */
    var &numvar ;
run;


/*Verifying Standardization Results */
proc means data=work.seattle_standardized n mean std min max;
    var &numvar;
    title "Verification of Standardized Numeric Variables";
run;


/* Step 11: Exploring Categorical Variables */
%let catvar = property_type room_type neighbourhood_cleansed instant_bookable;

proc freq data=work.seattle_standardized (keep=&catvar);
    tables &catvar / nocum nopercent;
    title "Frequency Distribution of Key Categorical Variables";
run;


/*Handle Missing Values for Categorical Variables*/
data work.seattle_category;
    set work.seattle_standardized;
    if missing(property_type) then delete;
    if missing(room_type) then delete;
    if missing(neighbourhood_cleansed) then delete;
run;

/* Checking if there is any missing values  */
proc freq data=work.seattle_category;
    tables property_type room_type neighbourhood_cleansed / missing;
    title "Missing Value Check for Key Categorical Variables";
run;

/*Categorical Variable Grouping */
data work.seattle_final;
    set work.seattle_category;

    /* Property Type Grouping*/
    if property_type in ("Entire condo","Entire apartment","Entire serviced apartment",
                         "Entire rental unit","Entire loft") then 
        property_group = "Apartment/Condo";

    else if property_type in ("Entire home","Entire townhouse","Entire bungalow",
                              "Entire cottage","Entire guest suite","Entire guesthouse",
                              "Entire vacation home","Entire villa","Farm stay") then 
        property_group = "House";

    else if property_type in ("Private room","Private room in home",
                              "Private room in guesthouse","Private room in condo",
                              "Private room in townhouse","Private room in bed and breakfast",
                              "Shared room in home","Shared room in hostel",
                              "Shared room in townhouse") then 
        property_group = "Room";

    else property_group = "Other";
    
    /*Room Type Grouping */
    if room_type = "Entire home/apt" then room_group = "Entire Unit";
	else room_group = "Room";


    /* Standardize text to ensure matching works correctly*/
   neighbourhood_cleansed = strip(propcase(compress(neighbourhood_cleansed)));
    length neighborhood_group $25.;

    /* Downtown & Tourist Hub */
    if neighbourhood_cleansed in (
        "Belltown","Central Business District","South Lake Union",
        "Pioneer Square","Pike-Market","First Hill","Broadway",
        "Stevens","Minor","East Queen Anne","West Queen Anne",
        "North Queen Anne","Lower Queen Anne","Interbay"
    ) then neighborhood_group = "Downtown & Tourist Hub";

    /* Residential North */
    else if neighbourhood_cleansed in (
        "Ballard","Adams","Green Lake","Phinney Ridge","Fremont",
        "Ravenna","Bryant","Maple Leaf","Greenwood","Wedgwood",
        "Laurelhurst","Windermere","View Ridge","North Beach/Blue Ridge",
        "Crown Hill","Broadview","Bitter Lake","Haller Lake"
    ) then neighborhood_group = "Residential North";

    /* Residential South & West Seattle */
    else if neighbourhood_cleansed in (
        "North Beacon Hill","Mid-Beacon Hill","South Beacon Hill",
        "Columbia City","Rainier Beach","Rainier View","Brighton","Dunlap",
        "Seward Park","Fauntleroy","Genesee","Gatewood","High Point",
        "Highland Park","North Admiral","Alki","Arbor Heights",
        "South Delridge","North Delridge","South Park"
    ) then neighborhood_group = "Residential South & West Seattle";

    else neighborhood_group = "Other";
    
    /*Creating dummy variable for instant bookable*/
     if instant_bookable = 'f' then instant_group = 1; 
	else instant_group= 0; 
run;

proc freq data=work.seattle_final;
    tables property_group room_group neighborhood_group instant_group / nocum nopercent;
    title "Categorization of Categorical Variables";
run;




/* Step 12: Data Partition*/
/*Spliting the data into train and test sets at an 80/20 split*/
proc surveyselect data=work.seattle_final
                  samprate=0.20 
                  seed=42 
                  out=work.full 
                  outall 
                  method=srs;
run;

data train test;
    set work.full;
    if selected = 0 then output work.train; 
    else output work.test;
    drop selected;
run;


/* Step 13: Model Building - LASSO Regression */

/* Define variables for LASSO */
%let lasso_num = accommodates minimum_nights maximum_nights 
					availability_365 host_listings_count;
    
%let lasso_cat = property_group room_group neighborhood_group
				 bath_cat bed_cat beds_cat instant_group;


ods graphics on;

proc glmselect data=work.train 
               outdesign(addinputvars)=work.reg_design
               plots(stepaxis=normb)=all;
    class &lasso_cat;
    model price_log = &lasso_num &lasso_cat / selection=lasso(stop=none choose=sbc);
    output out=work.train_score predicted=p_price_log residual=r_price_log;
    score data=work.test predicted residual out=work.test_score;
run;


/* Calculate performance metrics for TRAIN dataset */
data train_measure;
    set train_score;  
    residual_error = price_log - p_price_log;
    squared_error = residual_error * residual_error;
    trans_price = exp(price_log);
    trans_error = exp(residual_error);
    squared_prediction = p_price_log * p_price_log;
    trans_predicted_price = exp(p_price_log);  
    true_error = trans_price - trans_predicted_price;
    keep residual_error squared_error trans_price trans_error 
         squared_prediction trans_predicted_price true_error;  
run;

proc summary data=train_measure;
    var squared_error trans_error true_error;
    output out=train_sum_out sum=;
run;

data train_rmse_sum;
    set train_sum_out;
    rmse = sqrt(squared_error/_freq_);
    trans_rmse = sqrt(trans_error/_freq_);  
    true_rmse = sqrt(true_error/_freq_);
run;

proc print data=train_rmse_sum; 
    title 'Training Set Performance Metrics';
run;


/* Calculate performance metrics for TEST dataset */
data test_measure;
    set test_score;  
    residual_error = price_log - p_price_log;
    squared_error = residual_error * residual_error;
    trans_price = exp(price_log);
    trans_error = exp(residual_error);
    squared_prediction = p_price_log * p_price_log;
    trans_predicted_price = exp(p_price_log);  
    true_error = trans_price - trans_predicted_price;
    keep residual_error squared_error trans_price trans_error 
         squared_prediction trans_predicted_price true_error;  
run;

proc summary data=test_measure;
    var squared_error trans_error true_error;
    output out=test_sum_out sum=;
run;

data test_rmse_sum;
    set test_sum_out;
    rmse = sqrt(squared_error/_freq_);
    trans_rmse = sqrt(trans_error/_freq_);  
    true_rmse = sqrt(true_error/_freq_);
run;

proc print data=test_rmse_sum; 
    title 'Test Set Performance Metrics';
run;


/* Step 14: Regression Assumption*/
/* Create residual variable */
data train_score_diag;
    set train_score;
    residual = price_log - p_price_log;
run;

/* Linearity: Observed vs Predicted */
proc sgplot data=train_score;
    scatter x=p_price_log y=price_log;
    reg x=p_price_log y=price_log / lineattrs=(color=red);
    xaxis label='Predicted Values';
    yaxis label='Observed Values';
    title 'Linearity Check: Observed vs Predicted';
run;

/*Normality: Histogram of Residuals */
proc sgplot data=train_score_diag;
    histogram residual / binwidth=0.1;
    density residual / type=kernel;
    title 'Normality Check: Histogram of Residuals';
run;

/* Normality: Q-Q Plot */
proc univariate data=train_score_diag plots;
    var residual;
    qqplot / normal(mu=est sigma=est);
    title 'Normality Check: Q-Q Plot';
run;

/* Homoscedasticity: Residuals vs Predicted */
proc sgplot data=train_score_diag;
    scatter x=p_price_log y=residual;
    refline 0 / axis=y lineattrs=(color=red pattern=dash);
    xaxis label='Predicted Values';
    yaxis label='Residuals';
    title 'Homoscedasticity: Residuals vs Predicted';
run;

/*Independence: Standardized Residuals */
proc sgplot data=train_score_diag;
    scatter x=p_price_log y=residual;
    refline -1 / axis=y lineattrs=(color=black);
    refline 1 / axis=y lineattrs=(color=black);
    refline 0 / axis=y lineattrs=(color=red pattern=dash);
    xaxis label='Predicted Values';
    yaxis label='Residuals';
    title 'Independence: Standardized Residuals';
run;

/* Model R-Squared - Training Data */
proc reg data=train_score;
    model price_log = p_price_log;
    title "LASSO Model - R-Squared (Training Data)";
run;
quit;

/* Model R-Squared - Test Data */
proc reg data=test_score;
    model price_log = p_price_log;
    title "LASSO Model - R-Squared (Test Data)";
run;
quit;


/* Step 15: Decision Tree Model*/

/* Define path for saving model files */
%let path = /home/u64128366/Data Analysis Term Project;

title "Decision Tree Model - Full Training Set";
ods graphics on;

proc hpsplit data=work.train seed=42;
    class &lasso_cat;
    model price_log = &lasso_num &lasso_cat;
    prune costcomplexity;
    output out=tree_train_scored;
    code file="&path./tree_score.sas";
run;

/* Score TRAINING data with decision tree */
data tree_train_eval;
    set tree_train_scored;
    residual_sq = (p_price_log - price_log)**2;
    trans_price = exp(price_log);
    trans_predicted_price = exp(p_price_log);
    true_error_sq = (trans_price - trans_predicted_price)**2;
run;

/* Score TEST data with decision tree */
data tree_test_scored;
    set work.test;
    %include "&path./tree_score.sas";
run;

data tree_test_eval;
    set tree_test_scored;
    residual_sq = (p_price_log - price_log)**2;
    trans_price = exp(price_log);
    trans_predicted_price = exp(p_price_log);
    true_error_sq = (trans_price - trans_predicted_price)**2;
run;

/* Calculate RMSE for Decision Tree */
proc sql noprint;
    /* Training RMSE */
    select sqrt(mean(residual_sq)) as rmse_train,
           sqrt(mean(true_error_sq)) as true_rmse_train
    into :rmse_train_tree, :true_rmse_train_tree
    from tree_train_eval;
    
    /* Test RMSE */
    select sqrt(mean(residual_sq)) as rmse_test,
           sqrt(mean(true_error_sq)) as true_rmse_test
    into :rmse_test_tree, :true_rmse_test_tree
    from tree_test_eval;
quit;

%put ========================================;
%put Decision Tree Results:;
%put   Training RMSE: &rmse_train_tree;
%put   Training True RMSE: &true_rmse_train_tree;
%put   Test RMSE: &rmse_test_tree;
%put   Test True RMSE: &true_rmse_test_tree;
%put ========================================;

/* Print results in a table */
data tree_rmse_results;
    length model $30 dataset $10;
    model = "Decision Tree";
    
    dataset = "Training";
    rmse = input("&rmse_train_tree", 8.4);
    true_rmse = input("&true_rmse_train_tree", 8.4);
    output;
    
    dataset = "Test";
    rmse = input("&rmse_test_tree", 8.4);
    true_rmse = input("&true_rmse_test_tree", 8.4);
    output;
run;

proc print data=tree_rmse_results noobs;
    title "Decision Tree - RMSE Summary";
    var model dataset rmse true_rmse;
    format rmse true_rmse 8.4;
run;



/* Step 16: Random Forest Model*/
title "Random Forest Model - Full Training Set";

proc hpforest data=work.train
    maxtrees=300 
    vars_to_try=7
    seed=42
    maxdepth=15 
    leafsize=10;
    target price_log / level=interval;
    input &lasso_num / level=interval;
    input &lasso_cat / level=nominal;
    ods output fitstatistics=rf_train_fit;
    save file="&path./rfmodel.bin";
run;

/* Score TRAINING data with random forest */
proc hp4score data=work.train;
    id price_log;
    score file="&path./rfmodel.bin" out=rf_train_scored;
run;

/* Score TEST data with random forest */
proc hp4score data=work.test;
    id price_log;
    score file="&path./rfmodel.bin" out=rf_test_scored;
run;



/* Calculate RMSE for Random Forest */
proc sql noprint;
    /* Training RMSE */
    select sqrt(mean((price_log - p_price_log)**2)) as rmse_train,
           sqrt(mean((exp(price_log) - exp(p_price_log))**2)) as true_rmse_train
    into :rmse_train_rf, :true_rmse_train_rf
    from rf_train_scored;
    
    /* Test RMSE */
    select sqrt(mean((price_log - p_price_log)**2)) as rmse_test,
           sqrt(mean((exp(price_log) - exp(p_price_log))**2)) as true_rmse_test
    into :rmse_test_rf, :true_rmse_test_rf
    from rf_test_scored;
quit;

%put ========================================;
%put Random Forest Results:;
%put   Training RMSE: &rmse_train_rf;
%put   Training True RMSE: &true_rmse_train_rf;
%put   Test RMSE: &rmse_test_rf;
%put   Test True RMSE: &true_rmse_test_rf;
%put ========================================;

/* Print results in a table */
data rf_rmse_results;
    length model $30 dataset $10;
    model = "Random Forest";
    
    dataset = "Training";
    rmse = input("&rmse_train_rf", 8.4);
    true_rmse = input("&true_rmse_train_rf", 8.4);
    output;
    
    dataset = "Test";
    rmse = input("&rmse_test_rf", 8.4);
    true_rmse = input("&true_rmse_test_rf", 8.4);
    output;
run;

proc print data=rf_rmse_results noobs;
    title "Random Forest - RMSE Summary";
    var model dataset rmse true_rmse;
    format rmse true_rmse 8.4;
run;




/*Model Comparison*/
/* Combine LASSO results with macro variables */
proc sql noprint;
    select sqrt(mean(squared_error)) as rmse_train,
           sqrt(mean((trans_price - trans_predicted_price)**2)) as true_rmse_train
    into :rmse_train_lasso, :true_rmse_train_lasso
    from train_measure;
    
    select sqrt(mean(squared_error)) as rmse_test,
           sqrt(mean((trans_price - trans_predicted_price)**2)) as true_rmse_test
    into :rmse_test_lasso, :true_rmse_test_lasso
    from test_measure;
quit;

/* Create comprehensive comparison table */
data model_comparison;
    length model $30 dataset $10;
    
    /* LASSO */
    model = "LASSO Regression";
    dataset = "Training";
    rmse = input("&rmse_train_lasso", 8.4);
    true_rmse = input("&true_rmse_train_lasso", 8.4);
    output;
    
    dataset = "Test";
    rmse = input("&rmse_test_lasso", 8.4);
    true_rmse = input("&true_rmse_test_lasso", 8.4);
    output;
    
    /* Decision Tree */
    model = "Decision Tree";
    dataset = "Training";
    rmse = input("&rmse_train_tree", 8.4);
    true_rmse = input("&true_rmse_train_tree", 8.4);
    output;
    
    dataset = "Test";
    rmse = input("&rmse_test_tree", 8.4);
    true_rmse = input("&true_rmse_test_tree", 8.4);
    output;
    
    /* Random Forest */
    model = "Random Forest";
    dataset = "Training";
    rmse = input("&rmse_train_rf", 8.4);
    true_rmse = input("&true_rmse_train_rf", 8.4);
    output;
    
    dataset = "Test";
    rmse = input("&rmse_test_rf", 8.4);
    true_rmse = input("&true_rmse_test_rf", 8.4);
    output;
run;

proc print data=model_comparison noobs;
    title "Model Performance Comparison - All Models";
    var model dataset rmse true_rmse;
    format rmse true_rmse 8.4;
run;



