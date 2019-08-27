#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[46]:


import copy
import datetime
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import xlrd
from dateutil.relativedelta import relativedelta
# Configure Pandas options
import locale
import pandas as pd
from pathlib import Path
# Set display options for dataframes output as strings
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
locale.setlocale(locale.LC_NUMERIC, '')


# Use seaborn for platting
import seaborn as sns; sns.set()
# Configure logging
import logging
import sys
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Create STDERR handler
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(levelname)s - %(funcName)s - %(message)s')
handler.setFormatter(formatter)
# Set STDERR handler as the only handler 
logger.handlers = [handler]
# ### Examples
# ```
# # Set logging level
# logger.setLevel(logging.DEBUG)
# # Example to defer output
# if logger.isEnabledFor(logging.DEBUG):
#      logger.debug(display(df))
#  ```

# ## Configure debugging
import pdb
# Toggole automatic debugging at errors
get_ipython().run_line_magic('pdb', 'off')
# Set error display
get_ipython().run_line_magic('xmode', 'plain')
# ### Usage
# ```
# # Set a breakpoint
# pdb.set_trace()
# # open shell at error
# pdb.pm()
# ```

# Set project paths
home = os.path.expanduser("~")
print('home = ' + home)

projectRoot = os.path.dirname(os.getcwd())
print('projectRoot = ' + projectRoot)

# Set path to Box Sync
if os.path.isdir(Path(home, 'Box\\ Sync')):
    dirBox = Path(home, 'Box\\ Sync')
elif os.path.isdir(Path(home, 'Documents', 'Box\\ Sync')):
    dirBox = Path(home, 'Documents', 'Box\\ Sync')
else:
    print('Cannot locate Box Sync directory.')
print('dirBox = ' + str(dirBox))

# Set paths
dirDataPrivate = Path(dirBox, 'data', 'cmi')
print('dirDataPrivate = ' + str(dirDataPrivate))

dirDataPublic = Path(projectRoot, 'data')
print('dirDataPublic = ' + str(dirDataPublic))

dirDataEpic = Path(dirDataPrivate, 'epic')
print('dirDataEpic = ' + str(dirDataEpic))

dicts = Path(dirDataEpic, 'secrets')
sys.path.append(os.path.abspath(dicts))
import dicts


# # Notes about the raw data

# ## The following are identified for each vaccine pre-intervention file:  
# 
# * each patient seen during the preintervention time frame  
# * the first visit the patient had to any Department within the Location during the time frame  
# * the number of encounters to the Location during the time frame  
# * the last date the vaccine was given as long as that date is prior to the end of the baseline time frame  
# * if the date of the vaccine matches the visit date, there is a Yes in the column indicating the vaccine was given at the visit  
# * if the date of the vaccine was prior to the first baseline visit, there is a Yes in the column indicating the vaccine was given before the first visit  
# 
# ## The following are identified for each vaccine post-intervention file:  
# 
# * each patient seen during the post intervention time frame  
# * the first visit the patient had to any Department within the Location during the time frame  
# * number of encounters to the Location during the time frame - if a patient was seen in multiple departments, each department will show the same number of visits  
# * the last date the vaccine was given as long as that date is prior to the end of the post-intervention time frame  
# * For Td/Tdap and Pneumo/PCV, the dates of the vaccines will allow the analysis to include what was given first and the interval between the vaccines  
# * if the date of the vaccine matches the visit date, there is a Yes in the column indicating the vaccine was given at the visit  
# * if the date of the vaccine was prior to the first baseline visit, there is a Yes in the column indicating the vaccine was given before the first visit   
# 
# ## Patients will appear multiple times for the following reasons:
# 
# * Their age changed during the time frame so they appear as Age 60-64 and as 65+  
# * They were seen in more than one department within the location  
# 

# # Data cleaning helpers

# In[ ]:


def ingest_raw_data(file=None, convert=None):
    '''Loads raw data from .xlsx files or available pickle to a dictionay.

        Args:
            file (str): Name of file for saving and retrieving pickled data.
                Defaults to 'rawData.p'
            convert (bool): Specifies reading from .xlsx when True otherwise, data is
                read from pickle if available. Defaults to None.

        Returns:
            di (dict): Dictionary of dataframes indexed by filename.

    '''
    if file == None:
        file = 'rawData.p'
    else:
        file = file + '.p'
    if (convert == True) | (os.path.isfile(os.path.join(dirDataEpic,
                                                        file)) == False):

        # Create dict of dataframes indexed by filename
        files = [f for f in os.listdir(dirDataEpic) if f.endswith('.xlsx')]
        logger.info('Reading raw data.')
        di = {
            'df' + file.replace(".xlsx", ""): pd.read_excel(
                os.path.join(dirDataEpic, file))
            for file in files
        }
        save_results_to_pickle(obj=di, file=file, publicPath=dirDataPrivate, privatePath=dirDataEpic)
    else:
        di = get_results_from_pickle(file=file, publicPath=dirDataPrivate, privatePath=dirDataEpic)
    logger.info('Loaded raw data.')
    return di


# In[ ]:


def clean_epic_data(di=None, file=None, convert=None):
    '''Creates a dictionay of dataframes by vaccine and intervention timepoint

        Args:
            di (dict): Dictionary to clean.
            file (str, Optional): Name of file for saving and retrieving
                pickled data
            convert (bool, Optional): Executes data cleaning and saves 
                results when True otherwise, data is read from pickle if 
                available. Defaults to None.

        Returns:
            di (dict): Dictionary of dataframes

    '''
    if file == None:
        file = retrieve_name(di) + '.p'
    else:
        file = file + '.p'
    if (convert == True) | (os.path.isfile(os.path.join(dirDataEpic,
                                                        file)) == False):
        if di == None:
            raise UnboundLocalError(
                'Argument "di" must be passed when convert=True.')
        logger.info('Processing data')
        #     Merge in LOC_ID
        di = fix_flu_files(di)
        #   Add missing birthdays
        di = concat_birthdays(di)
        # Fix column labels
        di = standardize_column_names(di)
        # convert date columns to datetime
        di = convert_dates(di)
        # Convert variables to bools
        di = convert_bools(di)
        #  add additional data
        di = append_additional_data(di)
        #    Mark pre/post intervention records
        di = add_timepoint_indicator(di)
        # Strip department from practice name
        di = split_practice_name(di)

        save_results_to_pickle(di, file, privatePath=dirDataEpic, publicPath=dirDataPrivate)
    else:
        di = get_results_from_pickle(file=file, privatePath=dirDataEpic, 
                                     publicPath=dirDataPrivate)
    return di


def fix_flu_files(di):
    '''Fills missing columns from values in available data.

        Args:
            di (dict): Dictionary of dataframes with missing columns

        Returns:
            di (dict): Dictionary of dataframes with missing columns appended and filled
    '''
    problemFiles = [
        'dfFlu2014-15_AllLocs', 'dfFlu2017-18_AllLocs',
        'dfFlu2014-15_AdditionalLocs', 'dfFlu2017-18_AdditionalLocs'
    ]
    dfPractices = pd.DataFrame()
    # Collect LOC_IDs to fix flu files
    for k, df in di.items():
        if k not in problemFiles:
            dfPractices = pd.concat([dfPractices, df[['LOC_ID', 'PRACTICE']]],
                                    axis=0).drop_duplicates()
    # Iterate through dict to fix the flu files
    for k, df in di.items():
        if k in problemFiles:
            di[k] = di[k].merge(dfPractices)
            logger.debug('Added LOC_ID to ' + k)
    logger.info('Fixed missing data in flu files.')
    return di


def concat_birthdays(di):
    '''Joins missing BIRTH_DATE column from available data.

        Args: 
            di (dict): Dictionary of dataframes

        Returns:
            di (dict): Dictionary with BIRTH_DATE column joined to all dataframes
    '''
    problemFiles = ['dfPreIntervZoster_AllLOCs']
    dfBirthdays = pd.DataFrame()
    for k, df in di.items():
        # Create list of birthdays
        if k not in problemFiles:
            dfBirthdays = pd.concat(
                [dfBirthdays, df[['PAT_MRN_ID', 'BIRTH_DATE']]],
                axis=0).drop_duplicates()
    for k, df in di.items():
        if k in problemFiles:
            di[k] = di[k].merge(dfBirthdays)
            logger.debug('Added birthdays to ' + k)
    logger.info('Added missing birthdays.')
    return di


def standardize_column_names(di):
    '''Applies map of old values: new values to make columns consistant.

        Args:
            di (dict): Dictionary of dataframes to be updated

        Returns:
            di (dict): Dict of dataframes with updated column labels
    '''
    mapper = {
        'NUM_OF_ENCS_2014_15': 'NUM_OF_ENCS',
        'NUM_OF_ENCS_2017_18': 'NUM_OF_ENCS',
        'NUMBER_OF_ENCOUNTERS': 'NUM_OF_ENCS',
        'LAST_FLU_DT_14_15': 'LAST_FLU_DT',
        'LAST_FLU_DT_2017_18': 'LAST_FLU_DT',
        "FIRST_FLU_SEASON_VISIT": "FIRST_FLU_VISIT_DT",
        "FIRST_BASELINE_VISIT": "FIRST_VISIT_DT",
        'FIRST_POSTINTERV_VISIT': "FIRST_VISIT_DT",
        'ZOST_AT_A_VISIT': 'ZOST_AT_VISIT',
        'FLUVAC_AT_A_VISIT': 'FLU_AT_VISIT',
        'FLUVAC_AT_VISIT': 'FLU_AT_VISIT',
        'FLUVAC_BEFORE_FIRST_VIS': 'FLU_BEFORE_FIRST_VIS',
        'LAST_ZOSTER_DT': 'LAST_ZOST_DT'
    }
    for k, df in di.items():
        di[k] = di[k].rename(mapper, axis='columns')
        # Special treatment for flu encounters
        if k in [
                'dfFlu2014-15', 'dfFlu2017-18', 'dfFlu2014-15_AdditionalLocs',
                'dfFlu2014-15_AllLocs', 'dfFlu2017-18_AdditionalLocs',
                'dfFlu2017-18_AllLocs'
        ]:
            di[k] = di[k].rename({'NUM_OF_ENCS': 'NUM_FLU_ENCS'},
                                 axis='columns')
        di[k] = di[k].drop('LIVING_STATUS', axis=1)
        logger.debug('Updated column labels for ' + k)
    logger.info('Standardized column names.')
    return di


def convert_dates(di):
    '''Converts text dates to datetime type.

        Args:
            di (dict): Dictionary of dataframes containing text dates

        Returns:
            di (dict): Dictionary of dataframes with converted datetimes
    '''
    # Variables to convert to dates
    dateVars = [
        'BIRTH_DATE', 'FIRST_VISIT_DT', 'FIRST_FLU_VISIT_DT', 'LAST_TDAP_DT',
        'LAST_TD_DT', 'LAST_ZOST_DT', 'LAST_PNEUMO_DT', 'LAST_PCV_DT',
        'LAST_FLU_DT'
    ]
    for k, df in di.items():
        dateCols = [i for i in list(df) if i in dateVars]
        df[dateCols] = df[dateCols].apply(pd.to_datetime)
        logger.debug(str(dateCols) + ' converted to datetime.')
    logger.info('Converted dates.')
    return di


def convert_bools(di):
    '''Converts text values in dataframes to boolean True or False.

        Args:
            di (dict): Dictionary of dataframes with truth values stored as text

        Returns:
            di (dict): Dictionary of dataframes with boolean truth values
    '''
    # Variables to convert to booleans
    boolVars = [
        'AGE_60_64', 'AGE_65PLUS', 'ZOST_BEFORE_FIRST_VIS', 'ZOST_AT_VISIT',
        'FLU_AT_VISIT', 'FLU_BEFORE_FIRST_VIS', 'PNEUMO_AT_VISIT',
        'PNEUMO_BEFORE_FIRST_VIS', 'PCV_BEFORE_FIRST_VIS', 'PCV_AT_VISIT',
        'TDAP_AT_VISIT', 'TDAP_BEFORE_FIRST_VIS', 'TD_BEFORE_FIRST_VIS',
        'TD_AT_VISIT'
    ]
    # Build dictionaries of True / False strings
    dictTF = {
        'YES': True,
        'NO': False,
        'Y': True,
        'N': False,
        'True': True,
        'False': False,
        'NaN': False
    }

    # Build lists of T/F values
    trues = [k for k, v in dictTF.items() if v == True]
    falses = [k for k, v in dictTF.items() if v == False]
    for k, df in di.items():
        # perform conversion with error handling
        for i in boolVars:
            if i in df.columns:
                countTrue = len(df[df[i] == True]) + len(df[df[i].isin(trues)])
                countFalse = len(df[df[i] == False]) + len(
                    df[df[i].isin(falses)]) + len(df[df[i].isnull()])
                df[i] = df[i].fillna(False).astype(str).replace(dictTF).astype(
                    bool)
                df[i] = df[i].astype(bool)

                # check results
                countTrueAfter = len(df[df[i] == True])
                countFalseAfter = len(df[df[i] == False])

                assert(countTrue == countTrueAfter) & (countFalse == countFalseAfter) &                     (len(df[i]) == countTrueAfter + countFalseAfter),                     'T/F substitution error on ' + i
                logger.debug('T/F substitution complete for ' + i)
    logger.info('Converted boolean variables')
    return di


def append_additional_data(di):
    '''Combines multiple data pulls into files by vaccine & intervention period.

        Args:
            di (dict): Dictionary of dataframes to be combined

        Returns:
            diNew (dict): New dictionary of dataframes combined per specification

    '''
    # These are the data files
    matches = [('Flu2014-15_AdditionalLocs.xlsx', 'Flu2014-15_AllLocs.xlsx'),
               ('Flu2017-18_AdditionalLocs.xlsx', 'Flu2017-18_AllLocs.xlsx'),
               ('PostIntervPneumo_AdditionalLOCs.xlsx',
                'PostIntervPneumo_AllLOCs.xlsx'),
               ('PostIntervTDAP_TD_AdditionalLOCs.xlsx',
                'PostIntervTDAP_TD_AllLOCs.xlsx'),
               ('PostIntervZoster_AdditionalLOCs.xlsx',
                'PostIntervZoster_AllLOCs.xlsx'),
               ('PreIntervPneumo_AdditionalLOCs.xlsx',
                'PreIntervPneumo_AllLOCs.xlsx'),
               ('PreIntervTDAP_TD_AdditionalLOCs.xlsx',
                'PreIntervTDAP_TD_AllLOCs.xlsx'),
               ('PreIntervZoster_AdditionalLOCs.xlsx',
                'PreIntervZoster_AllLOCs.xlsx')]
    # rename pairs of data files
    diNew = {}
    for (additional, allLocs) in matches:
        # rename keys for convenience
        additional = 'df' + additional.replace('.xlsx', '')
        allLocs = 'df' + allLocs.replace('.xlsx', '')
        k = allLocs.replace('_AllLOCs', '').replace('_AllLocs', '')

        # Record dtypes
        diDtypes = {}
        diDtypes.update(di[allLocs].dtypes.to_dict())
        diDtypes.update(di[additional].dtypes.to_dict())
        # Merge the two data files on common columns and add to a new dict
        # Apply saved dtypes
        diNew[k] = pd.concat([di[allLocs], di[additional]],
                             ignore_index=True,
                             sort=False,
                             join='outer').drop_duplicates().apply(
                                 lambda x: x.astype(diDtypes[x.name]))
        logger.debug('Additional Data appended to ' + k)
    logger.info('Combined additional data')
    return diNew


def add_timepoint_indicator(di):
    '''Adds indicator for pre/post intervention identification.

        Args:
            di (dict): Dictionary of dataframes

        Returns:
            di (dict): Dictionary of dataframes with BASELINE column set
                to True for pre-intervention and False for post-intervention
    '''
    for k, df in di.items():
        #     assign baseline indicator
        if k in [
                'dfFlu2014-15', 'dfPreIntervPneumo', 'dfPreIntervTDAP_TD',
                'dfPreIntervZoster'
        ]:
            df['BASELINE'] = True
            logger.info(
                'Added BASELINE=True intervention timepoint indicator to ' + k)
        elif k in [
                'dfFlu2017-18', 'dfPostIntervPneumo', 'dfPostIntervTDAP_TD',
                'dfPostIntervZoster'
        ]:
            df['BASELINE'] = False
            logger.info(
                'Added BASELINE=False intervention timepoint indicator to ' +
                k)
    logger.info('Added timepoint indicator')
    return di


def split_practice_name(di):
    '''Splits department id from practice name

        Args:
            di (dict): Dictionary of dataframes

        Returns:
            di (dict): Dictionary of dataframe with split practice names
    '''
    for k, df in di.items():
        # Split pratice label to department id and name
        di[k][['DEPT_ID', 'PRACTICE']] = df['PRACTICE'].str.split(
            '-', expand=True)
        di[k]['DEPT_ID'] = df['DEPT_ID'].astype('int')
        logger.debug('Split practice names in ' + str(k))
    logger.info('Split practice names')
    return di


# In[ ]:


def dedup_records(di, file=None, convert=None):
    '''De-duplicates a dictionay of dataframes of vaccine and intervention
        timepoints. Saves to .p and .csv

        Args:
            di (dict): Dictionary of processed Epic data.
            file (str, Optional): File to save pickled data.
            convert (bool, Optional): Executes data cleaning and saves results
                when True. Otherwise, data is read from pickle if available.

        Returns:
            df (dataframe): Dataframe of combined, de-duplicated dataframes

    '''
    if file == None:
        file = 'dfCombined.p'
    else:
        file = file + '.p'
    if (convert == True) | (os.path.isfile(os.path.join(dirDataEpic,
                                                        file)) == False):
        # Combine to files by vaccine
        di = combine_vaccine_data(di)
        # de-duplicate data
        df = process_columns(di)
        logger.info('Deduplicated data.')
        #  Save results
        save_results_to_pickle(
            obj=df,
            file=file,
            privatePath=dirDataEpic,
            publicPath=dirDataPrivate)
    else:
        df = get_results_from_pickle(
            file=file, privatePath=dirDataEpic, publicPath=dirDataPublic)
    return df


def combine_vaccine_data(di):
    '''Combine files by vaccine and intervention timepoint into a dataframe by vaccine.

        Args:
            di (dict): Dictionary of dataframes to be combined

        Returns:
            df (df): Dataframes combined into one
    '''
    dfs = [('dfFlu2014-15', 'dfFlu2017-18', 'dfFlu'),
           ('dfPreIntervPneumo', 'dfPostIntervPneumo', 'dfPneumo'),
           ('dfPreIntervTDAP_TD', 'dfPostIntervTDAP_TD', 'dfTDAP'),
           ('dfPreIntervZoster', 'dfPostIntervZoster', 'dfZoster')]
    news = []
    for pre_k, post_k, new_k in dfs:
        # combine
        di[new_k] = di[pre_k].append(di[post_k], sort=False).drop_duplicates()
        # customize for flu
        if 'Flu' in pre_k:
            di[new_k] = di[new_k].rename(
                columns={
                    'NUM_OF_ENCS': 'NUM_FLU_ENCS',
                    'FIRST_VISIT_DT': 'FIRST_FLU_VISIT_DT'
                })
        news.append(di[new_k])
        logger.info('Combined and updated ' + new_k)
    diNew = {}
    df = pd.concat(news, sort=False).drop_duplicates()
    logger.info('Combined all ')
    return df


def process_columns(df):
    '''Apply deduplication logic to specified dataframe by column.
    
        Args:
            df (DataFrame): DataFrame to process by column.
            
        Returns:
            df (DataFrame): Original DataFrame with per-column logic applied.
    '''

    # Iterate through the columns to identify and deduplicate by column
    for col in df.columns.values:
        # These are the permissible duplicates or handled elsewhere
        if col in [
                'PAT_MRN_ID', 'PRACTICE', 'AGE_60_65', 'LOC_ID', 'DEPT_ID',
                'AGE_60_64', 'BASELINE'
        ]:
            logger.debug('Skipping ' + col)
            pass
        else:
            logger.info('Processing ' + col)
            # Fix patients who had a birthday
            if col == 'AGE_65PLUS':
                # Limit transformations to patients at timepoints
                idx = ['BASELINE', 'PAT_MRN_ID']

                # Assume all patients who turned 65 during the period,
                # started the period at 65.
                import operator
                df.loc[:, col] = df.groupby(
                    idx)[col].transform(lambda x: bool(x.max()))
                df['AGE_60_64'] = df.groupby(
                    idx)[col].transform(lambda x: bool(operator.not_(x.max())))

            elif col in ['FIRST_VISIT_DT', 'FIRST_FLU_VISIT_DT']:
                # Limit transformations to patients at timepoints at locations
                idx = ['BASELINE', 'PAT_MRN_ID', 'LOC_ID']
                # Set these columns to oldest available value
                df[col] = df.groupby(idx)[col].transform(lambda x: x.min())

            elif col in [
                    'LAST_FLU_DT', 'NUM_OF_ENCS', 'NUM_FLU_ENCS',
                    'LAST_PNEUMO_DT', 'LAST_PCV_DT', 'LAST_TDAP_DT',
                    'LAST_TD_DT', 'LAST_ZOST_DT'
            ]:
                # Limit transformations to patients at timepoints at locations
                idx = ['BASELINE', 'PAT_MRN_ID', 'LOC_ID']
                # Set values to max available value for these columns
                df[col] = df.groupby(idx)[col].transform(lambda x: x.max())

            elif col in [
                    'FLU_AT_VISIT', 'FLU_BEFORE_FIRST_VIS', 'PNEUMO_AT_VISIT',
                    'PCV_AT_VISIT', 'PNEUMO_BEFORE_FIRST_VIS',
                    'PCV_BEFORE_FIRST_VIS', 'TDAP_AT_VISIT', 'TD_AT_VISIT',
                    'TDAP_BEFORE_FIRST_VIS', 'TD_BEFORE_FIRST_VIS',
                    'ZOST_AT_VISIT', 'ZOST_BEFORE_FIRST_VIS'
            ]:
                # Limit transformations to patients at timepoints at locations
                idx = ['BASELINE', 'PAT_MRN_ID', 'LOC_ID']
                # Set the column to true if any entry is true for the group
                df[col] = df.groupby(
                    idx)[col].transform(lambda x: bool(x.max()))

            elif col == 'BIRTH_DATE':

                def fn(group):
                    # check values
                    assert group[col].max() == group[col].min(
                    ), 'errors in birthday'

                df.groupby(['PAT_MRN_ID']).apply(fn)
            else:
                logger.debug("No logic for " + col)

    return df.drop_duplicates().sort_values(
        ['BASELINE', 'LOC_ID', 'DEPT_ID', 'PAT_MRN_ID'])


# # Site data cleaning helpers

# In[ ]:


def clean_intervention_list(di, convert=None):
    '''Creates a datafame with site information.

        Args:
            di (dict): Dictionary of Epic data.
            convert (bool, Optional): Executes data cleaning and saves 
            results when True otherwise, data is read from pickle if available.
            Defaults to None.

        Returns:
            dfSites (dataframe): Merge of site information table with data
            from Epic records        
    '''
    # Fetch Site information
    file = 'dfSites.p'
    if (convert == True) | (os.path.isfile(os.path.join(dirDataPublic,
                                                        file)) == False):
        dfSites = clean_intervention_dates_sheet()
        dfSites = create_practice_list(di, dfSites)
        #  Save results
        save_results_to_pickle(
            obj=dfSites,
            file=file,
            privatePath=dirDataEpic,
            publicPath=dirDataPrivate)
    else:
        dfSites = get_results_from_pickle(
            file=file, privatePath=dirDataEpic, publicPath=dirDataPublic)
    return dfSites


def clean_intervention_dates_sheet():
    '''Imports and cleans intervention dates and Epic location worksheet
    
        Returns:
            dfSites (dataframe): Dataframe from Excel worksheet
    '''
    cols = [
        'LOC_ID', 'LOC', 'Clinic Location Name', 'Street Address', 'PHASE',
        'Date of intervention', 'Notes'
    ]
    dfSites = pd.read_excel(
        os.path.join(dirDataPrivate,
                     'Epic ID and Intervention Dates 4 Pillars.xlsx'))

    dfSites.columns = dfSites.columns.str.strip()
    dfSites = dfSites[cols].rename(columns={
            "LOC": "DEPT_ID"
        }).reset_index(drop=True)
    dfSites = dfSites.dropna(subset=['DEPT_ID'])
    dfSites['DEPT_ID'] = dfSites['DEPT_ID'].astype('int')
    return dfSites


def create_practice_list(di, dfSites):
    '''Joins list of unique departments and practices found in Epic data with
    intervention dates and locations.
    
        Args:
            di (dict): Dictionary of dataframes from Epic export
            dfSites (dataframe): Dataframe with department, locations and practice names
            
        Returns:
            dfPractices (dataframe): Dataframe of unique departments and practices joind
            with intervention dates and locations
    '''
    # Create list of unique departments, locations & practices found in the data
    dfPractices = pd.DataFrame()
    for k, df in di.items():
        df2 = df.groupby(['DEPT_ID', 'LOC_ID',
                          'PRACTICE']).size().reset_index(name='Freq').drop(
                              'Freq', axis=1)
        dfPractices = dfPractices.append(df2, ignore_index=True)
    # Merge Pracitces from Epic with Intervention list
    dfPractices = dfPractices.merge(
        dfSites, on=['LOC_ID', 'DEPT_ID'], how='outer').sort_values(
            by=['LOC_ID', 'DEPT_ID']).drop_duplicates().reset_index(drop=True)
    dfPractices = dfPractices.dropna(subset=['PRACTICE'])
    logger.info('Practice info processed')
    return dfPractices


# In[ ]:


def clean_and_calculate_site_data(df, dfSites, file=None, convert=None):
    '''Creates a dictionay of dataframes with cleaned sites and reduced to location.

        Args:
            df (dataframe): Dataframe of processed Epic data.
            file (str, Optional): File to save pickled data.
            convert (bool, Optional): Executes data cleaning and saves 
                results when True otherwise, data is read from pickle if available.

        Returns:
            di (dict): Dictionary of dataframes aggregated to location 

    '''
    if file == None:
        file = retrieve_name(df) + '.p'
    else:
        file = file + '.p'
    if (convert == True) | (os.path.isfile(os.path.join(dirDataEpic,
                                                        file)) == False):
        logger.info('Processing data')

        dfProbs = dfSites[
            dfSites['Date of intervention'].isnull()].reset_index()[[
                'LOC_ID', 'DEPT_ID', 'PRACTICE'
            ]]
        startingProbs = questionable_departments(df, dfProbs)
        # Update site records
        df = update_site_names(df, diUpdates=dicts.diUpdates)
        # Drop records from excluded sites
        df = drop_site_records(df, diDrops=dicts.diDrops)

        endingProbs = questionable_departments(df, dfProbs)
        assert len(startingProbs) > len(endingProbs), 'Problem with cleaning'
        df = compute_vaccine_logic(df)
        di = reduce_to_locations(df, diOmit=dicts.diOmit)
        # include sites in dict
        di['dfSites'] = dfSites
        #  Save results
        save_results_to_pickle(obj=di, file=file, publicPath=dirDataPrivate, privatePath=dirDataEpic)
    else:
        di = get_results_from_pickle(file=file, publicPath=dirDataPrivate, privatePath=dirDataEpic)
    return di


def questionable_departments(df, dfProbs):
    '''Shows records aggregated by count for specified departments

        Args:
            df (dataframe): Dataframe with questionable records by department.
            dfProbs (dataframe): Dataframe of problem departments. Must 
                include 'DEPT_ID' column

        Returns:
            dfGrouped (dataframe): Dataframe of records indexed and grouped
                for inspection. Displays when logger is set to DEBUG.

    '''
    # All records with a Department problem
    idx = ['LOC_ID', 'BASELINE', 'DEPT_ID', 'PRACTICE']
    dfProbView = df[df.DEPT_ID.isin(
        dfProbs['DEPT_ID'])].drop_duplicates().reset_index().set_index(
            idx).sort_index()
    # Groupby location/Timepoint
    dfGrouped = dfProbView.reset_index().groupby(idx)['PAT_MRN_ID'].agg(
        ["count"])
    return dfGrouped


def update_site_names(df, diUpdates):
    '''Replaces site names and ids.

        Args:
            df (dataframe): Dataframe with records to be updated
            diUpdates (dictionary): Dictonary of values in the format 
                <old>:<new> to be replaced
        Returns:
            df (dataframe): Dataframe with updated names and ids
    '''
    # Update site records
    df = df.replace(to_replace=diUpdates)
    # Check that all sites were removed
    assert len(df.query(
        'DEPT_ID.isin(@diUpdates.keys())')) == 0, 'Failed to update all sites.'
    return df


def drop_site_records(df, diDrops):
    '''Drops records indexed to specified sites.

        Args:
            df (dataframe): Dataframe with records to be updated
            diDrops (dictionary): Dictonary of sites with data to drop in the format 
                'DEPT_ID': 'PRACTICE'
        Returns:
            df (dataframe): Dataframe with records dropped by site
    '''
    # Update site records
    df = df.query('~DEPT_ID.isin(@diDrops.keys())')
    # Check that all sites were removed
    assert len(df.query(
        'DEPT_ID.isin(@diDrops.keys())')) == 0, 'Problem dropping records.'
    return df


def compute_vaccine_logic(df):
    '''Computes vaccination variables for patients.
    
        Args:
            df (dataframe): Dataframe of processesd data
                
        Returns:
            df (dataframe): Dataframe of processed data with vaccine logic applied
    '''
    vaxVars = ['FLU', 'PNEUMO', 'PCV', 'TDAP', 'TD', 'ZOST']
    for vax in vaxVars:
        firstVisit = df['FIRST_VISIT_DT']
        numEncs = df['NUM_OF_ENCS']
        age65 = df['AGE_65PLUS']
        vaxBefore = df[vax + '_BEFORE_FIRST_VIS']
        lastVaxDate = df['LAST_' + vax + '_DT']
        vaxHere = df[vax + '_AT_VISIT']
        if vax in ['FLU']:
            # Use flu variables
            firstVisit = df['FIRST_FLU_VISIT_DT']
            numEncs = df['NUM_FLU_ENCS']
            # Assign vaccine eligibility to all unvaccinated
            df[vax + '_ELIGIBLE'] = np.where(vaxBefore == False, True, False)
        elif vax in ['PNEUMO', 'PCV']:
            # Assign vaccine eligibility to all unvaccinated over 65
            df[vax + '_ELIGIBLE'] = np.where(
                ((vaxBefore == False) & (age65 == True)), True, False)
        elif vax in ['TDAP']:
            # Assign vaccine eligibility to all unvaccinated
            df[vax + '_ELIGIBLE'] = np.where((vaxBefore == False), True, False)
        elif vax in ['TD']:
            # Assign vaccine eligibility to all unvaccinated, with a Tdap >10 years ago
            df[vax + '_ELIGIBLE'] = np.where(
                ((vaxBefore == False) &
                 ((df['LAST_TDAP_DT'] <
                   firstVisit - datetime.timedelta(days=3650)) == True)) |
                (((lastVaxDate <
                   firstVisit - datetime.timedelta(days=3650)) == True) &
                 ((df['LAST_TDAP_DT'] <
                   firstVisit - datetime.timedelta(days=3650)) == True)), True,
                False)
        elif vax in ['ZOST']:
            # Assign vaccine eligibility to all unvaccinated
            df[vax + '_ELIGIBLE'] = np.where(vaxBefore == False, True, False)
            # Assign vaccine eligibility to all over 60
        else:
            print('Missing logic for ' + vax)
            break
        # All vaccine logic
        eligible = df[vax + '_ELIGIBLE']
        # Find vaccines administered elsewhere
        df[vax + '_VACC_ELSEWHERE'] = np.where(
            (lastVaxDate > firstVisit) & (vaxHere == False), True, False)
        vaxElsewhere = df[vax + '_VACC_ELSEWHERE']
        # Assign vaccinated status for vaccine administrations anywhere
        df[vax + '_VACCINATED'] = np.where(
            (vaxElsewhere == True) | (vaxHere == True), True, False)
        vaccinated = df[vax + '_VACCINATED']
        # Assign missed opportunities for eligible unvaccinated
        df[vax + '_MISSED_OPS'] = np.where(
            (eligible == True) & (vaccinated == False), numEncs, np.nan)
    return df


def reduce_to_locations(df, diOmit):
    '''Excludes non-primary care visits using two different strategies.
        
        Description:
            Strategy 1 returns dfWithWic which includes patients seen in 
            walk-in-clinics in location visit counts. It assumes that vaccine
            status is prioritized for these patients and penalizes sites for 
            missing opportunities at these visits.
            
            Strategy 2 returns dfNoWic which excludes patients seen in 
            walk-in-clinics from location visit counts. It assumes that vaccine
            status is not prioritized for these patients and does not penalize 
            sites for missing opportunities at these visits.

        Args:
            df (dataframe): Dataframe with records to be updated
            diOmit (dict): Dictonary of sites to omit in the format 
                'DEPT_ID': 'PRACTICE' 

        Returns:
            di (dict): Dictionary with two dataframes produced by strategies
    '''
    di = {}
    di['dfCombined'] = df

    # Strategy 1
    # Drop and deduplicate
    dfWithWic = df.drop(['DEPT_ID', 'PRACTICE'], axis=1).drop_duplicates()
    di['dfWithWic'] = dfWithWic

    # Strategy 2
    dfNoWic = df.query('~DEPT_ID.isin(@diOmit.keys())')
    # Check that all sites were removed
    assert len(dfNoWic.query(
        'DEPT_ID.isin(@diOmit.keys())')) == 0, 'Incorrect sites included.'
    # Now drop to LOC_ID
    dfNoWic = dfNoWic.drop(['DEPT_ID', 'PRACTICE'], axis=1).drop_duplicates()
    di['dfNoWic'] = dfNoWic
    
    # Strategy 3
    lsOmit = [11207, 1145, 11156, 11131]
    dfDropWic = df.query('~LOC_ID.isin(@lsOmit)')
    assert len(dfDropWic.query('LOC_ID.isin(@lsOmit)')) == 0, 'Incorrect sites included'
    # Now drop to LOC_ID
    dfDropWic = dfDropWic.drop(['DEPT_ID', 'PRACTICE'], axis=1).drop_duplicates()
    di['dfDropWic'] = dfDropWic
    return di


# In[ ]:


def aggregate_and_calculate(di, file=None, convert=None):
    '''Creates a dictionay of dataframes aggregated to levels with outcomes.

        Args:
            di (dict): Dictionary of processed Epic data.
            file (str, Optional): File to save pickled data.
            convert (bool, Optional): Executes data cleaning and saves 
                results when True otherwise, data is read from pickle if available.

        Returns:
            di (dict): Dictionary of dataframes aggregated with outcomes

    '''
    if file == None:
        file = retrieve_name(di) + '.p'
    else:
        file = file + '.p'
    if (convert == True) | (os.path.isfile(os.path.join(dirDataEpic,
                                                        file)) == False):
        # Execute logic
        di = aggregate_to_locations(di)
        di = compute_outcomes(di)

        logger.info('Aggregated data and calculated outcomes.')
        #  Save results
        save_results_to_pickle(
            obj=di,
            file=file,
            privatePath=dirDataEpic,
            publicPath=dirDataPrivate)
    else:
        di = get_results_from_pickle(
            file=file, privatePath=dirDataEpic, publicPath=dirDataPublic)
    return di


def aggregate_to_locations(di):
    '''Aggregates from patient level to location level.
    
        Args:
            di (dict): Dictionary of processed data
            
        Returns:
            di (dict): Dictionary of processed data with aggregated dataframes
    '''
    for k, df in copy.copy(di).items():
        
        if 'Wic' in k:
            aggFxns = {
                'PAT_MRN_ID': 'count',
                'AGE_60_64': 'sum',
                'AGE_65PLUS': 'sum',
                'FIRST_VISIT_DT': 'min',
                'NUM_OF_ENCS': 'sum',
                'FIRST_FLU_VISIT_DT': 'min',
                'NUM_FLU_ENCS': 'sum',
                'LAST_FLU_DT': 'max',
                'FLU_AT_VISIT': 'sum',
                'FLU_VACC_ELSEWHERE': 'sum',
                'FLU_ELIGIBLE': 'sum',
                'FLU_VACCINATED': 'sum',
                'FLU_MISSED_OPS': 'sum',
                'LAST_PNEUMO_DT': 'max',
                'PNEUMO_AT_VISIT': 'sum',
                'PNEUMO_VACC_ELSEWHERE': 'sum',
                'PNEUMO_ELIGIBLE': 'sum',
                'PNEUMO_VACCINATED': 'sum',
                'PNEUMO_MISSED_OPS': 'sum',
                'LAST_PCV_DT': 'max',
                'PCV_AT_VISIT': 'sum',
                'PCV_VACC_ELSEWHERE': 'sum',
                'PCV_ELIGIBLE': 'sum',
                'PCV_VACCINATED': 'sum',
                'PCV_MISSED_OPS': 'sum',
                'LAST_TDAP_DT': 'max',
                'TDAP_AT_VISIT': 'sum',
                'TDAP_VACC_ELSEWHERE': 'sum',
                'TDAP_ELIGIBLE': 'sum',
                'TDAP_VACCINATED': 'sum',
                'TDAP_MISSED_OPS': 'sum',
                'LAST_TD_DT': 'max',
                'TD_AT_VISIT': 'sum',
                'TD_VACC_ELSEWHERE': 'sum',
                'TD_ELIGIBLE': 'sum',
                'TD_VACCINATED': 'sum',
                'TD_MISSED_OPS': 'sum',
                'LAST_ZOST_DT': 'max',
                'ZOST_AT_VISIT': 'sum',
                'ZOST_VACC_ELSEWHERE': 'sum',
                'ZOST_ELIGIBLE': 'sum',
                'ZOST_VACCINATED': 'sum',
                'ZOST_MISSED_OPS': 'sum'
            }
            # Aggregate outcomes by timepoint and location
            dfOutcomesLocs = df.groupby(
                ['BASELINE', 'LOC_ID'],
                sort=True).agg(aggFxns).reset_index()
            dfOutcomesLocs.rename(columns={
                    'PAT_MRN_ID': 'N_PATIENT'
                }, inplace=True)
            # save to dict
            di.update({k + '_Outcomes_Locs': dfOutcomesLocs})
            # Aggregate outcomes by Timepoint
            aggFxns.update({'LOC_ID': 'count'})
            dfOutcomesBaseline = df.groupby('BASELINE').agg(
                aggFxns).reset_index()
            dfOutcomesBaseline.rename(columns={
                    'PAT_MRN_ID': 'N_PATIENT',
                    'LOC_ID': 'N_LOC'
                }, inplace=True)
            # save to dict
            di.update({k + '_Outcomes_Baseline': dfOutcomesBaseline})
        elif k == 'dfSites':
            # Aggregate sites to locations
            dfSites_Locs = df.groupby('LOC_ID').agg({
                'PRACTICE': 'first',
                'Date of intervention': 'min',
                'PHASE': 'min'
            }).reset_index()
            # save to dict
            di.update({'dfSites_Locs': dfSites_Locs})
    return di


def compute_outcomes(di):
    '''Computes vaccination outcomes for aggregated groups.
    
        Args:
            di (dict): Dictionary of processesd data
            vaxVars (list): List of vaccine strings used to identify outcomes
                and data columns
                
        Returns:
            di (dict): Dictionary of processed data with vaccine outcomes
    '''
    vaxVars = ['FLU', 'PNEUMO', 'PCV', 'TDAP', 'TD', 'ZOST']
    for k, df in di.items():
        if '_Outcomes_Locs' in k:
            for vax in vaxVars:
                df[vax +
                   '_VAX_RATE'] = df[vax + '_VACCINATED'] / df['N_PATIENT']
                if vax == 'FLU':
                    df['FLU_MISSED_OPS_RATE'] = df['FLU_MISSED_OPS'] / df[
                        'NUM_FLU_ENCS']
                else:
                    df[vax + '_MISSED_OPS_RATE'] = df[
                        vax + '_MISSED_OPS'] / df['NUM_OF_ENCS']
                di.update({k: df})
    return di


# In[ ]:


def make_presentable(di, file=None, convert=None):
    '''Creates a dictionay of dataframes conformed to tidy data structure

    Args:
        di (dict): Dictionary of dataframes aggregated with outcomes
        file (str, Optional): File to save pickled data.
        convert (bool, Optional): Executes data cleaning and saves 
            results when True otherwise, data is read from pickle if available.

    Returns:
        di (dict): Dictionary of dataframes conformed to tidy data structure

    '''
    if file == None:
        file = retrieve_name(di) + '.p'
    else:
        file = file + '.p'
    if (convert == True) | (os.path.isfile(os.path.join(dirDataEpic,
                                                        file)) == False):
        # Execute logic
        for k, df in di.items(): 
            if k not in ['dfSites', 'dfSites_Locs']:
                df = tidy_data(df)
                df = make_readable(df)
                di.update({k:df})
        logger.info('Made data tidy.')
        #  Save results
        save_results_to_pickle(
            obj=di,
            file=file,
            privatePath=dirDataEpic,
            publicPath=dirDataPrivate)
    else:
        di = get_results_from_pickle(
            file=file, privatePath=dirDataEpic, publicPath=dirDataPublic)
    return di

def tidy_data(df):
    df['AGE_60_64'] = df['AGE_60_64'].replace({True:'60-64', False:'65+'})
    df=df.drop(['AGE_65PLUS'], axis=1).rename(columns={'AGE_60_64': 'Age'})
    return df

def make_readable(df):
    df['BASELINE'] = df['BASELINE'].replace({True: 'Baseline', False: 'Follow-up'})
    return df


# # General Utilities

# In[ ]:


def get_data(convert=None):
    '''Completes data procesing and returns objects for analysis.
    
        Args:
            convert (bool, Optional, Default: False): Specifies conversion from
                raw data when true. Otherwise, data is loaded from storage.    
    '''
    if convert == None:
        convert = False

    di = ingest_raw_data(file='rawData', convert=convert)
    di = clean_epic_data(di, file='di', convert=convert)
    dfSites = clean_intervention_list(di, convert=convert)
    dfCombined = dedup_records(di, file='dfCombined', convert=convert)
    di2 = clean_and_calculate_site_data(
        dfCombined, dfSites=dfSites, file='di2', convert=convert)
    di2 = aggregate_and_calculate(di2, file='di3', convert=convert)
    di2 = make_presentable(di2, file='di4', convert=convert)
    logger.info('\n DataFrames available:\n' + str(list(di2)))
    return di2


def save_results_to_pickle(obj, file, privatePath, publicPath=None):
    '''Saves function results to specified pickle file.
    
        Args:
            obj (object): Object to pickle
            file (str): Name to use for the saved file.
            privatePath (str path): Path to local storage directory.
            publicPath (str path, Optional): Internet-accessible storage
                directory.
    '''
    pickle.dump(obj, open(os.path.join(privatePath, file), 'wb'))
    logger.debug('Saved ' + os.path.join(privatePath, file))
    if publicPath != None:
        pickle.dump(obj, open(os.path.join(publicPath, file), 'wb'))
        logger.debug('Saved ' + os.path.join(dirDataPublic, file))


def get_results_from_pickle(file, privatePath, publicPath=None):
    '''Retrieves pickle saved by sister function save_results_to_pickle
    
        Args:
            file (str): Name the saved file.
            privatePath (str path): Path to local storage directory.
            publicPath (str path, Optional): Internet-accessible storage
                directory.
        Returns:
            obj (object): Object loaded from pickle
    '''
    try:
        obj = pickle.load(open(os.path.join(privatePath, file), 'rb'))
        logger.info('Using stored data from ' +
                    os.path.join(privatePath, file))
    except error as e:
        logger.debug(e)
        try:
            obj = pickle.load(open(os.path.join(publicPath, file), 'rb'))
            logger.info('Using stored data from ' +
                        os.path.join(publicPath, file))
        except error as e:
            logger.debug(e)
            logger.info('No stored data available.')
    return obj


# In[ ]:


def get_patient_counts(df):
    idx = ['LOC_ID', 'BASELINE', 'DEPT_ID', 'PRACTICE']
    # https://stackoverflow.com/questions/15570099/pandas-pivot-tables-row-subtotals
    try:
        # Groupby practice and aggregate with count
        grouped1 = df.groupby(idx)['PAT_MRN_ID'].agg(['count']).reset_index()
        # Groupby location and aggregate with count
        grouped2 = df.groupby(
            idx[:-3])['PAT_MRN_ID'].agg(['count']).reset_index()
        # Add columns so we can append and index
        grouped2['BASELINE'] = '**Total**'
        grouped = grouped2.append(grouped1, sort=True).set_index(
            idx).sort_index().rename(columns={'count': 'Patient count'})
    except:
        idx = idx[:-2]
        # Groupby location and aggregate with count
        grouped1 = df.groupby(idx)['PAT_MRN_ID'].agg(['count']).reset_index()
        grouped2 = df.groupby(
            idx[:-1])['PAT_MRN_ID'].agg(['count']).reset_index()
        # Add columns so we can append and index
        grouped2['BASELINE'] = '**Total**'
        # Append higher grouping and use alpha sort to position total row
        grouped = grouped2.append(grouped1, sort=True).set_index(
            idx).sort_index().rename(columns={'count': 'Patient count'})
    return grouped


# In[ ]:


def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df'] *
                                           aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov

