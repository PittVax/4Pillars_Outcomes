#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[84]:


# References
# https://pythonfordatascience.org/anova-python/
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import xlsxwriter
import time
# import researchpy as rp

# Import internal modules
get_ipython().run_line_magic('run', '4Pillars_Helpers.ipynb')

# Set logging level
logger.setLevel(logging.INFO)
# Load data
diDfs = get_data(convert=False)
outcomes = [
    'FLU_VAX_RATE', 'FLU_MISSED_OPS_RATE', 'PNEUMO_VAX_RATE',
    'PNEUMO_MISSED_OPS_RATE', 'PCV_VAX_RATE', 'PCV_MISSED_OPS_RATE',
    'TDAP_VAX_RATE', 'TDAP_MISSED_OPS_RATE', 'TD_VAX_RATE',
    'TD_MISSED_OPS_RATE', 'ZOST_VAX_RATE', 'ZOST_MISSED_OPS_RATE'
]
vaxVars = ['FLU', 'PNEUMO', 'PCV', 'TDAP', 'TD', 'ZOST']

columnLabels = {
    'PAT_MRN_ID': 'Patient ID',
    'LOC_ID': 'Location ID',
    'AGE_60_64': '60-64',
    'AGE_65PLUS': 'Over 65',
    'BASELINE': 'Time period',
    'PHASE': 'Intervention phase',
    'BIRTH_DATE': 'Birth date',
    'FIRST_FLU_VISIT_DT': 'First visit of flu season',
    'FIRST_VISIT_DT': 'First visit during time period',
    'FLU_AT_VISIT': 'Flu vaccine administered at a visit',
    'FLU_BEFORE_FIRST_VIS': 'Flu vaccine before first visit',
    'FLU_ELIGIBLE': 'Flu vaccine eligible',
    'FLU_MISSED_OPS': 'Flu missed opportunities',
    'FLU_VACCINATED': 'Current flu immunization',
    'FLU_VACC_ELSEWHERE': 'Flu vaccine administered elsewhere',
    'LAST_FLU_DT': 'Last visit of flu season',
    'LAST_PCV_DT': 'Last PCV vaccination',
    'LAST_PNEUMO_DT': 'Last pneumo vaccination',
    'LAST_TDAP_DT': 'Last Tdap vaccination',
    'LAST_TD_DT': 'Last TD vaccination',
    'LAST_ZOST_DT': 'Last zoster vaccination',
    'NUM_FLU_ENCS': 'Visits during flu season',
    'NUM_OF_ENCS': 'Number of visits during time period',
    'PCV_AT_VISIT': 'PCV administered at a visit',
    'PCV_BEFORE_FIRST_VIS': 'PCV administered before first visit',
    'PCV_ELIGIBLE': 'PCV eligible',
    'PCV_MISSED_OPS': 'PCV missed opportunities',
    'PCV_VACCINATED': 'Current PCV immunization',
    'PCV_VACC_ELSEWHERE': 'PCV administered elsewhere',
    'PNEUMO_AT_VISIT': 'Pneumo vaccine administered at a visit',
    'PNEUMO_BEFORE_FIRST_VIS': 'Pneumo vaccine before first visit',
    'PNEUMO_ELIGIBLE': 'Pneumo vaccine eligible',
    'PNEUMO_MISSED_OPS': 'Pneumo missed opportunities',
    'PNEUMO_VACCINATED': 'Current pneumo immunization',
    'PNEUMO_VACC_ELSEWHERE': 'Pneumo vaccine administered elsewhere',
    'TDAP_AT_VISIT': 'Tdap administered at a visit',
    'TDAP_BEFORE_FIRST_VIS': 'Tdap administered before first visit',
    'TDAP_ELIGIBLE': 'Tdap vaccine eligible',
    'TDAP_MISSED_OPS': 'Tdap missed opportunities',
    'TDAP_VACCINATED': 'Current Tdap immunization',
    'TDAP_VACC_ELSEWHERE': 'Tdap vaccine administered elsewhere',
    'TD_AT_VISIT': 'Td administered at a visit',
    'TD_BEFORE_FIRST_VIS': 'Td administered before first visit',
    'TD_ELIGIBLE': 'Td vaccine eligible',
    'TD_MISSED_OPS': 'Td missed opportunities',
    'TD_VACCINATED': 'Current Td immunization',
    'TD_VACC_ELSEWHERE': 'Td administered elsewhere',
    'ZOST_AT_VISIT': 'Zoster vaccine administered at a visit',
    'ZOST_BEFORE_FIRST_VIS': 'Zoster vaccine administered before first visit',
    'ZOST_ELIGIBLE': 'Zoster vaccine eligible',
    'ZOST_MISSED_OPS': 'Zoster vaccine missed opportunities',
    'ZOST_VACCINATED': 'Current zoster immunization',
    'ZOST_VACC_ELSEWHERE': 'Zoster vaccine administered elsewhere',
    'FLU_VAX_RATE': 'Flu vaccination rate',
    'FLU_MISSED_OPS_RATE': 'Flu vaccine missed opportunities rate',
    'PNEUMO_VAX_RATE': 'Pneumo vaccination rate',
    'PNEUMO_MISSED_OPS_RATE': 'Pneumo vaccine missed opportunities rate',
    'PCV_VAX_RATE': 'PCV vaccination rate',
    'PCV_MISSED_OPS_RATE': 'PCV vaccine missed opportunities rate',
    'TDAP_VAX_RATE': 'Tdap vaccination rate',
    'TDAP_MISSED_OPS_RATE': 'Tdap vaccine missed opportunities rate',
    'TD_VAX_RATE': 'Td vaccination rate',
    'TD_MISSED_OPS_RATE': 'Td vaccine missed opportunities rate',
    'ZOST_VAX_RATE': 'Zoster vaccination rate',
    'ZOST_MISSED_OPS_RATE': 'Zoster vaccine missed opportunities rate',
    'WIC': 'Walk-in-clinic',
    'FLU': 'Influenza',
    'PNEUMO': 'Pneumo',
    'TDAP': 'Tdap',
    'TD': 'Td',
    'ZOST': 'Zoster',
    'AT_VISIT': 'Vaccine administered at a visit',
    'BEFORE_FIRST_VIS': 'Vaccine administered before first visit',
    'ELIGIBLE': 'Vaccine eligible',
    'MISSED_OPS': 'Vaccine missed opportunities',
    'VACCINATED': 'Immunized',
    'VACC_ELSEWHERE': 'Vaccine administered elsewhere',
    'VAX_RATE': 'Vaccination rate',
    'MISSED_OPS_RATE': 'Vaccine missed opportunities rate',
    'df': 'Df',
    'N_PATIENT': 'Patients',
    'sum_sq': 'Sum sq',
    'Date of intervention': 'Start date of intervention',
    'PHASE': 'Phase'
}


# ## Data
# * Data is available in the dictionary `diDfs`. DataFrames can be selected by index as `diDfs['dfIndex']`.
# * Underscores (`_`) precede an aggregated feature (`_Outcomes`) then the aggregation level. (`_Locs`).
# * `dfCombined` contains all cleaned EMR data in tidy format.
#     * `dfWithWic` excludes walk-in-clinic departments, but retains patients with visits to that department.
#     * `dfNoWic` excludes walk-in-clinic departments and all patients with visits to the that department.
#     * `dfDropWic` excludes all locations with >= 5% difference between the first two strategies.
# * `dfSites` contains all practice information in tidy format.
#     * `dfSites_Locs` is practice information aggregated to location-level.

# ### Patient-level data
# Patient-level data came aggregated to the time period and location. A location is a group of billing departments that roll up to a single billing entity. For example a location may include Dr. Smith's East Office, Dr. Smith's West Office and Dr. Smith's Walk-in-clinics at both offices. A partial selection of the dataset is shown below.
# * Columns show the variables collected from Epic including:  
#     * the date of the first visit the patient had to any department within the Location during the time frame  
#     * the total number of encounters to any department(s) of the location during the time frame  
#     * vaccination status before the first visit of the time frame  
#     * vaccine eligibility  
#     * vaccine administration during the time frame  
#     * vaccine administration at a visit  
# * Vaccination outcomes
#     * `_VAX_RATE` is ratio of vaccinated patients to total location population
#     * `_MISSED_OPS_RATE` is the ratio of visits by unvaccinated patients to total visits  

# In[85]:


# Redacted for privacy
# diDfs['dfCombined'].set_index(
#     ['PAT_MRN_ID', 'BASELINE', 'LOC_ID', 'DEPT_ID',
#      'PRACTICE']).sort_index().head()


# ###  Location-level aggregated data
# The patient-level data is aggregated to the location using sum, min, or max as noted in the source code.  
# * Each row is a location.
# * N = the count of patients with at least one visit to any department in the location.

# In[86]:


# Redacted for privancy
# diDfs['dfWithWic'].set_index(['BASELINE', 'LOC_ID']).head()


# ### Site information

# In[87]:


# Redacted for privacy
# diDfs['dfSites']


# ## Analyses

# ### Differences between WIC & NoWIC
# Location data was aggregated using two different strategies to tabulate visits to the walk-in-clinics. Due to the structure of the available data, excluding visits to the walk-in-clinics may or may not bias results for the location. Vaccination outcomes may differ in this department for several reasons. From a provider perspective, walk-in-clinic appointments are focused more on an emergent conditions and less on preventive care. Though vaccination is rarely contra-indicated by the typical conditions addressed in walk-in-clinics, it may be overshadowed by more urgent complaints. Similarly, from a patient perspective, vaccination at a walk-in-clinic visit may be perceived as too burdensome to consider while injured or ill. Likelihood of vaccination may also vary from other departments as not all patients seen in a location's walk-in-clinic are necessarily primary care patients at the location. That is, patients whose primary care provider practices at the location often seek urgent treatment at the location's walk-in-clinic, but the reverse may not be true.
# 
# To assess the impact of two strategies for aggregating patient-level data to location data, a test of the conditions is necessary.  
# * Strategy 1 (Filter) returns a data set which includes patients seen in the locations' walk-in-clinics in the aggregated location counts. If a large proportion of walk-in patients do not also visit the location for preventive services, the population served by the clinic will include a number of visits where assessment of vaccination may differ from its priority during a scheduled visit. Consequently, the location will show more missed opportunities if walk-in visits do not consider vaccine status with the same importance as during primary care visits.  
# * Strategy 2 (Exclude) returns a data set which excludes patients seen in walk-in-clinics from location visit counts. This eliminates the count of visits from patients whose medical home resides elsewhere, but penalizes sites who do prioritize vaccination at walk-in clinics. For example locations who code drop-in vaccination clinics as walk-in visits or whose clinical staff do use urgent care visits as an opportunity to vaccinate, will not be credited for these efforts.  
# * Strategy 3 (Drop) returns a data set which excludes all locations with more than a 5% difference between patient counts produced by strategy 1 and strategy 2.
# 
# Analysis below shows that all strategies yield similar results. Data from strategy 2 will be used for the remainder of analyses.

# #### Does strategy for aggregating WIC visits matter?
# No
# Charts below compare the 3 strategies to aggregate results with the available data.
# All 3 strategies yield similar distributions.
# ANOVAs are significant for the same outcomes
# 
# Conclusion: Use Strategy 2 (`dfNoWic`) which represents the desired context

# In[88]:


dfDropWic_Outcomes_Locs = diDfs['dfDropWic_Outcomes_Locs']
dfDropWic_Outcomes_Locs['WIC'] = 'Drop'
dfNoWic_Outcomes_Locs = diDfs['dfNoWic_Outcomes_Locs']
dfNoWic_Outcomes_Locs['WIC'] = 'Exclude'
dfWithWic_Outcomes_Locs = diDfs['dfWithWic_Outcomes_Locs']
dfWithWic_Outcomes_Locs['WIC'] = 'Filter'
data = pd.concat(
    [dfWithWic_Outcomes_Locs, dfNoWic_Outcomes_Locs, dfDropWic_Outcomes_Locs])
idCols = ['BASELINE', 'LOC_ID', 'WIC']
cols = idCols + outcomes

df = data.melt(id_vars=idCols, value_vars=outcomes)
df[['Vaccine', 'Variable']] = df['variable'].str.split('_', n=1, expand=True)

df = df.replace(columnLabels)

df['BASELINE'] = df['BASELINE'].replace({True: 'Baseline', False: 'Follow-up'})
df['Variable'] = df['Variable'].replace({
    'VAX_RATE':
    'Vaccination rate',
    'MISSED_OPS_RATE':
    'Missed opportunities rate'
})

timestr = time.strftime("%Y%m%d-%H%M")
for vax, df in df.groupby(['Vaccine']):
    plot = sns.catplot(x='BASELINE',
                       order=['Baseline', 'Follow-up'],
                       y="value",
                       kind='violin',
                       hue='WIC',
                       col="Variable",
                       row='Vaccine',
                       sharex=True,
                       sharey=False,
                       margin_titles=False,
                       data=df)
    plt.show(plot)
#     plot.savefig(os.path.join(projectRoot, 'reports', 'figures', 'plt_violin_' + vax + "_" + timestr + '.png'))


# In[89]:


groupedWic = get_patient_counts(diDfs['dfWithWic'])
groupedNoWic = get_patient_counts(diDfs['dfNoWic'])
df = groupedWic.join(groupedNoWic, rsuffix='_rt').rename(
    columns={
        'Patient count': 'Patient count with WIC (Strategy 1)',
        'Patient count_rt': 'Patient count without WIC (Strategy 2)'
    })

df['Percent Difference'] = (df['Patient count with WIC (Strategy 1)'] -
                            df['Patient count without WIC (Strategy 2)']) / (
                                df['Patient count with WIC (Strategy 1)'] +
                                df['Patient count without WIC (Strategy 2)'])
print('Differing patient counts from alternative handling of WIC.')
df = df[df['Percent Difference'] > 0].query(
    'BASELINE != "**Total**"').reset_index().sort_values('Percent Difference')
df = df.sort_values('Percent Difference')
df['> 0.05 Difference'] = np.where(df['Percent Difference'] > .05, '*',
                                         '')
print('Locations with >= .05 Difference')
df.index.name = 'Site ID'
df.iloc[:, 2:].to_excel(
    os.path.join(projectRoot, 'reports', 'tables', 'wic-strategies.xlsx'))
df.iloc[:, 2:]
df = df[df['Percent Difference'] > .05].query(
    'BASELINE != "**Total**"').reset_index().sort_values('Percent Difference')
locFilter = list(df['LOC_ID'])
locFilter


# ### Descriptives

# #### Intervention dates

# In[90]:


dfFirstDates = diDfs['dfCombined'].groupby(['BASELINE']).agg({
    'FIRST_VISIT_DT':
    'min',
    'FIRST_FLU_VISIT_DT':
    'min',
    'LAST_FLU_DT':
    'max'
}).reset_index()

dfFirstDates[dfFirstDates['BASELINE'] == 'Baseline']
sites = diDfs['dfSites'].groupby('PHASE').agg({
    'Date of intervention': 'min',
    'LOC_ID': 'count'
}).reset_index().rename(
    columns={
        'PHASE': 'Phase',
        'Date of intervention': 'Intervention start',
        'LOC_ID': 'Location count'
    })
display(sites)
display('Total locations = ' + str(sites['Location count'].sum()))
dfFirstDates[dfFirstDates['BASELINE'] == 'Follow-up']


# #### Descriptors

# In[91]:


# Prepare a dataframe
dfData = diDfs['dfNoWic'][['LOC_ID', 'BASELINE', 'PAT_MRN_ID', 'Age']].copy()

# Merge with phase information (assumes locations that span multiple phases are classified in the first phase)
dfSites = diDfs['dfSites'][[
    'LOC_ID', 'PHASE', 'Date of intervention'
]].dropna().drop_duplicates().groupby(['LOC_ID']).agg({
    'Date of intervention': 'first',
    'PHASE': 'first'
}).reset_index()
dfData = pd.merge(left=dfData, right=dfSites, on='LOC_ID')


# ##### By Timepoint

# In[92]:


ageXtime = pd.crosstab(dfData['BASELINE'],
                       dfData['Age'],
                       margins=True,
                       margins_name='Total',
                       rownames=[None],
                       colnames=[""])

display(ageXtime.style.format("{:,}"))
ageXtime.to_excel(
    os.path.join(projectRoot, 'reports', 'tables', 'ageXtime.xlsx'))


# ##### By Phase

# In[93]:


df = dfData[dfData['BASELINE'] == 'Baseline']
phaseXage = pd.crosstab(df['PHASE'],
                        df['Age'],
                        margins=True,
                        margins_name='Total',
                        rownames=[None],
                        colnames=[""])
phaseXage = phaseXage.merge(sites.set_index(['Phase']),
                            left_index=True,
                            right_index=True).append(
                                phaseXage.loc['Total', :]).fillna('')
display(phaseXage)
phaseXage.to_excel(
    os.path.join(projectRoot, 'reports', 'tables', 'phaseXage.xlsx'))


# ##### By location
# See below

# #### All Location-level outcomes

# In[94]:


dfData = diDfs['dfNoWic_Outcomes_Locs']
# dfData = diDfs['dfNoWic']
vaxVars = ['FLU', 'PNEUMO', 'PCV', 'TDAP', 'TD', 'ZOST']


def format_with_multiindex(df,
                           fmt,
                           multiRows=pd.IndexSlice[:],
                           multiCols=pd.IndexSlice[:]):
    """Formats specified values of a multi-indexed dataframe in place
    
    """
    df.loc[multiRows, multiCols] = df.loc[multiRows, multiCols].applymap(
        fmt.format)
    return df

# Combined dataset
dfComb = pd.DataFrame()
# Perform aggregations and summarize by vaccine then generate summary table
for vax in vaxVars:
    # restrict encounters for flu season
    if vax == 'FLU':
        encs = 'NUM_FLU_ENCS'
    else:
        encs = 'NUM_OF_ENCS'
    # Gather columns for the vaccine
    cols = ['BASELINE', 'LOC_ID', 'N_PATIENT', encs]
    cols.extend([
        c for c in list(dfData)
        if ((c.split('_', 1)[0] == vax) and ("_DT" not in c))
    ])
    # Start with empty dataframe
    df = pd.DataFrame()
    # Rename flu encs column so concat works
    df = dfData[cols].rename(columns={'NUM_FLU_ENCS': 'NUM_OF_ENCS'})
    # Converts vaccine-specific outcome list to general outcomes dict for column renaming
    cols = {
        col: col.split('_', 1)[1] if col.split('_', 1)[0] == vax else col
        for col in cols
    }
    df = df.rename(columns=cols).rename(columns=columnLabels).rename(
        columns={'N_PATIENT': 'Patients'})
    # Reorder columns
    df = df.reindex([
        'Time period', 'Patients', 'Number of visits during time period',
        'Vaccine eligible', 'Vaccine administered at a visit',
        'Vaccine administered elsewhere', 'Immunized',
        'Vaccine missed opportunities', 'Vaccination rate',
        'Vaccine missed opportunities rate'
    ],
                    axis=1)
    df = df.groupby(['Time period',
                     ]).agg(['sum', 'mean', 'std', 'min', 'max', 'median'])
    if vax != "PCV":
        df['Vaccine'] = vax.capitalize()
    else:
        df['Vaccine'] = vax
    df.set_index('Vaccine', append=True, inplace=True)
    dfComb = dfComb.append(df).sort_index()
# Drop sums that don't make sense
dfComb.drop([('Vaccination rate', 'sum'),
             ('Vaccine missed opportunities rate', 'sum')],
            axis=1,
            inplace=True)
# Format columns
idx = pd.IndexSlice
dfComb = format_with_multiindex(
    dfComb, multiCols=idx[:"Vaccine missed opportunities", :], fmt="{:,.2f}")
dfComb = format_with_multiindex(dfComb,
                                multiCols=idx["Vaccination rate":, :],
                                fmt="{:,.4f}")
display(dfComb)

# https://pbpython.com/improve-pandas-excel-output.html
writer = pd.ExcelWriter(os.path.join(projectRoot, 'reports', 'tables',
                                     'combined_outcomes.xlsx'))
#                         engine='xlsxwriter')
dfComb.to_excel(writer, sheet_name='combined')

for c in list(dfComb.columns.levels[0]):
    if c == 'Vaccine':
        pass
    else:
        df = dfComb.loc[:,slice(c,c)]
        display(df)
        df.to_excel(writer, c[:15].replace(" ", "_"))
writer.save()


# In[95]:


idx = pd.IndexSlice
cols = (['Patients', 'Vaccine eligible', 'Immunized', 'Vaccination rate'], [
    'Number of visits during time period', 'Vaccine administered at a visit',
    'Vaccine missed opportunities rate'
])
for ls in cols:
    df = dfComb.loc[idx[:, :, ], idx[ls, ['mean', 'std']]]
    display(df)
    #     df.to_excel(os.path.join(projectRoot, 'reports', 'tables', 'outcomes'+ str(len(ls)) +'.xlsx'))


# In[96]:


# Patients by location
df = dfComb['Patients'].droplevel('Vaccine').drop_duplicates()
df.to_excel(
    os.path.join(projectRoot, 'reports', 'tables', 'patientsXloc.xlsx'))


# ### Differences before/after Intervention

# #### ANOVA tests of outcomes
# 

# ##### Compare WIC selection
# N=80 is ANOVA of strategy 2 (Exclude)  
# N=72 is ANOVA of strategy 3 (Drop)  
# Both return similar results.  
# Use Strategy 2 dfNoWic  

# In[97]:


def find_sigs(df, groupers, outcomes, vaxVars, residual=False, show=False):
    dfResults = pd.DataFrame()
    for vax in vaxVars:
        cols = [
            outcome for outcome in outcomes if outcome.startswith(vax + '_')
        ] + groupers
        for col in cols:
            if col not in groupers:
                model = ols(col + ' ~ C(BASELINE)', data=df[cols]).fit()
                summary = model.summary()
                aov_table = sm.stats.anova_lm(model, typ=2)
                aov_table['Outcome'] = col
                dfResults = dfResults.append(aov_table)
                if show == True:
                    display(model.summary())
    dfResults = dfResults.reset_index().rename(columns={'index': 'Group'})
    if residual == False:
        dfResults = dfResults[~(dfResults['Group'] == 'Residual')]
    return dfResults


# In[98]:


# Drop location over 5% difference
dfData = diDfs['dfDropWic_Outcomes_Locs']
len(dfData)
vaxVars = ['FLU', 'PNEUMO', 'PCV', 'TDAP', 'TD', 'ZOST']
idx = ['Outcome']
df = find_sigs(dfData,
               groupers=['BASELINE'],
               outcomes=outcomes,
               vaxVars=vaxVars,
               show=True).replace(columnLabels).rename(columnLabels,
                                                       axis=1).round(4)
df = df[df['PR(>F)'] <= .05]
df.set_index(idx)


# In[99]:


# ANOVA of strategy 2 (Exclude)
dfData = diDfs['dfNoWic_Outcomes_Locs']
len(dfData)
vaxVars = ['FLU', 'PNEUMO', 'PCV', 'TDAP', 'TD', 'ZOST']
idx = ['Outcome']
dfResult = find_sigs(dfData,
                     groupers=['BASELINE'],
                     outcomes=outcomes,
                     vaxVars=vaxVars).replace(columnLabels).rename(
                         columnLabels, axis=1).round(4)
dfResult = dfResult[dfResult['PR(>F)'] <= .05]
dfResult.set_index(idx)


# ### All results

# In[100]:


# Define data
dfData = diDfs['dfNoWic_Outcomes_Locs']
# display(dfData.set_index(['LOC_ID', 'BASELINE']).sort_index().head())
vaxVars = ['FLU', 'PNEUMO', 'PCV', 'TDAP', 'TD', 'ZOST']
idx = ['Group', 'Outcome']
dfResult = find_sigs(dfData,
                     groupers=['BASELINE'],
                     outcomes=outcomes,
                     vaxVars=vaxVars,
                     show=False).round(4)

# Generate analysis table
dfAllResults = pd.DataFrame()
for vax in vaxVars:
    # Build df of outcome columns from observations
    cols = ['BASELINE', 'LOC_ID']
    cols.extend([
        c for c in list(dfData)
        if ((vax + '_' in c) and (c in list(dfResult['Outcome'])))
    ])
    dfAgg = dfData[cols].groupby('BASELINE').agg('mean')
    dfAgg = dfAgg.sort_index(ascending=False).drop('LOC_ID', axis=1)
    colFilter = [
        c for c in list(dfAgg)
        if ((c.endswith('VAX_RATE') | c.endswith('MISSED_OPS_RATE')))
    ]
    # Join observations with stats
    if list(dfAgg[colFilter]) != []:
        dfAllRes = dfResult[dfResult['Outcome'].isin(list(dfAgg[colFilter]))]
        dfAllRes = dfAllRes.merge(dfAgg.round(4).transpose().reset_index(),
                                  left_on='Outcome',
                                  right_on='index')
        # Add classifier columns
        dfAllRes['Vaccine'] = vax
        dfAllRes['Outcome'] = dfAllRes['Outcome'].str.replace(vax + '_', "")
        dfAllResults = dfAllResults.append(dfAllRes)
dfAllResults = dfAllResults[[
    'Outcome', 'Baseline', 'Follow-up', 'sum_sq', 'df', 'F', 'PR(>F)',
    'Vaccine'
]].replace(columnLabels).rename(columnLabels, axis=1)

# Mark significance
mask = ((dfAllResults['PR(>F)'] < .05))
dfAllResults.loc[mask, 'Significant < .05'] = '*'
mask = ((dfAllResults['Outcome'] == 'Vaccination rate') &
        (dfAllResults['PR(>F)'] < .05) & ((dfAllResults['Follow-up']) >
                                          (dfAllResults['Baseline'])))
dfAllResults.loc[mask, 'Significant < .05 in expected direction'] = '*'
mask = ((dfAllResults['Outcome'] == 'Vaccine missed opportunities rate') &
        (dfAllResults['PR(>F)'] < .05) & ((dfAllResults['Follow-up']) <
                                          (dfAllResults['Baseline'])))
dfAllResults.loc[mask, 'Significant < .05 in expected direction'] = '*'
dfAllResults = dfAllResults.fillna('')
display(dfAllResults.set_index(['Outcome', 'Vaccine']).sort_index())
dfAllResults.set_index(['Outcome', 'Vaccine']).sort_index().to_excel(
    os.path.join(projectRoot, 'reports', 'tables', 'allresults.xlsx'))


# In[ ]:




