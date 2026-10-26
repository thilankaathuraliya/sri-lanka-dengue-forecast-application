"""Historical NGBoost dengue prediction dashboard. Experimental, not a public health service."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Sri Lankan Dengue — NGBoost', page_icon='🦟', layout='wide')
st.title('Sri Lankan Dengue Prediction Dashboard')
st.caption('MSc Machine Learning | NGBoost probabilistic regression | Historical research demonstration')
st.warning('Experimental historical forecast only — not a current dengue alert or validated public-health forecast. Nominal 90% intervals were under-calibrated (69.2% historical coverage).')

needed = ['dengue_ngboost_bundle.joblib', 'dengue_ngboost_metadata.json', 'data/dengue_history.xlsx']
missing = [x for x in needed if not (ROOT/x).is_file()]
if missing:
    st.error('Missing required files: ' + ', '.join(missing))
    st.info('Download these files from Section 10 of the NGBoost Colab notebook and upload them to the repository root. Keep the dataset inside data/.')
    st.stop()

@st.cache_resource
def load_model():
    bundle = joblib.load(ROOT/'dengue_ngboost_bundle.joblib')
    with open(ROOT/'dengue_ngboost_metadata.json', encoding='utf-8') as f:
        metadata = json.load(f)
    assert all(k in bundle for k in ('preprocessor','model','features'))
    if bundle['features'] != metadata['features']:
        raise ValueError('Feature order differs between model bundle and metadata')
    return bundle, metadata

@st.cache_data
def load_data():
    data = pd.read_excel(ROOT/'data/dengue_history.xlsx', sheet_name='All Data', usecols='A:C')
    data.columns = ['Date','City','Value']
    data['Date'] = pd.to_datetime(data.Date, errors='coerce').dt.to_period('M').dt.to_timestamp()
    data['City'] = data.City.astype('string').str.strip().astype(str)
    data['Value'] = pd.to_numeric(data.Value, errors='coerce')
    if data[['Date','City','Value']].isna().any().any() or data.duplicated(['City','Date']).any() or (data.Value<0).any():
        raise ValueError('Invalid historical data: missing, duplicate or negative values')
    return data.sort_values(['City','Date']).reset_index(drop=True)

try:
    bundle, metadata = load_model()
    df = load_data()
except Exception as exc:
    st.exception(exc)
    st.stop()

last_month = df.Date.max()
forecast_month = last_month + pd.DateOffset(months=1)
if metadata.get('forecast_month') and metadata['forecast_month'] != str(forecast_month.date()):
    st.error('The model forecast month and dataset last month do not match. Check your exported files.')
    st.stop()

cities = sorted(df.City.unique())
city = st.sidebar.selectbox('Select location', cities, index=cities.index('Colombo') if 'Colombo' in cities else 0)
st.sidebar.markdown('**Forecast mode:** One month ahead from last recorded data')
st.sidebar.caption(f'Last observed: {last_month:%B %Y}  |  Forecast: {forecast_month:%B %Y}')
st.sidebar.caption(f'Model trained through: {metadata.get("training_end", "2018-12")}')
st.sidebar.info('The model was fitted on data before 2019. Its January 2021 forecast uses observed lag values through December 2020.')

history = df.loc[df.City == city].sort_values('Date').copy()
if len(history) < 12:
    st.error('At least 12 consecutive months of historical observations are required.')
    st.stop()
expected = pd.date_range(history.Date.min(), history.Date.max(), freq='MS')
if not history.Date.reset_index(drop=True).equals(pd.Series(expected)):
    st.error('Incomplete monthly city history; cannot construct consistent lag features.')
    st.stop()

st.subheader('Interactive NGBoost prediction')
st.caption('Historical one-month-ahead scenario. Change December 2020 cases to explore how the January 2021 prediction responds. Other earlier months remain observed.')
observed_last = float(history.Value.iloc[-1])
with st.form('interactive_forecast'):
    input_city = st.text_input('Selected location', value=city, disabled=True)
    input_month = st.text_input('Forecast month', value=f'{forecast_month:%Y-%m}', disabled=True)
    edited_last = st.number_input(
        f'Dengue cases in {last_month:%B %Y} (previous month)',
        min_value=0, max_value=1000000,
        value=int(observed_last), step=1,
        help='Overrides only the latest observed month for this scenario; older historical observations are unchanged.'
    )
    submitted = st.form_submit_button('Predict dengue cases', type='primary')

# Streamlit reruns on form submission. All features are recomputed from the same
# scenario history, using only observations prior to the forecast month.
scenario = history.copy()
scenario.loc[scenario.index[-1], 'Value'] = float(edited_last)
v = scenario.Value.to_numpy(dtype=float)
assert scenario.Date.max() == last_month
assert forecast_month == last_month + pd.DateOffset(months=1)
assert np.isfinite(v).all() and (v >= 0).all()
row = {'City': city, 'Date': forecast_month,
       'month_sin': np.sin(2*np.pi*forecast_month.month/12),
       'month_cos': np.cos(2*np.pi*forecast_month.month/12)}
for lag in (1,2,3,6,12):
    row[f'lag_{lag}'] = float(v[-lag])
for w in (3,6,12):
    row[f'rolling_mean_{w}'] = float(v[-w:].mean())
    row[f'rolling_std_{w}'] = float(v[-w:].std(ddof=0))
row['recent_change'] = float(v[-1]-v[-2])
features = bundle['features']
inputs = pd.DataFrame([row])
if any(f not in inputs.columns for f in features):
    st.error('The exported model requests an unsupported feature. Verify its metadata.')
    st.stop()
try:
    transformed = bundle['preprocessor'].transform(inputs[features])
    dist = bundle['model'].pred_dist(transformed)
    mu = float(np.asarray(dist.loc).reshape(-1)[0])
    sigma = float(np.asarray(dist.scale).reshape(-1)[0])
    if not (np.isfinite(mu) and np.isfinite(sigma) and sigma >= 0):
        raise ValueError('NGBoost returned invalid distribution parameters')
    point = max(0.0, float(np.expm1(mu)))
    lower = max(0.0, float(np.expm1(mu + norm.ppf(.05)*sigma)))
    upper = max(0.0, float(np.expm1(mu + norm.ppf(.95)*sigma)))
except Exception as exc:
    st.error('Prediction failed. Confirm the exported bundle and installed dependency versions match Colab.')
    st.exception(exc)
    st.stop()

if edited_last != observed_last:
    st.info(f'Scenario: {last_month:%B %Y} changed from {observed_last:,.0f} to {edited_last:,.0f} cases. The lag and rolling features below were recalculated. This is a what-if scenario, not a validated causal intervention.')
else:
    st.caption('Using the original recorded previous-month value. Change it and click Predict dengue cases to explore a scenario.')
st.subheader(f'{city} — {forecast_month:%B %Y} historical forecast')
a,b,c,d = st.columns(4)
a.metric('NGBoost median prediction', f'{point:,.0f}')
b.metric('Previous-month baseline', f'{v[-1]:,.0f}')
c.metric('Nominal 90% lower', f'{lower:,.0f}')
d.metric('Nominal 90% upper', f'{upper:,.0f}')
st.caption('Predictions rounded for display. Interval is model-based, not calibrated to 90% observed coverage.')
with st.expander('Show features sent to NGBoost'):
    st.dataframe(inputs[features].T.rename(columns={0: 'Input value'}), use_container_width=True)

st.subheader('Recorded dengue cases')
fig=go.Figure()
fig.add_trace(go.Scatter(x=history.Date,y=history.Value,mode='lines',name='Recorded cases',line=dict(color='#2471A3')))
fig.add_trace(go.Scatter(x=[forecast_month],y=[point],mode='markers',name='NGBoost forecast',marker=dict(color='#F39C12',size=13)))
fig.add_trace(go.Scatter(x=[forecast_month],y=[v[-1]],mode='markers',name='Previous-month baseline',marker=dict(color='#239B56',size=11)))
fig.add_trace(go.Scatter(x=[forecast_month,forecast_month],y=[lower,upper],mode='lines+markers',name='Nominal 90% interval',line=dict(color='#E67E22',width=5)))
fig.update_layout(xaxis_title='Month',yaxis_title='Reported dengue cases',height=460,legend=dict(orientation='h'))
st.plotly_chart(fig,use_container_width=True)

st.subheader('Historical evaluation — 2019–2020')
st.caption('Previously viewed historical period; NOT a fresh blind test. NGBoost MAE 87.054; previous-month MAE 97.037; NGBoost R² 0.751. Empirical coverage of nominal 90% intervals: 69.2%.')
eval_file = ROOT/'dengue_ngboost_historical_predictions.csv'
if eval_file.is_file():
    historical=pd.read_csv(eval_file,parse_dates=['Date'])
    if {'Date','Value','NGBoost','lag_1'}.issubset(historical.columns):
        monthly=historical.groupby('Date',as_index=False)[['Value','NGBoost','lag_1']].sum()
        eval_fig=go.Figure()
        for col,label in [('Value','Observed'),('NGBoost','NGBoost'),('lag_1','Previous month')]:
            eval_fig.add_trace(go.Scatter(x=monthly.Date,y=monthly[col],name=label,mode='lines'))
        eval_fig.update_layout(yaxis_title='Cases across all locations',height=410)
        st.plotly_chart(eval_fig,use_container_width=True)
else:
    st.info('Optional: add dengue_ngboost_historical_predictions.csv to display the historical comparison graph.')

st.subheader('Model explainability — 2018 validation')
st.caption('Permutation importance from the completed NGBoost Colab experiment. Values are recorded results, not recomputed by this dashboard. Positive values mean shuffling increased MAE; negative values mean shuffling decreased MAE. Not causal evidence.')
importance_file = ROOT/'dengue_ngboost_permutation_importance.csv'
if importance_file.is_file():
    importance = pd.read_csv(importance_file)
    if not {'Feature','MAE increase'}.issubset(importance.columns):
        st.error('Importance CSV is missing Feature or MAE increase columns.')
        st.stop()
    importance = importance.sort_values('MAE increase', ascending=False)
else:
    st.caption('Displaying previously recorded validation importance results (optional CSV not present).')
    importance = pd.DataFrame([
        ('lag_1', 124.819), ('month_cos', 11.560), ('month_sin', 5.000),
        ('City', 1.694), ('lag_3', 0.854), ('rolling_mean_6', -4.441),
        ('lag_12', -4.735), ('rolling_mean_3', -5.350)
    ], columns=['Feature', 'MAE increase'])
bar = go.Figure(go.Bar(x=importance['MAE increase'], y=importance['Feature'], orientation='h'))
bar.update_layout(xaxis_title='Change in validation MAE (cases)', height=390, yaxis=dict(autorange='reversed'))
st.plotly_chart(bar, use_container_width=True)
with st.expander('View permutation importance values'):
    st.dataframe(importance, hide_index=True)

with st.expander('Model details and limitations'):
    st.json(metadata)
    st.markdown('**Limitations:** historical observational data, limited external predictors, imperfect outbreak sensitivity, and poorly calibrated model-based prediction intervals. The model predicts the inverse-transformed log-scale median, not the arithmetic mean.')
