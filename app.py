import pandas as pd
from fbprophet import Prophet

# Read data
df = pd.read_csv('data.csv')
holidays = pd.read_csv('holidays.csv')

# Feature select
cols = {'日期':'ds', '尖峰負載(MW)':'y'}
df = df.rename(columns=cols)[cols.values()]
df['ds'] = df['ds'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

# Build and train model
m = Prophet(holidays=holidays)
m.fit(df)

# Predict
future = m.make_future_dataframe(periods=7)
forecast = m.predict(future)

# Save results
cols = {'ds': 'date', 'yhat': 'peak_load(MW)'}
forecast = forecast[forecast['ds'].between('2019-04-02', '2019-04-08')].rename(columns=cols)[cols.values()]
forecast['date'] = forecast['date'].apply(lambda x: x.strftime('%Y%m%d'))
forecast['peak_load(MW)'] = forecast['peak_load(MW)'].apply(lambda x: round(x))
forecast.to_csv('submission.csv', index=False)
