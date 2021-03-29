# Streamlit app
import streamlit as st
# Stats model modules
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import pacf, acf
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Plotly modules
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff


@st.cache
def load_data():
    dta = sm.datasets.sunspots.load_pandas().data
    dta.YEAR = pd.to_datetime(dta.YEAR, format='%Y')
    dta = dta.set_index("YEAR")
    return dta


dta = load_data()  # sm.datasets.sunspots.load_pandas().data

# dta = dta.sort_values(by="YEAR", ascending=True)
st.title("Time series prediction")
st.subheader("Data set: Yearly sun activity since 1700")


# @st.cache
# def general_plot(dta):
# SCATTER AND LINES OF THE TS
fig = go.Figure()
fig = px.line(dta, x=dta.index, y="SUNACTIVITY")
# return fig
st.write(fig)  # (general_plot(dta))

st.subheader("Below are the ACF and PCF plots of the time series")
# ACF and PCF plots
# fig = plt.figure(figsize=(10, 10))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)
# st.write(fig)

# ACF


@st.cache
def plot_lag_correlations(cf_series, name):

    ylabel = "Autocorrelation" if name == "ACF" else "Partial Autocorrelation"
    df_acf = cf_series
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(df_acf)),
        y=df_acf,
        name=f'{name}', mode='markers'
    ))
    # Add shapes
    for i in range(len(df_acf)):
        fig.add_shape(type="line",
                      x0=i, y0=0, x1=i, y1=df_acf[i],
                      line=dict(color="RoyalBlue", width=0.9)
                      )
    # fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        title=f"{name}",
        xaxis_title="Lag",
        yaxis_title=ylabel,
        height=600,
    )
    return fig


# Create the plot with the function and show it
acf_plot = plot_lag_correlations(acf(dta['SUNACTIVITY'], nlags=100), 'ACF')
st.write(acf_plot)
pacf_plot = plot_lag_correlations(pacf(dta['SUNACTIVITY'], nlags=100), 'PACF')
st.write(pacf_plot)


# Parameters of ARIMA as user input
st.sidebar.subheader("Chosing parameters of ARIMA(p,q,d)")
p = st.sidebar.slider('p', 0, 5, 3)
q = st.sidebar.slider('q', 0, 5, 1)
d = st.sidebar.slider('d', 0, 5, 0)

# Fitting the model
st.subheader("Fitting an ARMA model")
st.write("Estimated parameters from the input model:")


# @st.cache
# def fit_model(p, q, d):
arma_mod = sm.tsa.statespace.SARIMAX(
    dta[["SUNACTIVITY"]], order=(p, q, d), trend='c').fit(disp=False)
# eturn(arma_mod)


# Fit model if parameters change
# arma_mod = fit_model(p, q, d)
st.write(arma_mod.params.reset_index().rename(columns={0: 'estimate'}))

# st.write(arma_mod.resid.reset_index())
# Getting the residuals
residuals = arma_mod.resid.reset_index().rename(
    columns={0: 'Residual', 'YEAR': 'Year'})
# Line plot of residuals
st.subheader("Residuals plot")
fig = px.line(residuals, x="Year", y="Residual")
st.write(fig)

# Distribution of residuals
st.subheader("Residuals distribution")
hist_data = [residuals.Residual]
group_labels = ['distplot']  # name of the dataset
fig = ff.create_distplot(hist_data, group_labels, show_rug=False)
fig.update_layout(title_text='Hist and Curve Plot of Residuals')
st.write(fig)

# Start and end of predictions as sliders
st.sidebar.subheader("Chosing Start and End of prediction horizon")
start_pred = st.sidebar.slider('Horizon start', 1990, 2000, 1990)
end_pred = st.sidebar.slider('Horizon end', 2001, 2012, 2010)


@ st.cache
def predict(start_pred, end_pred):
    return arma_mod.predict(
        start=str(start_pred), end=str(end_pred), dynamic=True).reset_index().rename(columns={'index': 'YEAR'})


predict_sunspots = predict(start_pred, end_pred)


# Plot predicted against true values
fig = go.Figure(layout=go.Layout(height=600, width=800))
fig.add_trace(go.Scatter(x=dta.loc['1960':].index, y=dta['1960':].SUNACTIVITY,
                         mode='lines+markers',
                         name='lines+markers'))
fig.add_trace(go.Scatter(x=predict_sunspots.YEAR, y=predict_sunspots.predicted_mean,
                         mode='lines+markers',
                         name='lines+markers'))

st.write(fig)
