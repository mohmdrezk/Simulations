import plotly 
import plotly.graph_objs as go
import numpy as np

def make_plotly_fig(dfx):
    
    num_categs = dfx['y'].nunique()
    
    fig = plotly.tools.make_subplots(rows=1, cols=1, subplot_titles=('feature_1 vs feature_2'))
    fig['layout'].update(title='feature_1 vs feature_2',
                         xaxis=dict(title='feature_1'),
                         yaxis=dict(title='feature_2'))
    for i_cat in range(num_categs):
        fig.append_trace(
            {'x': dfx.loc[dfx['y']==i_cat,'feature_1'],
             'y': dfx.loc[dfx['y']==i_cat,'feature_2'],
             'mode': 'markers',
             'type': 'scatter',
             'name': f'Category {i_cat}'},
            row=1, col=1)
    return fig


def make_figure2(df):
    
    # calculate standard error of the mean
    sem = np.std(df, axis=0, ddof=1) / np.sqrt(df.shape[0])*100
    sem = sem.values.tolist()
    
    average_values = df.mean()*100
    average_values = average_values.values.tolist()
    fig = go.Figure(data=go.Scatter(
            x= ['Without Averaging' ,'With Averaging'],
            y=average_values,
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=sem,
                visible=True)
        ))
    
    
    
    return fig