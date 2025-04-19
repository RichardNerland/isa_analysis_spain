import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from simple_isa_model import run_simple_simulation, _create_degree_definitions

# Initialize the Dash application
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Get the base degree definitions to use for defaults
base_degrees = _create_degree_definitions()

# Define the layout
app.layout = html.Div([
    html.H1("ISA Model Simulation Dashboard", style={'textAlign': 'center'}),
    
    # Main container with sidebar and content
    html.Div([
        # Sidebar with inputs
        html.Div([
            html.H3("Simulation Parameters"),
            
            # Program selection
            html.Label("Program Type"),
            dcc.RadioItems(
                id='program-type',
                options=[
                    {'label': 'Ecuador', 'value': 'Ecuador'},
                    {'label': 'Guatemala', 'value': 'Guatemala'}
                ],
                value='Ecuador',
                style={'marginBottom': '20px'}
            ),
            
            # Core simulation parameters
            html.Label("Number of Students"),
            dcc.Input(
                id='num-students',
                type='number',
                value=200,
                min=1,
                max=1000,
                step=1,
                style={'width': '100%', 'marginBottom': '15px'}
            ),
            
            html.Label("Number of Simulations"),
            dcc.Input(
                id='num-sims',
                type='number',
                value=20,
                min=1,
                max=100,
                step=1,
                style={'width': '100%', 'marginBottom': '15px'}
            ),
            
            html.Label("Simulation Years"),
            dcc.Input(
                id='num-years',
                type='number',
                value=25,
                min=1,
                max=50,
                step=1,
                style={'width': '100%', 'marginBottom': '15px'}
            ),
            
            html.Label("Apply Graduation Delay"),
            dcc.RadioItems(
                id='apply-graduation-delay',
                options=[
                    {'label': 'Yes', 'value': True},
                    {'label': 'No', 'value': False}
                ],
                value=False,
                style={'marginBottom': '20px'}
            ),
            
            # Economic parameters
            html.H4("Economic Parameters"),
            
            html.Label("Initial Unemployment Rate"),
            html.Div(style={'marginBottom': '20px'}, children=[
                dcc.Slider(
                    id='unemployment-rate',
                    min=0.01,
                    max=0.2,
                    step=0.01,
                    value=0.08,
                    marks={i/100: f'{i}%' for i in range(1, 21, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ]),
            
            html.Label("Initial Inflation Rate"),
            html.Div(style={'marginBottom': '20px'}, children=[
                dcc.Slider(
                    id='inflation-rate',
                    min=0.0,
                    max=0.1,
                    step=0.005,
                    value=0.02,
                    marks={i/100: f'{i}%' for i in range(0, 11, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ]),
            
            # ISA parameters
            html.H4("ISA Parameters"),
            
            html.Label("ISA Percentage"),
            html.Div(style={'marginBottom': '20px'}, children=[
                dcc.Slider(
                    id='isa-percentage',
                    min=0.05,
                    max=0.2,
                    step=0.01,
                    value=0.1,
                    marks={i/100: f'{i}%' for i in range(5, 21, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ]),
            
            html.Label("ISA Threshold ($)"),
            dcc.Input(
                id='isa-threshold',
                type='number',
                value=13000,
                min=5000,
                max=30000,
                step=1000,
                style={'width': '100%', 'marginBottom': '15px'}
            ),
            
            html.Label("ISA Cap ($)"),
            dcc.Input(
                id='isa-cap',
                type='number',
                min=5000,
                max=100000,
                step=1000,
                style={'width': '100%', 'marginBottom': '15px'}
            ),
            
            html.Label("ISA Payment Years Limit"),
            dcc.Input(
                id='limit-years',
                type='number',
                value=10,
                min=1,
                max=20,
                step=1,
                style={'width': '100%', 'marginBottom': '15px'}
            ),
            
            # Cost parameters
            html.H4("Cost and Fee Parameters"),
            
            html.Label("Cost per Student ($)"),
            dcc.Input(
                id='price-per-student',
                type='number',
                min=1000,
                max=50000,
                step=500,
                style={'width': '100%', 'marginBottom': '15px'}
            ),
            
            html.Label("Performance Fee (%)"),
            html.Div(style={'marginBottom': '20px'}, children=[
                dcc.Slider(
                    id='performance-fee',
                    min=0.0,
                    max=0.1,
                    step=0.005,
                    value=0.025,
                    marks={i/100: f'{i}%' for i in range(0, 11, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ]),
            
            html.Label("Annual Fee per Student ($)"),
            dcc.Input(
                id='annual-fee',
                type='number',
                value=300,
                min=0,
                max=1000,
                step=50,
                style={'width': '100%', 'marginBottom': '15px'}
            ),
            
            # Program-specific parameters
            html.Div(id='program-specific-params'),
            
            # Run simulation button
            html.Div(id='run-button-container'),
            
            # Status display
            html.Div(id='simulation-status', style={'marginTop': '15px'})
            
        ], style={'width': '25%', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
        
        # Main content area - Tabs for different visualizations
        html.Div([
            dcc.Tabs([
                dcc.Tab(label='Financial Summary', children=[
                    html.Div([
                        html.Div(id='financial-metrics', style={'marginTop': '20px'}),
                        html.Div([
                            dcc.Graph(id='annual-payments-graph'),
                            dcc.Graph(id='cumulative-returns-graph')
                        ], style={'display': 'flex', 'flexDirection': 'column'})
                    ])
                ]),
                
                dcc.Tab(label='Student Outcomes', children=[
                    html.Div([
                        html.Div(id='student-metrics', style={'marginTop': '20px'}),
                        dcc.Graph(id='active-students-graph'),
                        dcc.Graph(id='cap-stats-graph')
                    ])
                ]),
                
                dcc.Tab(label='Pathway Analysis', children=[
                    html.Div([
                        html.Div(id='pathway-metrics', style={'marginTop': '20px'}),
                        dcc.Graph(id='pathway-sankey-graph')
                    ])
                ]),
                
                dcc.Tab(label='Degree Parameters', children=[
                    html.Div([
                        html.H3("Modify Degree Parameters"),
                        html.Div(id='degree-params-container')
                    ])
                ])
            ], style={'marginTop': '20px'})
        ], style={'width': '75%', 'padding': '20px'})
    ], style={'display': 'flex', 'marginTop': '20px'}),
    
    # Store for simulation results and degree parameters
    dcc.Store(id='simulation-results'),
    dcc.Store(id='degree-params')
])

# Callback to update program-specific parameters based on program selection
@callback(
    Output('program-specific-params', 'children'),
    Output('degree-params', 'data'),
    Output('run-button-container', 'children'),
    Input('program-type', 'value')
)
def update_program_params(program_type):
    if program_type == 'Ecuador':
        params = html.Div([
            html.H4("Ecuador Program Parameters"),
            
            html.Label("Year 1 Completion Probability"),
            html.Div(style={'marginBottom': '20px'}, children=[
                dcc.Slider(
                    id='ecu-year1-completion',
                    min=0.5,
                    max=1.0,
                    step=0.05,
                    value=0.9,
                    marks={i/10: f'{i/10:.1f}' for i in range(5, 11, 1)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ]),
            
            html.Label("Placement Probability"),
            html.Div(style={'marginBottom': '20px'}, children=[
                dcc.Slider(
                    id='ecu-placement',
                    min=0.4,
                    max=1.0,
                    step=0.05,
                    value=0.8,
                    marks={i/10: f'{i/10:.1f}' for i in range(4, 11, 1)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ]),
            
            html.Label("NA Completion Probability"),
            html.Div(style={'marginBottom': '20px'}, children=[
                dcc.Slider(
                    id='ecu-na-completion',
                    min=0.4,
                    max=1.0,
                    step=0.05,
                    value=0.85,
                    marks={i/10: f'{i/10:.1f}' for i in range(4, 11, 1)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ])
        
        run_button = html.Button(
            'Run Ecuador Simulation', 
            id='run-ecuador-sim', 
            n_clicks=0, 
            style={'width': '100%', 'marginTop': '20px', 'padding': '10px',
                   'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none'}
        )
        
        # Default price for Ecuador
        default_price = 9000
        default_cap = 27000
        
    else:  # Guatemala
        params = html.Div([
            html.H4("Guatemala Program Parameters"),
            
            html.Label("Placement Probability"),
            html.Div(style={'marginBottom': '20px'}, children=[
                dcc.Slider(
                    id='guat-placement',
                    min=0.4,
                    max=1.0,
                    step=0.05,
                    value=0.85,
                    marks={i/10: f'{i/10:.1f}' for i in range(4, 11, 1)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ]),
            
            html.Label("Advancement Probability"),
            html.Div(style={'marginBottom': '20px'}, children=[
                dcc.Slider(
                    id='guat-advancement',
                    min=0.2,
                    max=0.8,
                    step=0.05,
                    value=0.4,
                    marks={i/10: f'{i/10:.1f}' for i in range(2, 9, 1)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ])
        
        run_button = html.Button(
            'Run Guatemala Simulation', 
            id='run-guatemala-sim', 
            n_clicks=0, 
            style={'width': '100%', 'marginTop': '20px', 'padding': '10px',
                   'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none'}
        )
        
        # Default price for Guatemala
        default_price = 7500
        default_cap = 22500
    
    # Create a copy of the base degree definitions for this program
    degree_params = base_degrees.copy()
    
    return params, degree_params, run_button

# Callback to update price per student and ISA cap based on program selection
@callback(
    Output('price-per-student', 'value'),
    Output('isa-cap', 'value'),
    Input('program-type', 'value'),
    prevent_initial_call=True
)
def update_price_and_cap(program_type):
    if program_type == 'Ecuador':
        return 9000, 27000
    else:  # Guatemala
        return 7500, 22500

# Callback to update degree parameters UI
@callback(
    Output('degree-params-container', 'children'),
    Input('degree-params', 'data')
)
def update_degree_params_ui(degree_params):
    if not degree_params:
        return html.Div("No degree parameters available.")
    
    # Create input fields for each degree's parameters
    degree_inputs = []
    
    for name, params in degree_params.items():
        degree_inputs.append(html.Div([
            html.H4(f"Degree: {name}"),
            
            html.Label("Mean Earnings ($)"),
            dcc.Input(
                id=f'{name}-mean-earnings',
                type='number',
                value=params['mean_earnings'],
                min=0,
                max=100000,
                step=500,
                style={'width': '100%', 'marginBottom': '15px'}
            ),
            
            html.Label("Standard Deviation ($)"),
            dcc.Input(
                id=f'{name}-stdev',
                type='number',
                value=params['stdev'],
                min=0,
                max=20000,
                step=100,
                style={'width': '100%', 'marginBottom': '15px'}
            ),
            
            html.Label("Experience Growth Rate"),
            html.Div(style={'marginBottom': '20px'}, children=[
                dcc.Slider(
                    id=f'{name}-growth',
                    min=0.0,
                    max=0.1,
                    step=0.005,
                    value=params['experience_growth'],
                    marks={i/100: f'{i}%' for i in range(0, 11, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ]),
            
            html.Hr(style={'marginTop': '20px', 'marginBottom': '20px'})
        ], style={'marginBottom': '20px'}))
    
    return html.Div(degree_inputs)

# Callback for Ecuador simulation
@callback(
    Output('simulation-results', 'data', allow_duplicate=True),
    Output('simulation-status', 'children', allow_duplicate=True),
    Input('run-ecuador-sim', 'n_clicks'),
    State('program-type', 'value'),
    State('num-students', 'value'),
    State('num-sims', 'value'),
    State('num-years', 'value'),
    State('apply-graduation-delay', 'value'),
    State('unemployment-rate', 'value'),
    State('inflation-rate', 'value'),
    State('isa-percentage', 'value'),
    State('isa-threshold', 'value'),
    State('isa-cap', 'value'),
    State('limit-years', 'value'),
    State('price-per-student', 'value'),
    State('performance-fee', 'value'),
    State('annual-fee', 'value'),
    # Ecuador params
    State('ecu-year1-completion', 'value'),
    State('ecu-placement', 'value'),
    State('ecu-na-completion', 'value'),
    prevent_initial_call='initial_duplicate'
)
def run_ecuador_simulation(n_clicks, program_type, num_students, num_sims, num_years, apply_graduation_delay,
                   unemployment_rate, inflation_rate, isa_percentage, isa_threshold, isa_cap,
                   limit_years, price_per_student, performance_fee, annual_fee,
                   ecu_year1_completion, ecu_placement, ecu_na_completion):
    
    if n_clicks > 0:
        # Set up parameters dictionary
        params = {
            'program_type': 'Ecuador',  # Force Ecuador since this is the Ecuador simulation callback
            'num_students': num_students,
            'num_sims': num_sims,
            'num_years': num_years,
            'apply_graduation_delay': apply_graduation_delay,
            'initial_unemployment_rate': unemployment_rate,
            'initial_inflation_rate': inflation_rate,
            'isa_percentage': isa_percentage,
            'isa_threshold': isa_threshold,
            'limit_years': limit_years,
            'performance_fee_pct': performance_fee,
            'annual_fee_per_student': annual_fee,
        }
        
        # Set default values for cap and price if they're not provided
        if isa_cap is None:
            params['isa_cap'] = 27000  # Default for Ecuador
        else:
            params['isa_cap'] = isa_cap
            
        if price_per_student is None:
            params['price_per_student'] = 9000  # Default for Ecuador
        else:
            params['price_per_student'] = price_per_student
        
        # Add Ecuador-specific parameters
        params.update({
            'ecu_year1_completion_prob': ecu_year1_completion,
            'ecu_placement_prob': ecu_placement,
            'ecu_na_completion_prob': ecu_na_completion,
            # Add defaults for Guatemala parameters
            'guat_placement_prob': 0.85,
            'guat_advancement_prob': 0.40,
        })
        
        # Set a random seed for reproducibility
        params['random_seed'] = 42
        
        try:
            # Run the simulation
            results = run_simple_simulation(**params)
            
            # Format the results for storage in dcc.Store
            formatted_results = {}
            for key, value in results.items():
                if isinstance(value, pd.Series):
                    formatted_results[key] = value.to_dict()
                elif isinstance(value, pd.DataFrame):
                    formatted_results[key] = value.to_dict('list')
                else:
                    formatted_results[key] = value
            
            return formatted_results, html.Div("Ecuador simulation completed successfully!", style={'color': 'green'})
        
        except Exception as e:
            return None, html.Div(f"Error: {str(e)}", style={'color': 'red'})
    
    return None, html.Div()

# Callback for Guatemala simulation
@callback(
    Output('simulation-results', 'data', allow_duplicate=True),
    Output('simulation-status', 'children', allow_duplicate=True),
    Input('run-guatemala-sim', 'n_clicks'),
    State('program-type', 'value'),
    State('num-students', 'value'),
    State('num-sims', 'value'),
    State('num-years', 'value'),
    State('apply-graduation-delay', 'value'),
    State('unemployment-rate', 'value'),
    State('inflation-rate', 'value'),
    State('isa-percentage', 'value'),
    State('isa-threshold', 'value'),
    State('isa-cap', 'value'),
    State('limit-years', 'value'),
    State('price-per-student', 'value'),
    State('performance-fee', 'value'),
    State('annual-fee', 'value'),
    # Guatemala params
    State('guat-placement', 'value'),
    State('guat-advancement', 'value'),
    prevent_initial_call='initial_duplicate'
)
def run_guatemala_simulation(n_clicks, program_type, num_students, num_sims, num_years, apply_graduation_delay,
                     unemployment_rate, inflation_rate, isa_percentage, isa_threshold, isa_cap,
                     limit_years, price_per_student, performance_fee, annual_fee,
                     guat_placement, guat_advancement):
    
    if n_clicks > 0:
        # Set up parameters dictionary
        params = {
            'program_type': 'Guatemala',  # Force Guatemala since this is the Guatemala simulation callback
            'num_students': num_students,
            'num_sims': num_sims,
            'num_years': num_years,
            'apply_graduation_delay': apply_graduation_delay,
            'initial_unemployment_rate': unemployment_rate,
            'initial_inflation_rate': inflation_rate,
            'isa_percentage': isa_percentage,
            'isa_threshold': isa_threshold,
            'limit_years': limit_years,
            'performance_fee_pct': performance_fee,
            'annual_fee_per_student': annual_fee,
        }
        
        # Set default values for cap and price if they're not provided
        if isa_cap is None:
            params['isa_cap'] = 22500  # Default for Guatemala
        else:
            params['isa_cap'] = isa_cap
            
        if price_per_student is None:
            params['price_per_student'] = 7500  # Default for Guatemala
        else:
            params['price_per_student'] = price_per_student
        
        # Add Guatemala-specific parameters
        params.update({
            'guat_placement_prob': guat_placement,
            'guat_advancement_prob': guat_advancement,
            # Add defaults for Ecuador parameters
            'ecu_year1_completion_prob': 0.90,
            'ecu_placement_prob': 0.80,
            'ecu_na_completion_prob': 0.85,
        })
        
        # Set a random seed for reproducibility
        params['random_seed'] = 42
        
        try:
            # Run the simulation
            results = run_simple_simulation(**params)
            
            # Format the results for storage in dcc.Store
            formatted_results = {}
            for key, value in results.items():
                if isinstance(value, pd.Series):
                    formatted_results[key] = value.to_dict()
                elif isinstance(value, pd.DataFrame):
                    formatted_results[key] = value.to_dict('list')
                else:
                    formatted_results[key] = value
            
            return formatted_results, html.Div("Guatemala simulation completed successfully!", style={'color': 'green'})
        
        except Exception as e:
            return None, html.Div(f"Error: {str(e)}", style={'color': 'red'})
    
    return None, html.Div()

# Callbacks to update visualizations based on simulation results

@callback(
    Output('financial-metrics', 'children'),
    Input('simulation-results', 'data')
)
def update_financial_metrics(results):
    if not results:
        return html.Div("Run a simulation to see financial metrics.")
    
    # Create a summary of key financial metrics
    metrics = html.Div([
        html.H4("Financial Summary"),
        
        html.Div([
            html.Div([
                html.H5("Real (Inflation-Adjusted) Returns"),
                html.P([
                    html.Strong("Total Investment: "), 
                    f"${results['total_investment']:,.2f}"
                ]),
                html.P([
                    html.Strong("Avg Total Repayment: "), 
                    f"${results['average_total_payment']:,.2f}"
                ]),
                html.P([
                    html.Strong("Avg Investor Repayment: "), 
                    f"${results['average_investor_payment']:,.2f}"
                ]),
                html.P([
                    html.Strong("Avg Malengo Revenue: "), 
                    f"${results['average_malengo_payment']:,.2f}"
                ]),
                html.P([
                    html.Strong("Total IRR: "), 
                    f"{results['IRR']*100:.2f}%"
                ]),
                html.P([
                    html.Strong("Investor IRR: "), 
                    f"{results['investor_IRR']*100:.2f}%"
                ]),
            ], style={'width': '50%'}),
            
            html.Div([
                html.H5("Nominal Returns"),
                html.P([
                    html.Strong("Avg Total Repayment: "), 
                    f"${results['average_nominal_total_payment']:,.2f}"
                ]),
                html.P([
                    html.Strong("Avg Investor Repayment: "), 
                    f"${results['average_nominal_investor_payment']:,.2f}"
                ]),
                html.P([
                    html.Strong("Avg Malengo Revenue: "), 
                    f"${results['average_nominal_malengo_payment']:,.2f}"
                ]),
                html.P([
                    html.Strong("Total IRR: "), 
                    f"{results['nominal_IRR']*100:.2f}%"
                ]),
                html.P([
                    html.Strong("Investor IRR: "), 
                    f"{results['nominal_investor_IRR']*100:.2f}%"
                ]),
            ], style={'width': '50%'})
        ], style={'display': 'flex'})
    ])
    
    return metrics

@callback(
    Output('annual-payments-graph', 'figure'),
    Input('simulation-results', 'data')
)
def update_annual_payments_graph(results):
    if not results:
        return go.Figure()
    
    # Create annual payments graph
    fig = go.Figure()
    
    # Extract data
    years = list(range(1, len(results['payment_by_year']) + 1))
    
    # Add real payment traces
    fig.add_trace(go.Scatter(
        x=years, 
        y=list(results['investor_payment_by_year'].values()),
        mode='lines',
        name='Investor (Real)',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=years, 
        y=list(results['malengo_payment_by_year'].values()),
        mode='lines',
        name='Malengo (Real)',
        line=dict(color='green')
    ))
    
    # Add nominal payment traces
    fig.add_trace(go.Scatter(
        x=years, 
        y=list(results['nominal_investor_payment_by_year'].values()),
        mode='lines',
        name='Investor (Nominal)',
        line=dict(color='blue', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=years, 
        y=list(results['nominal_malengo_payment_by_year'].values()),
        mode='lines',
        name='Malengo (Nominal)',
        line=dict(color='green', dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='Annual Payments',
        xaxis_title='Year',
        yaxis_title='Average Payment per Student Cohort ($)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template='plotly_white'
    )
    
    return fig

@callback(
    Output('cumulative-returns-graph', 'figure'),
    Input('simulation-results', 'data')
)
def update_cumulative_returns_graph(results):
    if not results:
        return go.Figure()
    
    # Create cumulative returns graph
    fig = go.Figure()
    
    # Extract data
    years = list(range(1, len(results['payment_by_year']) + 1))
    
    # Calculate cumulative returns
    real_investor_cumulative = np.cumsum(list(results['investor_payment_by_year'].values()))
    nominal_investor_cumulative = np.cumsum(list(results['nominal_investor_payment_by_year'].values()))
    
    # Add cumulative return traces
    fig.add_trace(go.Scatter(
        x=years, 
        y=real_investor_cumulative,
        mode='lines',
        name='Investor Returns (Real)',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=years, 
        y=nominal_investor_cumulative,
        mode='lines',
        name='Investor Returns (Nominal)',
        line=dict(color='blue', dash='dash')
    ))
    
    # Add investment line
    fig.add_trace(go.Scatter(
        x=[1, len(years)],
        y=[results['total_investment'], results['total_investment']],
        mode='lines',
        name='Total Investment',
        line=dict(color='red', dash='dot')
    ))
    
    # Calculate and add breakeven points
    try:
        real_breakeven_year = next(i + 1 for i, val in enumerate(real_investor_cumulative) 
                                 if val >= results['total_investment'])
        fig.add_trace(go.Scatter(
            x=[real_breakeven_year],
            y=[results['total_investment']],
            mode='markers',
            name=f'Breakeven (Real): Year {real_breakeven_year}',
            marker=dict(color='red', size=12, symbol='star')
        ))
    except (StopIteration, TypeError):
        pass
    
    try:
        nominal_breakeven_year = next(i + 1 for i, val in enumerate(nominal_investor_cumulative) 
                                    if val >= results['total_investment'])
        fig.add_trace(go.Scatter(
            x=[nominal_breakeven_year],
            y=[results['total_investment']],
            mode='markers',
            name=f'Breakeven (Nominal): Year {nominal_breakeven_year}',
            marker=dict(color='green', size=12, symbol='star')
        ))
    except (StopIteration, TypeError):
        pass
    
    # Update layout
    fig.update_layout(
        title='Cumulative Investor Returns',
        xaxis_title='Year',
        yaxis_title='Cumulative Returns per Student Cohort ($)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template='plotly_white'
    )
    
    return fig

@callback(
    Output('student-metrics', 'children'),
    Input('simulation-results', 'data')
)
def update_student_metrics(results):
    if not results:
        return html.Div("Run a simulation to see student metrics.")
    
    # Create a summary of key student metrics
    metrics = html.Div([
        html.H4("Student Outcome Summary"),
        
        html.Div([
            html.Div([
                html.H5("Employment & Repayment"),
                html.P([
                    html.Strong("Employment Rate: "), 
                    f"{results['employment_rate']*100:.2f}%"
                ]),
                html.P([
                    html.Strong("Ever Employed Rate: "), 
                    f"{results['ever_employed_rate']*100:.2f}%"
                ]),
                html.P([
                    html.Strong("Repayment Rate: "), 
                    f"{results['repayment_rate']*100:.2f}%"
                ]),
            ], style={'width': '50%'}),
            
            html.Div([
                html.H5("Student Cap Statistics"),
                html.P([
                    html.Strong("Hit Payment Cap: "), 
                    f"{results['cap_stats']['payment_cap_pct']*100:.2f}%"
                ]),
                html.P([
                    html.Strong("Hit Years Cap: "), 
                    f"{results['cap_stats']['years_cap_pct']*100:.2f}%"
                ]),
                html.P([
                    html.Strong("Paid but No Cap: "), 
                    f"{results['cap_stats']['no_cap_paid_pct']*100:.2f}%"
                ]),
                html.P([
                    html.Strong("Never Paid: "), 
                    f"{results['cap_stats'].get('never_paid_pct', 0)*100:.2f}%"
                ]),
            ], style={'width': '50%'})
        ], style={'display': 'flex'})
    ])
    
    return metrics

@callback(
    Output('active-students-graph', 'figure'),
    Input('simulation-results', 'data')
)
def update_active_students_graph(results):
    if not results:
        return go.Figure()
    
    # Create active students graph
    fig = go.Figure()
    
    # Extract data
    years = list(range(1, len(results['active_students_by_year']) + 1))
    
    # Add active students trace
    fig.add_trace(go.Scatter(
        x=years, 
        y=list(results['active_students_by_year'].values()),
        mode='lines+markers',
        name='Active Students',
        line=dict(color='purple')
    ))
    
    # Update layout
    fig.update_layout(
        title='Active Students Over Time',
        xaxis_title='Year',
        yaxis_title='Average Number of Active Students',
        template='plotly_white'
    )
    
    return fig

@callback(
    Output('cap-stats-graph', 'figure'),
    Input('simulation-results', 'data')
)
def update_cap_stats_graph(results):
    if not results:
        return go.Figure()
    
    # Extract cap statistics
    cap_stats = results['cap_stats']
    
    # Create pie chart for cap statistics
    labels = ['Hit Payment Cap', 'Hit Years Cap', 'Paid but No Cap', 'Never Paid']
    values = [
        cap_stats['payment_cap_pct'],
        cap_stats['years_cap_pct'],
        cap_stats['no_cap_paid_pct'],
        cap_stats.get('never_paid_pct', 0)
    ]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        textinfo='label+percent',
        marker_colors=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    )])
    
    fig.update_layout(
        title='Student Payment Cap Statistics',
        template='plotly_white'
    )
    
    return fig

@callback(
    Output('pathway-metrics', 'children'),
    Output('pathway-sankey-graph', 'figure'),
    Input('simulation-results', 'data'),
    Input('program-type', 'value')
)
def update_pathway_analysis(results, program_type):
    if not results:
        return html.Div("Run a simulation to see pathway analysis."), go.Figure()
    
    # Create pathway metrics display
    metrics = html.Div([
        html.H4("Pathway Analysis"),
        
        html.Div([
            html.H5("Initial Degree Distribution"),
            html.Div([
                html.P([
                    html.Strong(f"{degree}: "), 
                    f"{pct*100:.1f}%"
                ]) for degree, pct in results['degree_pcts'].items()
            ]),
        ])
    ])
    
    # Create Sankey diagram based on program type
    if program_type == 'Ecuador':
        # Ecuador pathway probabilities
        y1_completion = results.get('ecu_year1_completion_prob', 0.9)
        placement = results.get('ecu_placement_prob', 0.8)
        na_completion = results.get('ecu_na_completion_prob', 0.85)
        
        # Calculate derived probabilities
        prob_stay_home = 1.0 - y1_completion
        prob_start_nha = y1_completion
        prob_placed_given_start_nha = placement
        prob_promote_given_placed = na_completion
        prob_promote_overall = prob_start_nha * prob_placed_given_start_nha * prob_promote_given_placed
        prob_placed_no_promote = prob_start_nha * prob_placed_given_start_nha * (1.0 - prob_promote_given_placed)
        prob_not_placed = prob_start_nha * (1.0 - prob_placed_given_start_nha)
        
        # Define nodes and links
        nodes = [
            {"name": "Start"},
            {"name": "Complete Year 1"},
            {"name": "STAY_HOME (Dropout)"},
            {"name": "Placed"},
            {"name": "Not Placed"},
            {"name": "Promoted to NA"},
            {"name": "Remain NHA"}
        ]
        
        links = [
            {"source": 0, "target": 1, "value": y1_completion * 100, "label": f"{y1_completion*100:.1f}%"},
            {"source": 0, "target": 2, "value": prob_stay_home * 100, "label": f"{prob_stay_home*100:.1f}%"},
            {"source": 1, "target": 3, "value": prob_start_nha * placement * 100, "label": f"{placement*100:.1f}%"},
            {"source": 1, "target": 4, "value": prob_start_nha * (1-placement) * 100, "label": f"{(1-placement)*100:.1f}%"},
            {"source": 3, "target": 5, "value": prob_placed_given_start_nha * prob_promote_given_placed * 100, "label": f"{prob_promote_given_placed*100:.1f}%"},
            {"source": 3, "target": 6, "value": prob_placed_given_start_nha * (1-prob_promote_given_placed) * 100, "label": f"{(1-prob_promote_given_placed)*100:.1f}%"}
        ]
        
    else:  # Guatemala
        # Guatemala pathway probabilities
        placement = results.get('guat_placement_prob', 0.85)
        advancement = results.get('guat_advancement_prob', 0.4)
        
        # Calculate derived probabilities
        prob_stay_home = 1.0 - placement
        prob_start_entry = placement
        prob_promote_given_placed = advancement
        prob_promote_overall = prob_start_entry * prob_promote_given_placed
        prob_entry_no_promote = prob_start_entry * (1.0 - prob_promote_given_placed)
        
        # Define nodes and links
        nodes = [
            {"name": "Start"},
            {"name": "Placed (HOSP_ENTRY)"},
            {"name": "STAY_HOME (Not Placed)"},
            {"name": "Promoted to HOSP_ADV"},
            {"name": "Remain HOSP_ENTRY"}
        ]
        
        links = [
            {"source": 0, "target": 1, "value": placement * 100, "label": f"{placement*100:.1f}%"},
            {"source": 0, "target": 2, "value": prob_stay_home * 100, "label": f"{prob_stay_home*100:.1f}%"},
            {"source": 1, "target": 3, "value": prob_promote_given_placed * 100, "label": f"{prob_promote_given_placed*100:.1f}%"},
            {"source": 1, "target": 4, "value": (1-prob_promote_given_placed) * 100, "label": f"{(1-prob_promote_given_placed)*100:.1f}%"}
        ]
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = [node["name"] for node in nodes],
          color = "blue"
        ),
        link = dict(
          source = [link["source"] for link in links],
          target = [link["target"] for link in links],
          value = [link["value"] for link in links],
          label = [link.get("label", "") for link in links]
        )
    )])
    
    fig.update_layout(
        title=f"{program_type} Program Pathway Analysis",
        font=dict(size=10),
        template='plotly_white'
    )
    
    return metrics, fig

# Initialize the default values for ISA cap and price per student on page load
@app.callback(
    Output('isa-cap', 'value', allow_duplicate=True),
    Output('price-per-student', 'value', allow_duplicate=True),
    Input('program-type', 'value'),
    prevent_initial_call='initial_duplicate'
)
def initialize_values(program_type):
    if program_type == 'Ecuador':
        return 27000, 9000
    else:  # Guatemala
        return 22500, 7500

# Run the app
if __name__ == '__main__':
    app.run(debug=True) 