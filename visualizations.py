import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def create_risk_gauge(risk_score):
    """Create beautiful risk gauge"""
    if risk_score < 30:
        color = "#10b981"
    elif risk_score < 70:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "CVD Risk Score", 'font': {'size': 28, 'color': '#1e40af'}},
        number={'font': {'size': 56, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#1e40af"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 30], 'color': '#d1fae5'},
                {'range': [30, 70], 'color': '#fef3c7'},
                {'range': [70, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 5},
                'thickness': 0.8,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Arial, sans-serif"}
    )
    
    return fig

def create_trend_chart(df):
    """Create risk trend over time"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['assessment_date'],
        y=df['risk_score'],
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#2563eb', width=3),
        marker=dict(size=10, color='#2563eb', line=dict(width=2, color='white')),
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.1)'
    ))
    
    # Add risk zones
    fig.add_hrect(y0=0, y1=30, fillcolor="#10b981", opacity=0.1, line_width=0)
    fig.add_hrect(y0=30, y1=70, fillcolor="#f59e0b", opacity=0.1, line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor="#ef4444", opacity=0.1, line_width=0)
    
    fig.update_layout(
        title="Risk Score Trend",
        xaxis_title="Date",
        yaxis_title="Risk Score",
        height=400,
        hovermode='x unified',
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'family': "Arial, sans-serif"},
        showlegend=False
    )
    
    return fig

def create_biomarker_comparison(current, previous):
    """Compare current vs previous biomarkers"""
    biomarkers = ['Total Chol', 'LDL', 'HDL', 'Triglycerides', 'Glucose', 'BMI', 'BP Systolic']
    current_vals = [
        current['cholesterol_total'], current['cholesterol_ldl'],
        current['cholesterol_hdl'], current['triglycerides'],
        current['glucose'], current['bmi'], current['blood_pressure_systolic']
    ]
    
    if previous is not None:
        previous_vals = [
            previous['cholesterol_total'], previous['cholesterol_ldl'],
            previous['cholesterol_hdl'], previous['triglycerides'],
            previous['glucose'], previous['bmi'], previous['blood_pressure_systolic']
        ]
    else:
        previous_vals = [0] * len(biomarkers)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Previous',
        x=biomarkers,
        y=previous_vals,
        marker_color='#cbd5e1'
    ))
    
    fig.add_trace(go.Bar(
        name='Current',
        x=biomarkers,
        y=current_vals,
        marker_color='#2563eb'
    ))
    
    fig.update_layout(
        title="Biomarker Comparison",
        barmode='group',
        height=400,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'family': "Arial, sans-serif"}
    )
    
    return fig

def create_population_distribution(df):
    """Create population risk distribution"""
    fig = px.pie(
        df,
        names='risk_category',
        values='count',
        title='Patient Risk Distribution',
        color='risk_category',
        color_discrete_map={
            'Low Risk': '#10b981',
            'Moderate Risk': '#f59e0b',
            'High Risk': '#ef4444'
        },
        hole=0.4
    )
    
    fig.update_layout(
        height=400,
        paper_bgcolor="white",
        font={'family': "Arial, sans-serif"}
    )
    
    return fig

def create_risk_factor_radar(data):
    """Create radar chart for risk factors"""
    categories = ['Age', 'Cholesterol', 'Blood Pressure', 'Glucose', 'BMI', 'Lifestyle']
    
    # Normalize values to 0-100 scale
    age_score = min((data['age'] - 30) * 2, 100)
    chol_score = min((data['cholesterol_total'] - 150) / 2, 100)
    bp_score = min((data['blood_pressure_systolic'] - 100) / 1.5, 100)
    glucose_score = min((data['glucose'] - 70) / 2, 100)
    bmi_score = min((data['bmi'] - 18.5) * 3, 100)
    lifestyle_score = (data['smoking'] * 50) + ((2 - data['physical_activity']) * 25)
    
    values = [age_score, chol_score, bp_score, glucose_score, bmi_score, lifestyle_score]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Risk Factors',
        line_color='#2563eb',
        fillcolor='rgba(37, 99, 235, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        height=400,
        paper_bgcolor="white",
        title="Risk Factor Profile"
    )
    
    return fig

def create_risk_timeline_projection(current_risk, age, with_changes=False):
    """Create future risk projection timeline"""
    # Calculate future risk based on age progression and lifestyle
    years = [0, 1, 3, 5, 10]
    ages = [age + y for y in years]
    
    # Without changes: risk increases with age (CVD risk increases ~1-2% per year after 40)
    if age >= 40:
        risk_increase_rate = 1.5  # 1.5 points per year
    else:
        risk_increase_rate = 0.8  # slower increase for younger people
    
    baseline_risks = [
        current_risk,
        min(100, current_risk + (risk_increase_rate * 1)),
        min(100, current_risk + (risk_increase_rate * 3)),
        min(100, current_risk + (risk_increase_rate * 5)),
        min(100, current_risk + (risk_increase_rate * 10))
    ]
    
    # With changes: risk decreases initially, then stabilizes
    improved_risks = [
        current_risk,
        max(0, current_risk - 15),  # Significant drop in year 1
        max(0, current_risk - 25),  # More improvement by year 3
        max(0, current_risk - 30),  # Peak improvement at year 5
        max(0, current_risk - 28)   # Slight increase after 10 years (aging)
    ]
    
    fig = go.Figure()
    
    # Without lifestyle changes line
    fig.add_trace(go.Scatter(
        x=years,
        y=baseline_risks,
        mode='lines+markers',
        name='Without Changes',
        line=dict(color='#ef4444', width=3, dash='dash'),
        marker=dict(size=10, color='#ef4444', line=dict(width=2, color='white')),
        hovertemplate='Year %{x}<br>Age: ' + str(age) + '+%{x}<br>Risk: %{y:.0f}<extra></extra>'
    ))
    
    # With lifestyle changes line
    fig.add_trace(go.Scatter(
        x=years,
        y=improved_risks,
        mode='lines+markers',
        name='With Lifestyle Changes',
        line=dict(color='#10b981', width=3),
        marker=dict(size=10, color='#10b981', line=dict(width=2, color='white')),
        hovertemplate='Year %{x}<br>Age: ' + str(age) + '+%{x}<br>Risk: %{y:.0f}<extra></extra>'
    ))
    
    # Add risk zones
    fig.add_hrect(y0=0, y1=30, fillcolor="#10b981", opacity=0.1, line_width=0)
    fig.add_hrect(y0=30, y1=70, fillcolor="#f59e0b", opacity=0.1, line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor="#ef4444", opacity=0.1, line_width=0)
    
    # Add annotations
    fig.add_annotation(
        x=10, y=baseline_risks[-1],
        text=f"Risk at age {ages[-1]}<br>without changes:<br><b>{baseline_risks[-1]:.0f}</b>",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#ef4444",
        bgcolor="white",
        bordercolor="#ef4444",
        borderwidth=2,
        font=dict(size=12, color="#ef4444")
    )
    
    fig.add_annotation(
        x=10, y=improved_risks[-1],
        text=f"Risk at age {ages[-1]}<br>with changes:<br><b>{improved_risks[-1]:.0f}</b>",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#10b981",
        bgcolor="white",
        bordercolor="#10b981",
        borderwidth=2,
        font=dict(size=12, color="#10b981")
    )
    
    fig.update_layout(
        title="Your Cardiovascular Risk Projection",
        xaxis_title="Years from Now",
        yaxis_title="Risk Score",
        height=450,
        hovermode='x unified',
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'family': "Arial, sans-serif"},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(range=[0, 105])
    )
    
    return fig

def create_feature_importance_chart(feature_contributions):
    """Create feature importance bar chart showing what drives risk"""
    features = list(feature_contributions.keys())
    importances = list(feature_contributions.values())
    
    # Sort by importance
    sorted_pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    features = [p[0] for p in sorted_pairs]
    importances = [p[1] for p in sorted_pairs]
    
    # Color bars by contribution level
    colors = ['#ef4444' if i > 25 else '#f59e0b' if i > 15 else '#2563eb' for i in importances]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=importances,
        y=features,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=[f'{i:.1f}%' for i in importances],
        textposition='outside',
        hovertemplate='%{y}<br>Contribution: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="What's Driving Your Risk?",
        xaxis_title="Contribution to Overall Risk (%)",
        yaxis_title="",
        height=400,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'family': "Arial, sans-serif"},
        xaxis=dict(range=[0, max(importances) * 1.2]),
        showlegend=False
    )
    
    return fig

