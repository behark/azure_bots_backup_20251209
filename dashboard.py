#!/usr/bin/env python3
"""
Real-time Trading Bot Performance Dashboard
Launch with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime
import subprocess

# Page config
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .profit {
        color: #00ff00;
    }
    .loss {
        color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 Trading Bot Performance Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")

    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

    # Run analysis button
    if st.button("📊 Run Full Analysis", use_container_width=True):
        with st.spinner("Running analysis..."):
            result = subprocess.run(
                ["python3", "analyze_bot_performance_detailed.py"],
                capture_output=True,
                text=True
            )
            st.success("Analysis complete!")

    st.markdown("---")

    # Filters
    st.subheader("Filters")
    min_signals = st.slider("Min Signals", 0, 100, 5)
    show_losing_only = st.checkbox("Show Losing Bots Only")

    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# Load latest report
@st.cache_data(ttl=60)
def load_latest_report():
    reports = sorted(Path('.').glob('bot_performance_report_*.json'), reverse=True)
    if not reports:
        return None

    with open(reports[0]) as f:
        return json.load(f)

data = load_latest_report()

if not data:
    st.error("No performance reports found. Please run the analysis first.")
    st.code("python3 analyze_bot_performance_detailed.py")
    st.stop()

# Prepare data
bot_data = []
for name, bot in data.items():
    if bot.get('status') == 'NO_DATA':
        continue

    metrics = bot.get('overall_metrics', {})
    if metrics.get('total_signals', 0) < min_signals:
        continue

    if show_losing_only and metrics.get('total_pnl', 0) >= 0:
        continue

    bot_data.append({
        'Bot': name.replace('_bot', ''),
        'Status': bot.get('status', 'UNKNOWN'),
        'Signals': metrics.get('total_signals', 0),
        'Win Rate (%)': metrics.get('win_rate', 0),
        'Total PnL (%)': metrics.get('total_pnl', 0),
        'Avg PnL (%)': metrics.get('avg_pnl', 0),
        'Best Trade (%)': metrics.get('best_trade', 0),
        'Worst Trade (%)': metrics.get('worst_trade', 0),
        'TP1': metrics.get('tp1_hits', 0),
        'TP2': metrics.get('tp2_hits', 0),
        'TP3': metrics.get('tp3_hits', 0),
        'SL': metrics.get('sl_hits', 0),
        'Open': bot.get('open_positions', 0),
    })

bot_df = pd.DataFrame(bot_data)

if bot_df.empty:
    st.warning("No bot data matches the current filters.")
    st.stop()

# Overall Metrics
st.header("📈 Overall Performance")

col1, col2, col3, col4, col5 = st.columns(5)

total_pnl = bot_df['Total PnL (%)'].sum()
avg_win_rate = bot_df['Win Rate (%)'].mean()
total_signals = bot_df['Signals'].sum()
profitable_bots = len(bot_df[bot_df['Total PnL (%)'] > 0])
total_open = bot_df['Open'].sum()

col1.metric("Total PnL", f"{total_pnl:.2f}%",
            delta=f"{total_pnl:.2f}%",
            delta_color="normal")
col2.metric("Avg Win Rate", f"{avg_win_rate:.1f}%")
col3.metric("Total Signals", f"{total_signals:,}")
col4.metric("Profitable Bots", f"{profitable_bots}/{len(bot_df)}")
col5.metric("Open Positions", total_open)

st.markdown("---")

# Main Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Bot PnL Ranking")

    fig_pnl = px.bar(
        bot_df.sort_values('Total PnL (%)', ascending=True),
        y='Bot',
        x='Total PnL (%)',
        color='Total PnL (%)',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        orientation='h',
        height=600
    )
    fig_pnl.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

with col2:
    st.subheader("🎯 Win Rate vs PnL")

    fig_scatter = px.scatter(
        bot_df,
        x='Win Rate (%)',
        y='Total PnL (%)',
        size='Signals',
        color='Total PnL (%)',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        hover_data=['Bot', 'Signals', 'Open'],
        height=600
    )

    # Add quadrant lines
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_scatter.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)

    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# TP/SL Analysis
st.subheader("🎯 Take Profit & Stop Loss Analysis")

col1, col2 = st.columns(2)

with col1:
    # TP/SL breakdown by bot
    tp_sl_data = []
    for _, row in bot_df.iterrows():
        tp_sl_data.extend([
            {'Bot': row['Bot'], 'Type': 'TP1', 'Count': row['TP1']},
            {'Bot': row['Bot'], 'Type': 'TP2', 'Count': row['TP2']},
            {'Bot': row['Bot'], 'Type': 'TP3', 'Count': row['TP3']},
            {'Bot': row['Bot'], 'Type': 'SL', 'Count': row['SL']},
        ])

    tp_sl_df = pd.DataFrame(tp_sl_data)

    fig_tpsl = px.bar(
        tp_sl_df,
        x='Bot',
        y='Count',
        color='Type',
        title='TP/SL Distribution by Bot',
        barmode='stack',
        color_discrete_map={
            'TP1': '#90EE90',
            'TP2': '#32CD32',
            'TP3': '#228B22',
            'SL': '#FF6B6B'
        }
    )
    st.plotly_chart(fig_tpsl, use_container_width=True)

with col2:
    # Overall TP/SL pie chart
    total_tp1 = bot_df['TP1'].sum()
    total_tp2 = bot_df['TP2'].sum()
    total_tp3 = bot_df['TP3'].sum()
    total_sl = bot_df['SL'].sum()

    fig_pie = go.Figure(data=[go.Pie(
        labels=['TP1', 'TP2', 'TP3', 'SL'],
        values=[total_tp1, total_tp2, total_tp3, total_sl],
        marker=dict(colors=['#90EE90', '#32CD32', '#228B22', '#FF6B6B']),
        hole=0.3
    )])
    fig_pie.update_layout(title='Overall Exit Distribution')
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# Detailed Table
st.subheader("📊 Detailed Bot Performance")

# Color-code the table
def color_pnl(val):
    if val > 0:
        return 'background-color: #d4edda'
    elif val < 0:
        return 'background-color: #f8d7da'
    return ''

styled_df = bot_df.style.applymap(
    color_pnl,
    subset=['Total PnL (%)', 'Avg PnL (%)']
).format({
    'Win Rate (%)': '{:.2f}',
    'Total PnL (%)': '{:.2f}',
    'Avg PnL (%)': '{:.2f}',
    'Best Trade (%)': '{:.2f}',
    'Worst Trade (%)': '{:.2f}',
})

st.dataframe(styled_df, use_container_width=True, height=400)

# Export options
st.markdown("---")
st.subheader("💾 Export Data")

col1, col2, col3 = st.columns(3)

with col1:
    csv = bot_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"bot_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

with col2:
    json_str = bot_df.to_json(orient='records', indent=2)
    st.download_button(
        label="📥 Download JSON",
        data=json_str,
        file_name=f"bot_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# Top Performers & Worst Performers
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏆 Top 5 Performers")
    top5 = bot_df.nlargest(5, 'Total PnL (%)')
    st.dataframe(
        top5[['Bot', 'Total PnL (%)', 'Win Rate (%)', 'Signals']],
        hide_index=True,
        use_container_width=True
    )

with col2:
    st.subheader("📉 Bottom 5 Performers")
    bottom5 = bot_df.nsmallest(5, 'Total PnL (%)')
    st.dataframe(
        bottom5[['Bot', 'Total PnL (%)', 'Win Rate (%)', 'Signals']],
        hide_index=True,
        use_container_width=True
    )

# Recommendations
st.markdown("---")
st.subheader("💡 Automated Recommendations")

recommendations = []

# Check for losing bots
for _, row in bot_df.iterrows():
    if row['Total PnL (%)'] < -10 and row['Signals'] >= 10:
        recommendations.append(f"❌ **DISABLE {row['Bot']}**: Loss of {row['Total PnL (%)']:.2f}% ({row['Signals']} signals)")
    elif row['Win Rate (%)'] < 35 and row['Signals'] >= 20:
        recommendations.append(f"⚠️ **REVIEW {row['Bot']}**: Low win rate {row['Win Rate (%)']:.1f}%")
    elif row['Total PnL (%)'] > 20 and row['Win Rate (%)'] > 55:
        recommendations.append(f"✅ **INCREASE ALLOCATION {row['Bot']}**: Strong performer ({row['Total PnL (%)']:.2f}% PnL, {row['Win Rate (%)']:.1f}% WR)")

if recommendations:
    for rec in recommendations:
        st.markdown(rec)
else:
    st.success("✅ All bots performing within acceptable ranges")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Auto-refresh enabled | Data cached for 60s")
