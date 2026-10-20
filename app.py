from __future__ import annotations
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from data_loader import fetch_nflverse, parse_csv
from engine import (automatic_comparison, game_coverage, player_involvement,
                    prepare_plays, summary, DataQualityError)

st.set_page_config(page_title='NFL Strategy Lab | SF × WAS', page_icon='🏈', layout='wide')
st.markdown('''<style>
.block-container {padding-top:1.6rem; max-width:1170px}
[data-testid="stMetric"] {background:#f2f5f9;border-radius:12px;padding:14px}
h1,h2,h3 {letter-spacing:-.025em}
</style>''', unsafe_allow_html=True)

st.title('🏈 NFL Strategy Lab')
st.caption('SF 49ers × Washington Commanders · Pregame educational analysis · V1')
st.info('This project explores historical play selection and situational efficiency. The resampling is **not** a calibrated game, player-stat, or betting prediction. Live-game ingestion and roster checks are not included yet.')

with st.sidebar:
    st.header('Data & scenario')
    st.markdown('**Pregame snapshot cutoff:** October 19, 2026, 12:00 AM (exclusive).')
    st.caption('This prevents tonight’s result from leaking into the analysis even when data updates later.')
    data_method = st.radio('Get data', ['Fetch public NFL play-by-play', 'Upload nflverse CSV file(s)'], horizontal=False)
    uploaded = None
    if data_method.startswith('Upload'):
        uploaded = st.file_uploader('Select one or more NFLverse play-by-play CSVs', type='csv', accept_multiple_files=True)
    offense = st.selectbox('Offensive team', ['SF', 'WAS'], format_func=lambda t: 'San Francisco 49ers' if t=='SF' else 'Washington Commanders')
    defense = 'WAS' if offense=='SF' else 'SF'
    st.markdown('**Game situation**')
    down = st.select_slider('Down', options=[1,2,3,4], value=2)
    to_go = st.slider('Yards to first down', 1, 20, 6)
    yardline = st.slider('Yards from opponent end zone', 5, 95, 65, 5)
    n = st.selectbox('Historical resamples per play type', [250,500,1000,2000], index=1)
    st.caption('The simulator chooses and compares both available play categories automatically. You do not need to select a play.')

@st.cache_data(ttl=6*3600, show_spinner='Downloading nflverse play-by-play (may take a minute)…')
def cached_fetch():
    return fetch_nflverse((2025,2026))

raw = None
try:
    if data_method.startswith('Fetch'):
        raw = cached_fetch()
    elif uploaded:
        raw = pd.concat([parse_csv(io.BytesIO(f.getvalue())) for f in uploaded], ignore_index=True)
except Exception as e:
    st.error('The public download was not available in this session. You can upload nflverse play-by-play CSVs instead.')
    with st.expander('Technical detail'):
        st.code(str(e))

if raw is None:
    st.warning('No play-by-play data loaded yet. To enable this live dashboard, use **Fetch** when internet is available or upload 2025 and 2026 play-by-play CSV files. The app never substitutes invented results.')
    st.markdown('**Source:** https://github.com/nflverse/nflverse-data/releases/tag/pbp')
    st.stop()

try:
    df = prepare_plays(raw, cutoff='2026-10-19', seasons=(2025,2026))
except DataQualityError as err:
    st.error(f'Data quality validation failed: {err}')
    st.stop()

regular_2026 = df[df['season']==2026]
if not ((regular_2026['posteam']=='SF').any() and (regular_2026['posteam']=='WAS').any()):
    st.error('The data does not include 2026 offensive snaps for both teams. Please upload the correct files.')
    st.stop()

st.markdown(f'**Analysis:** {offense} offense vs. {defense} defense · Down {down} · {to_go} yards to gain · {yardline} yards from goal')
page1, page2, page3, page4 = st.tabs(['Matchup overview','Automatic play comparison','Player involvement','Data validation'])

with page1:
    st.subheader('Actual 2026 play tendencies')
    st.dataframe(summary(df, offense, defense), hide_index=True, use_container_width=True)
    st.caption('Metrics use observed 2026 regular-season snaps only. Dropbacks include sacks, while runs may include scrambles. Positive EPA is a descriptive play-success definition.')
    sf = df[(df['posteam'] == offense) & (df['season'] == 2026)]
    if not sf.empty:
        top = sf.groupby(['game_date','play_type']).agg(
            plays=('epa','count'), epa_per_play=('epa','mean'),
            yards_per_play=('yards_gained','mean')).reset_index()
        fig = px.line(top, x='game_date', y='epa_per_play', color='play_type',
                      markers=True, title=f'{offense} EPA per play across 2026 game dates')
        fig.update_layout(xaxis_title='Game date', yaxis_title='EPA per play', legend_title='Play type')
        st.plotly_chart(fig, use_container_width=True)

with page2:
    st.subheader('What happened in similar historical situations?')
    st.caption('The model compares historical dropback/pass and run/scramble outcomes automatically. Recent plays involving these teams receive extra weight. The findings are observational, not causal recommendations.')
    try:
        comparison, scenarios = automatic_comparison(
            df, offense, defense, down=down, to_go=to_go,
            yardline=yardline, n=n, seed=20261019)
    except DataQualityError as exc:
        st.error(str(exc))
    else:
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        best = comparison.loc[comparison['Avg EPA/play in resample'].idxmax(), 'Play type']
        st.info(f'**Higher historical EPA for this situation:** {best}. Actual choices also depend on coverage, player availability, and execution, which this version does not model.')
        melted = pd.concat([pd.DataFrame({'Play type': r.category,
                                           'Yards gained': r.sampled_yards})
                            for r in scenarios], ignore_index=True)
        fig = px.histogram(melted, x='Yards gained', color='Play type', nbins=45,
                           barmode='overlay', opacity=.60,
                           title=f'Yardage in {n:,} historical resamples per category')
        fig.update_layout(xaxis_title='Yards gained or lost', yaxis_title='Resampled frequency')
        st.plotly_chart(fig, use_container_width=True)
        st.caption('The middle 80% yardage range describes the resampled observations; it is not a validated prediction interval. More resamples reduce Monte Carlo noise, not bias.')

with page3:
    st.subheader(f'{offense} — who has been involved?')
    st.caption('Recorded 2026 targets and rushing opportunities. These are historical observations, not forecasts; current injuries and active rosters are not yet checked.')
    player = player_involvement(df, offense, season=2026)
    if player.empty:
        st.warning('No tagged player observations in the supplied source.')
    else:
        role = st.selectbox('Opportunity type', sorted(player['Role'].unique()))
        p = player.loc[player['Role'] == role].head(15)
        st.dataframe(p, hide_index=True, use_container_width=True)
        fig = px.bar(p.head(10), x='Player', y='Plays', title='Recorded opportunities, not projections')
        st.plotly_chart(fig, use_container_width=True)

with page4:
    st.subheader('Data lineage & guardrails')
    st.dataframe(game_coverage(df, '2026-10-19'), hide_index=True, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric('Most recent game in dataset', str(df['game_date'].max().date()))
    c2.metric('Historical plays included', f'{len(df):,}')
    c3.metric('Current player availability', 'Not checked')
    st.write('**Primary source:** [nflverse NFL play-by-play](https://github.com/nflverse/nflverse-data/releases/tag/pbp). Input is limited to regular-season plays before October 19, 2026.')
    st.write('**Next development stage:** out-of-sample evaluation, roster checks, refined play types, and then a validated live-play feed.')
