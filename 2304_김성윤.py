import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pyreadstat
from scipy.stats import chi2_contingency, fisher_exact
import base64
import os

st.set_page_config(layout="wide")

# CSS 스타일 정의
st.markdown("""
    <style>
    body { background-color: #f0f2f6; }
    .title { font-size: 2.5em; color: #4B0082; text-align: center; animation: colorChange 10s infinite; }
    .top-text { 
        font-size: 1.8em; 
        color: #1E90FF; 
        text-align: center; 
        animation: colorChange 12s infinite; 
        margin-top: -10px; 
    }
    .section-header { font-size: 2em; color: #2E8B57; margin-top: 40px; animation: colorChange 15s infinite; }
    .subheader { font-size: 1.5em; color: #FF4500; margin-top: 20px; animation: colorChange 12s infinite; }
    .dataframe { border: 2px solid #4B0082; }
    @keyframes colorChange {
        0% { color: #4B0082; }
        25% { color: #2E8B57; }
        50% { color: #FF4500; }
        75% { color: #1E90FF; }
        100% { color: #4B0082; }
    }
    .source-table {
        margin-left: auto;
        margin-right: auto;
        width: 80%;
        border-collapse: collapse;
    }
    .source-table th, .source-table td {
        border: 1px solid #dddddd;
        text-align: center;
        padding: 8px;
    }
    .source-table th {
        background-color: #f2f2f2;
    }
    </style>
    """, unsafe_allow_html=True)

# 페이지 제목 및 상단 텍스트
st.markdown('<div class="title">🌟 천식으로 인한 결석일수 분석 🌟</div>', unsafe_allow_html=True)
st.markdown('<div class="top-text">2024 Interactive Physics</div>', unsafe_allow_html=True)
st.markdown('<div class="top-text">2304 김성윤</div>', unsafe_allow_html=True)

# 데이터 로딩 및 전처리 함수 정의
@st.cache_data(show_spinner=False)
def load_data(file_path):
    df, meta = pyreadstat.read_sav(file_path)
    return df, meta

@st.cache_data(show_spinner=False)
def preprocess_data(df, remove_na):
    if remove_na:
        df = df.replace({9999: np.nan, 8888: np.nan})
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

@st.cache_data(show_spinner=False)
def compute_correlations(df, method='pearson'):
    return df.corr(method=method)

# 데이터 로딩
df, meta = load_data("kyrbs2023.sav")  # 실제 파일 경로로 변경하세요.
remove_na = st.sidebar.checkbox("무응답(9999) 및 비해당(8888) 값을 결측치로 처리", value=True)
df = preprocess_data(df, remove_na)
numeric_df = df.select_dtypes(include=['float32', 'int32', 'float64', 'int64'])
categorical_df = df.select_dtypes(include=['object', 'category'])
st.sidebar.header("상관계수 방법 선택")
correlation_method = st.sidebar.selectbox("상관계수 방법을 선택하세요", options=['pearson', 'spearman', 'kendall'], index=0)
correlations = compute_correlations(numeric_df, method=correlation_method)
variable_labels = meta.column_names_to_labels

# 변수 카테고리 매핑
categories_mapping = {
    '주관적 상태': ['PR'], '식생활': ['F'], '신체활동': ['PA'], '비만 및 체중 조절': ['HT', 'WT', 'WC'],
    '정신건강': ['M'], '구강건강': ['O'], '개인위생': ['HW'], '손상예방': ['I'],
    '폭력': ['V'], '음주': ['AC'], '흡연': ['TC'], '성행태': ['SEX', 'S'],
    '약물': ['DR'], '아토피천식': ['AS', 'RH', 'ECZ'], '인터넷중독': ['INT'],
    '일반적 특성': ['CD', 'AGE', 'E', 'COVID']
}

categories_numeric = {category: {} for category in categories_mapping.keys()}
categories_numeric['기타'] = {}
categories_categorical = {category: {} for category in categories_mapping.keys()}
categories_categorical['기타'] = {}
display_name_to_var_name = {}

for var_name in numeric_df.columns:
    var_label = variable_labels.get(var_name, var_name)
    var_prefix = var_name.split('_')[0]
    found_category = next((category for category, prefixes in categories_mapping.items() if var_prefix in prefixes), '기타')
    variable_group = var_prefix
    categories_numeric[found_category].setdefault(variable_group, []).append({
        'name': var_name, 'label': var_label, 'display_name': f"{var_label} [{var_name}]"
    })
    display_name_to_var_name[f"{var_label} [{var_name}]"] = var_name

for var_name in categorical_df.columns:
    var_label = variable_labels.get(var_name, var_name)
    var_prefix = var_name.split('_')[0]
    found_category = next((category for category, prefixes in categories_mapping.items() if var_prefix in prefixes), '기타')
    variable_group = var_prefix
    categories_categorical[found_category].setdefault(variable_group, []).append({
        'name': var_name, 'label': var_label, 'display_name': f"{var_label} [{var_name}]"
    })
    display_name_to_var_name[f"{var_label} [{var_name}]"] = var_name

# 산점도 섹션
st.sidebar.header("산점도")
scatter_categories = ["선택 안 함"] + list(categories_numeric.keys()) + list(categories_categorical.keys())
scatter_category = st.sidebar.selectbox("산점도에 사용할 카테고리 선택", scatter_categories, index=0, key="scatter_category")

if scatter_category != "선택 안 함":
    scatter_variable_groups = categories_numeric[scatter_category] if scatter_category in categories_numeric else categories_categorical[scatter_category]
    if not scatter_variable_groups:
        st.sidebar.write("해당 카테고리에 변수가 없습니다.")
        st.stop()
    scatter_variables = [var for group in scatter_variable_groups.values() for var in group]
    scatter_variable_options = [var['display_name'] for var in scatter_variables]
    scatter_variable_display = st.sidebar.selectbox("산점도 변수 선택", scatter_variable_options, key="scatter_variable_select")
    single_var = display_name_to_var_name[scatter_variable_display]
else:
    st.sidebar.write("카테고리를 선택하고 변수를 선택하세요.")
    single_var = None

threshold = st.sidebar.slider("상관 계수 임계값 설정", 0.0, 1.0, 0.5, 0.01)

if single_var:
    single_var_correlations = correlations[[single_var]].dropna()
    single_var_correlations = single_var_correlations[single_var_correlations.index != single_var]
    single_var_correlations = single_var_correlations.assign(abs_correlation=single_var_correlations[single_var].abs())
    single_var_correlations = single_var_correlations[single_var_correlations['abs_correlation'] >= threshold]
    single_var_correlations = single_var_correlations.sort_values('abs_correlation', ascending=False)
    st.markdown(f'<div class="subheader">### \'{scatter_variable_display}\' 변수와 상관 계수 {threshold} 이상인 변수:</div>', unsafe_allow_html=True)
    correlation_vars = single_var_correlations.index.tolist()
    correlation_labels = [f"{variable_labels.get(var, var)} [{var}]" for var in correlation_vars]
    single_var_correlations_display = single_var_correlations.copy()
    single_var_correlations_display.index = correlation_labels
    st.dataframe(single_var_correlations_display[[single_var]].round(3), use_container_width=True)
else:
    st.write("상관관계를 분석할 변수를 선택하세요.")

# 데이터 시각화 섹션
st.markdown('<div class="section-header">## 데이터 시각화</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">### 산점도를 그릴 변수 선택 옵션</div>', unsafe_allow_html=True)

if 'plot_all' not in st.session_state:
    st.session_state.plot_all = False

plot_all = st.checkbox("상관계수가 높은 변수뿐만 아니라 다른 변수도 선택", value=st.session_state.plot_all, key="plot_all_checkbox")
st.session_state.plot_all = plot_all

scatter_vars = []

if not plot_all:
    if single_var and not single_var_correlations_display.empty:
        st.markdown('<div class="subheader">#### 상관계수가 높은 변수 중에서 선택</div>', unsafe_allow_html=True)
        available_vars = single_var_correlations_display.index.tolist()
        scatter_categories_filtered = sorted(list(set([
            next((category for category, prefixes in categories_mapping.items() if var.split('_')[0] in prefixes), '기타')
            for var in [display_name_to_var_name[dn] for dn in available_vars]
        ])))
        scatter_selected_categories = st.multiselect("산점도에 사용할 카테고리 선택", scatter_categories_filtered, default=[])
        scatter_var_displays = []
        scatter_vars = []
        for scatter_category in scatter_selected_categories:
            scatter_vars_in_category = [
                var['display_name'] for group in (categories_numeric[scatter_category] if scatter_category in categories_numeric else categories_categorical[scatter_category]).values()
                for var in group if var['display_name'] in available_vars
            ]
            if scatter_vars_in_category:
                session_key = f'scatter_var_high_corr_{scatter_category}'
                if session_key not in st.session_state:
                    st.session_state[session_key] = [scatter_vars_in_category[0]] if scatter_vars_in_category else []
                selected_vars = st.multiselect(
                    f"{scatter_category}에서 산점도를 그릴 변수 선택",
                    options=scatter_vars_in_category,
                    default=st.session_state.get(session_key, [scatter_vars_in_category[0]] if scatter_vars_in_category else []),
                    key=session_key
                )
                scatter_var_displays.extend(selected_vars)
                scatter_vars.extend([display_name_to_var_name[dn] for dn in selected_vars])
            else:
                st.write(f"선택한 카테고리 '{scatter_category}'에 상관계수가 높은 변수가 없습니다.")
    else:
        st.write("상관 계수가 임계값 이상인 변수가 없습니다.")
else:
    st.markdown('<div class="subheader">#### 전체 변수 중에서 선택</div>', unsafe_allow_html=True)
    all_labels = [f"{variable_labels.get(var, var)} [{var}]" for var in numeric_df.columns if var != single_var] + \
                 [f"{variable_labels.get(var, var)} [{var}]" for var in categorical_df.columns if var != single_var]
    scatter_categories_all = sorted(list(set([
        next((category for category, prefixes in categories_mapping.items() if var.split('_')[0] in prefixes), '기타')
        for var in [display_name_to_var_name[dn] for dn in all_labels]
    ])))
    scatter_selected_categories_all = st.multiselect("산점도에 사용할 카테고리 선택", scatter_categories_all, default=[])
    scatter_var_displays = []
    scatter_vars = []
    for scatter_category in scatter_selected_categories_all:
        scatter_vars_in_category_all = [
            var['display_name'] for group in (categories_numeric[scatter_category] if scatter_category in categories_numeric else categories_categorical[scatter_category]).values()
            for var in group if var['display_name'] in all_labels
        ]
        if scatter_vars_in_category_all:
            session_key = f'scatter_var_all_{scatter_category}'
            if session_key not in st.session_state:
                st.session_state[session_key] = [scatter_vars_in_category_all[0]] if scatter_vars_in_category_all else []
            selected_vars = st.multiselect(
                f"{scatter_category}에서 산점도를 그릴 변수 선택",
                options=scatter_vars_in_category_all,
                default=st.session_state.get(session_key, [scatter_vars_in_category_all[0]] if scatter_vars_in_category_all else []),
                key=session_key
            )
            scatter_var_displays.extend(selected_vars)
            scatter_vars.extend([display_name_to_var_name[dn] for dn in selected_vars])
        else:
            st.write(f"선택한 카테고리 '{scatter_category}'에 변수가 없습니다.")

if scatter_vars and single_var:
    st.markdown('<div class="subheader">### 산점도 옵션 설정</div>', unsafe_allow_html=True)
    jitter_strength_x = st.slider("X축 지터 강도", 0.0, 1.0, 0.1, step=0.01, key="jitter_strength_x")
    jitter_strength_y = st.slider("Y축 지터 강도", 0.0, 1.0, 0.1, step=0.01, key="jitter_strength_y")
    alpha = st.slider("점의 투명도 설정 (alpha)", 0.0, 1.0, 0.7, step=0.01, key="alpha_slider")
    scatter_df = df[[single_var] + scatter_vars].dropna().copy()
    sample_frac = st.slider("데이터 샘플링 비율 (%)", 10, 100, 100, step=10, key="sample_frac")
    if sample_frac < 100:
        scatter_df = scatter_df.sample(frac=sample_frac / 100, random_state=42)
    scatter_df['scatter_y'] = scatter_df[single_var] + np.random.normal(0, jitter_strength_y, size=len(scatter_df))
    st.markdown('<div class="subheader">### 산점도</div>', unsafe_allow_html=True)
    num_cols = 2
    cols = st.columns(num_cols)
    for idx, (scatter_var_display, scatter_var) in enumerate(zip(scatter_var_displays, scatter_vars)):
        scatter_df['scatter_x'] = scatter_df[scatter_var] + np.random.normal(0, jitter_strength_x, size=len(scatter_df))
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scattergl(
            x=scatter_df['scatter_x'],
            y=scatter_df['scatter_y'],
            mode='markers',
            marker=dict(
                size=5,
                color=scatter_df[single_var],
                colorscale='Viridis',
                opacity=alpha,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            text=scatter_df.apply(lambda row: f"{scatter_var_display}: {row[scatter_var]}<br>{scatter_variable_display}: {row[single_var]}", axis=1),
            hoverinfo='text'
        ))
        fig_scatter.update_layout(
            title=f"{variable_labels.get(single_var, single_var)} vs {scatter_var_display} 산점도",
            xaxis_title=scatter_var_display,
            yaxis_title=variable_labels.get(single_var, single_var),
            hovermode='closest',
            dragmode='zoom',
            width=400,
            height=300
        )
        with cols[idx % num_cols]:
            st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.write("산점도를 그릴 변수를 선택하세요.")

# 히트맵 섹션
st.markdown('<div class="section-header">## 히트맵</div>', unsafe_allow_html=True)
st.sidebar.header("히트맵")
selected_vars_multi = []
heatmap_selected_categories = st.sidebar.multiselect("히트맵에 사용할 카테고리 선택", list(categories_numeric.keys()) + list(categories_categorical.keys()), default=[])

if heatmap_selected_categories:
    for category in heatmap_selected_categories:
        variable_groups = categories_numeric[category] if category in categories_numeric else categories_categorical[category]
        num_vars_in_category = sum(len(group) for group in variable_groups.values())
        with st.sidebar.expander(f"{category} ({num_vars_in_category}개 변수)"):
            for variable_group, vars_in_group in variable_groups.items():
                display_names = [var['display_name'] for var in vars_in_group]
                select_all_key = f"{category}_{variable_group}_select_all"
                multiselect_key = f"{category}_{variable_group}_multiselect"
                st.markdown(f"**{variable_group}** ({len(display_names)}개 변수)")
                select_all = st.checkbox(f"{variable_group} 전체 선택", key=select_all_key)
                default_selection = display_names if select_all else []
                selected = st.multiselect(f"{variable_group} 변수 선택", options=display_names, default=default_selection, key=multiselect_key)
                selected_vars_multi.extend(selected)
else:
    st.sidebar.write("카테고리를 선택하세요.")

if selected_vars_multi:
    selected_var_names_multi = [display_name_to_var_name[dn] for dn in selected_vars_multi]
    binned_vars = []
    for var in selected_var_names_multi:
        if df[var].dtype != 'object':
            binned_var = f"{var}_binned"
            df[binned_var] = pd.cut(df[var], bins=5)
            binned_vars.append(binned_var)
        else:
            binned_vars.append(var)
    final_vars = binned_vars
    final_labels = [f"{variable_labels.get(var.split('_binned')[0], var.split('_binned')[0])} [{var}]" for var in final_vars]
    selected_corr_multi = correlations.loc[selected_var_names_multi, selected_var_names_multi]
    selected_corr_multi.index = final_labels
    selected_corr_multi.columns = final_labels
    selected_corr_rounded_multi = selected_corr_multi.round(3)
    fig_heatmap = px.imshow(
        selected_corr_rounded_multi,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="선택한 변수들의 상관관계 히트맵",
        labels={"color": "상관계수"},
        width=800,
        height=800
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown('<div class="subheader">### 교차표 데이터 검정 (카이제곱 검정)</div>', unsafe_allow_html=True)
    if len(selected_var_names_multi) == 2:
        var1, var2 = selected_var_names_multi
        contingency_table = pd.crosstab(df[var1], df[var2])
        contingency_table.index = contingency_table.index.astype(str)
        contingency_table.columns = contingency_table.columns.astype(str)
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        st.write(f"**카이제곱 통계량:** {chi2:.4f}")
        st.write(f"**자유도:** {dof}")
        st.write(f"**p-값:** {p:.10f}")
        expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
        st.dataframe(expected_df, use_container_width=True)
        st.write("**결론:** 두 변수 간에 유의한 상관관계가 있습니다." if p < 0.05 else "두 변수 간에 유의한 상관관계가 없습니다.")
        if contingency_table.shape == (2, 2):
            oddsratio, fisher_p = fisher_exact(contingency_table)
            st.write(f"**Fisher's Exact Test p-값:** {fisher_p:.10f}")
            st.write("**Fisher's Exact Test 결론:** 두 변수 간에 유의한 상관관계가 있습니다." if fisher_p < 0.05 else "두 변수 간에 유의한 상관관계가 없습니다.")
    else:
        st.write("카이제곱 검정을 위해 두 개의 변수를 선택하세요.")
else:
    st.write("히트맵을 생성하려면 변수를 선택하세요.")

# 막대그래프 섹션
st.markdown('<div class="section-header">## 막대그래프 시각화</div>', unsafe_allow_html=True)
st.sidebar.header("막대그래프")
st.sidebar.subheader("막대그래프의 x축 변수 선택")
bar_x_categories = ["선택 안 함"] + list(categories_numeric.keys()) + list(categories_categorical.keys())
bar_x_category = st.sidebar.selectbox("막대그래프의 x축 변수의 카테고리 선택", bar_x_categories, index=0, key="bar_x_category")

if bar_x_category != "선택 안 함":
    bar_x_variable_groups = categories_numeric[bar_x_category] if bar_x_category in categories_numeric else categories_categorical[bar_x_category]
    bar_x_vars_in_category = [var['display_name'] for group in bar_x_variable_groups.values() for var in group]
    if bar_x_vars_in_category:
        bar_x_var_display = st.sidebar.selectbox("막대그래프의 x축 변수 선택", bar_x_vars_in_category, key="bar_x_var")
        bar_x_var = display_name_to_var_name[bar_x_var_display]
    else:
        st.sidebar.write(f"선택한 카테고리 '{bar_x_category}'에 변수가 없습니다.")
        bar_x_var = None
else:
    bar_x_var = None

st.sidebar.subheader("막대그래프의 y축 변수 선택")
bar_y_categories = ["선택 안 함"] + list(categories_numeric.keys()) + list(categories_categorical.keys())
bar_y_category = st.sidebar.selectbox("막대그래프의 y축 변수의 카테고리 선택", bar_y_categories, index=0, key="bar_y_category")

if bar_y_category != "선택 안 함":
    bar_y_variable_groups = categories_numeric[bar_y_category] if bar_y_category in categories_numeric else categories_categorical[bar_y_category]
    bar_y_vars_in_category = [var['display_name'] for group in bar_y_variable_groups.values() for var in group]
    if bar_y_vars_in_category:
        bar_y_var_display = st.sidebar.selectbox("막대그래프의 y축 변수 선택", bar_y_vars_in_category, key="bar_y_var")
        bar_y_var = display_name_to_var_name[bar_y_var_display]
    else:
        st.sidebar.write(f"선택한 카테고리 '{bar_y_category}'에 변수가 없습니다.")
        bar_y_var = None
else:
    bar_y_var = None

bar_chart_type = st.sidebar.selectbox(
    "막대그래프 유형 선택",
    options=["평균 값 막대그래프", "합계 막대그래프", "중앙값 막대그래프"],
    index=0,
    key="bar_chart_type_new"
)

st.markdown('<div class="subheader">### 막대그래프</div>', unsafe_allow_html=True)

if bar_x_var and bar_y_var:
    aggregation_funcs = {"평균 값 막대그래프": (np.mean, "평균 값"),
                         "합계 막대그래프": (np.sum, "합계"),
                         "중앙값 막대그래프": (np.median, "중앙값")}
    agg_func, y_label = aggregation_funcs.get(bar_chart_type, (np.mean, "평균 값"))
    bar_data = df.groupby(bar_x_var)[bar_y_var].agg(agg_func).reset_index()
    bar_data.columns = ['Category', 'Value']
    bar_data['Category'] = bar_data['Category'].astype(str)
    fig_bar = px.bar(
        bar_data,
        x='Category',
        y='Value',
        title=f"{variable_labels.get(bar_y_var, bar_y_var)}의 {bar_chart_type} (x축: {variable_labels.get(bar_x_var, bar_x_var)} [{bar_x_var}])",
        labels={"Value": y_label, "Category": f"{variable_labels.get(bar_x_var, bar_x_var)} [{bar_x_var}]"},
        text='Value',
        color='Category',
        color_discrete_sequence=px.colors.qualitative.Plotly,
        width=800,
        height=600
    )
    fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_bar.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.write("막대그래프를 그릴 변수를 선택하세요.")

# 교차표 분석 섹션
st.markdown('<div class="section-header">## 교차표 분석</div>', unsafe_allow_html=True)
st.sidebar.header("교차표")
st.sidebar.subheader("교차표의 첫 번째 변수 선택")
crosstab_x_categories = ["선택 안 함"] + list(categories_numeric.keys()) + list(categories_categorical.keys())
crosstab_x_category = st.sidebar.selectbox("교차표의 첫 번째 변수의 카테고리 선택", crosstab_x_categories, index=0, key="crosstab_x_category")

if crosstab_x_category != "선택 안 함":
    crosstab_x_variable_groups = categories_numeric[crosstab_x_category] if crosstab_x_category in categories_numeric else categories_categorical[crosstab_x_category]
    crosstab_x_vars_in_category = [var['display_name'] for group in crosstab_x_variable_groups.values() for var in group]
    if crosstab_x_vars_in_category:
        crosstab_x_var_display = st.sidebar.selectbox("교차표의 첫 번째 변수 선택", crosstab_x_vars_in_category, key="crosstab_x_var")
        crosstab_x_var = display_name_to_var_name[crosstab_x_var_display]
    else:
        st.sidebar.write(f"선택한 카테고리 '{crosstab_x_category}'에 변수가 없습니다.")
        crosstab_x_var = None
else:
    crosstab_x_var = None

st.sidebar.subheader("교차표의 두 번째 변수 선택")
crosstab_y_categories = ["선택 안 함"] + list(categories_numeric.keys()) + list(categories_categorical.keys())
crosstab_y_category = st.sidebar.selectbox("교차표의 두 번째 변수의 카테고리 선택", crosstab_y_categories, index=0, key="crosstab_y_category")

if crosstab_y_category != "선택 안 함":
    crosstab_y_variable_groups = categories_numeric[crosstab_y_category] if crosstab_y_category in categories_numeric else categories_categorical[crosstab_y_category]
    crosstab_y_vars_in_category = [var['display_name'] for group in crosstab_y_variable_groups.values() for var in group]
    if crosstab_y_vars_in_category:
        crosstab_y_var_display = st.sidebar.selectbox("교차표의 두 번째 변수 선택", crosstab_y_vars_in_category, key="crosstab_y_var")
        crosstab_y_var = display_name_to_var_name[crosstab_y_var_display]
    else:
        st.sidebar.write(f"선택한 카테고리 '{crosstab_y_category}'에 변수가 없습니다.")
        crosstab_y_var = None
else:
    crosstab_y_var = None

if crosstab_x_var and crosstab_y_var:
    def convert_to_categorical(var):
        if df[var].dtype == 'object':
            return var
        else:
            binned_var = f"{var}_binned"
            df[binned_var] = pd.cut(df[var], bins=5)
            return binned_var

    crosstab_x_var_cat = convert_to_categorical(crosstab_x_var)
    crosstab_y_var_cat = convert_to_categorical(crosstab_y_var)
    contingency_table = pd.crosstab(df[crosstab_x_var_cat], df[crosstab_y_var_cat])
    contingency_table.index = contingency_table.index.astype(str)
    contingency_table.columns = contingency_table.columns.astype(str)
    # 축에 한글 변수명 추가
    x_label = variable_labels.get(crosstab_x_var, crosstab_x_var)
    y_label = variable_labels.get(crosstab_y_var, crosstab_y_var)
    fig_crosstab = px.imshow(
        contingency_table,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"'{x_label}'와 '{y_label}'의 교차표 히트맵",
        labels={"color": "빈도수", "x": x_label, "y": y_label},
        width=800,
        height=600
    )
    st.plotly_chart(fig_crosstab, use_container_width=True)
    st.markdown('<div class="subheader">### 교차표 데이터 검정 (카이제곱 검정)</div>', unsafe_allow_html=True)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    st.write(f"**카이제곱 통계량:** {chi2:.4f}")
    st.write(f"**자유도:** {dof}")
    st.write(f"**p-값:** {p:.10f}")
    expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
    st.dataframe(expected_df, use_container_width=True)
    st.write("**결론:** 두 변수 간에 유의한 상관관계가 있습니다." if p < 0.05 else "두 변수 간에 유의한 상관관계가 없습니다.")
    if contingency_table.shape == (2, 2):
        oddsratio, fisher_p = fisher_exact(contingency_table)
        st.write(f"**Fisher's Exact Test p-값:** {fisher_p:.10f}")
        st.write("**Fisher's Exact Test 결론:** 두 변수 간에 유의한 상관관계가 있습니다." if fisher_p < 0.05 else "두 변수 간에 유의한 상관관계가 없습니다.")
else:
    st.write("교차표를 생성하려면 두 개의 변수를 선택하세요.")

# Raw Data 보기 섹션
st.markdown('<div class="section-header">## Raw Data 보기</div>', unsafe_allow_html=True)
st.sidebar.header("Raw Data 보기")
raw_data_categories = ["모두 선택"] + list(categories_numeric.keys()) + list(categories_categorical.keys())
raw_data_selected_category = st.sidebar.selectbox("Raw Data를 볼 카테고리 선택", raw_data_categories, index=0, key="raw_data_category")
raw_data_selected_vars = []

if raw_data_selected_category != "모두 선택":
    raw_data_variable_groups = categories_numeric[raw_data_selected_category] if raw_data_selected_category in categories_numeric else categories_categorical[raw_data_selected_category]
    raw_data_variables = [var for group in raw_data_variable_groups.values() for var in group]
    raw_data_variable_options = [var['display_name'] for var in raw_data_variables]
    raw_data_selected_vars = st.sidebar.multiselect("Raw Data로 보고 싶은 변수를 선택하세요.", options=raw_data_variable_options, default=[], key="raw_data_vars")
else:
    raw_data_variable_options = [f"{variable_labels.get(var, var)} [{var}]" for var in df.columns]
    raw_data_selected_vars = st.sidebar.multiselect("Raw Data로 보고 싶은 변수를 선택하세요.", options=raw_data_variable_options, default=[], key="raw_data_vars_all")

if raw_data_selected_vars:
    raw_data_var_names = [display_name_to_var_name[dn] for dn in raw_data_selected_vars]
    raw_data_display = df[raw_data_var_names].head(100)
    st.markdown('<div class="subheader">### 선택한 변수들의 Raw Data (상위 100개 행)</div>', unsafe_allow_html=True)
    st.dataframe(raw_data_display, use_container_width=True)
else:
    st.write("Raw Data로 보고 싶은 변수를 선택하세요.")

# 자료 출처 섹션 추가
st.markdown('<div class="section-header">## 자료 출처</div>', unsafe_allow_html=True)

# 자료 출처 데이터 준비
source_data = {
    '조사차수': ['제19차'],
    '조사년도': ['2023년'],
    '조사기간': ['08.28.~10.19.'],
    '조사대상': ['중1~고3'],
    '목표대상 수(명)': ['56,935'],
    '참여자 수(명)': ['52,880'],
    '참여율(%)': ['92.9'],
    '링크': ['https://www.kdca.go.kr/yhs/']
}

# 데이터프레임 생성
source_df = pd.DataFrame(source_data)

# 링크를 클릭 가능하게 변경
source_df['링크'] = source_df['링크'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')

# 자료 출처 표 표시 (CSS 클래스 적용)
st.markdown(source_df.to_html(escape=False, index=False, classes='source-table'), unsafe_allow_html=True)


# 외부 URL 이미지 표시
image_url = "https://i.postimg.cc/y8d9RMbT/3.png"

# 이미지 중앙 정렬과 스타일 적용
st.markdown(f'''
    <div style='text-align: center; margin-top: 50px;'>
        <img src="{image_url}" alt="Footer Image" style="width: 50%; max-width: 250px; height: auto; ">
    </div>
    ''', unsafe_allow_html=True)

# 실행 명령어 안내 (주석 처리됨)
# streamlit run "C:\Users\USER\OneDrive\학교\2024 2학년\2학기\7 인피\2304_김성윤.py"
