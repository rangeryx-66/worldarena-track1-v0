#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

REASONS = [
    'good','robot_or_gripper_not_visible','contact_not_visible','object_moves_without_contact','no_task_progress',
    'action_video_mismatch','severe_flicker_or_exposure_jump','broken_or_black_frames','bad_compression_or_noise',
    'wrong_domain_or_wrong_robot','severe_physics_issue','ambiguous'
]
REASON_LABELS_ZH = {
    'good': '质量好 / 可用',
    'robot_or_gripper_not_visible': '机器人或夹爪不可见',
    'contact_not_visible': '关键接触不可见',
    'object_moves_without_contact': '物体无接触移动',
    'no_task_progress': '几乎无任务进展',
    'action_video_mismatch': '动作和视频不匹配',
    'severe_flicker_or_exposure_jump': '严重闪烁或曝光跳变',
    'broken_or_black_frames': '坏帧或黑帧',
    'bad_compression_or_noise': '压缩/噪声严重',
    'wrong_domain_or_wrong_robot': '场景或机器人不对',
    'severe_physics_issue': '严重物理问题',
    'ambiguous': '不确定 / 边界样本',
}
GUIDE = '''
**PASS** = 可以作为 SFT/A2V 正例训练数据。  
**REJECT** = 不应该作为 SFT/A2V 正例。

不要因为这些原因自动 REJECT：白背景、轻微 render grain、机械臂主体不完整入镜、夹爪从画面外进入、黑白机械臂占比变化大、局部轻微模糊或噪点。

应当 REJECT：视频坏了/黑帧/打不开、几乎没有动作或任务进展、关键时刻机器人/夹爪完全不可见但物体明显在动、物体无接触 teleport、action 和 video 明显不对应、严重曝光/颜色跳变/闪烁、物体消失/形变/严重穿模、明显不是目标机器人或目标场景。

边界样本二分类偏向 REJECT。
'''


def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    return ap.parse_args()


ANNOTATION_COLUMNS = ['human_label','human_reason','human_confidence','notes','annotated_at','annotator']


def ensure_annotation_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in ANNOTATION_COLUMNS:
        if c not in df.columns:
            df[c] = ''
    return df


def merge_existing_annotations(base_df: pd.DataFrame, labeled_df: pd.DataFrame) -> pd.DataFrame:
    base_df = ensure_annotation_columns(base_df.copy()).fillna('')
    labeled_df = ensure_annotation_columns(labeled_df.copy()).fillna('')
    if labeled_df.empty:
        return base_df

    keyed = {}
    if {'sample_id', 'episode_id'}.issubset(labeled_df.columns):
        for _, r in labeled_df.iterrows():
            key = (str(r.get('sample_id', '')), str(r.get('episode_id', '')))
            if str(r.get('human_label', '')):
                keyed[key] = r
    by_episode = {}
    if 'episode_id' in labeled_df.columns:
        for _, r in labeled_df.iterrows():
            eid = str(r.get('episode_id', ''))
            if str(r.get('human_label', '')):
                by_episode[eid] = r

    for idx, r in base_df.iterrows():
        key = (str(r.get('sample_id', '')), str(r.get('episode_id', '')))
        old = keyed.get(key) or by_episode.get(str(r.get('episode_id', '')))
        if old is None:
            continue
        for c in ANNOTATION_COLUMNS:
            base_df.at[idx, c] = old.get(c, '')
    return base_df


def load_data(base: Path, out: Path) -> pd.DataFrame:
    base_df = pd.read_csv(base).fillna('') if base.exists() else pd.DataFrame()
    if not out.exists():
        return ensure_annotation_columns(base_df)
    labeled_df = pd.read_csv(out).fillna('')
    if base_df.empty:
        return ensure_annotation_columns(labeled_df)
    if len(base_df) != len(labeled_df):
        return merge_existing_annotations(base_df, labeled_df)
    if 'episode_id' in base_df.columns and 'episode_id' in labeled_df.columns:
        if base_df['episode_id'].astype(str).tolist() != labeled_df['episode_id'].astype(str).tolist():
            return merge_existing_annotations(base_df, labeled_df)
    return ensure_annotation_columns(labeled_df)


def save_data(df: pd.DataFrame, base: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    df.to_csv(base, index=False)
    labeled = df['human_label'].astype(str).str.len().gt(0).sum()
    if labeled and labeled % 20 == 0:
        bdir = out.parent/'backups'; bdir.mkdir(parents=True, exist_ok=True)
        ts=datetime.now().strftime('%Y%m%d_%H%M%S')
        shutil.copy2(out, bdir/f'manual_qc_500_labeled_{ts}.csv')


def first_unlabeled_index(df: pd.DataFrame, indices: list[int], start_after: int = -1) -> int:
    for idx in indices:
        if idx > start_after and not str(df.at[idx,'human_label']):
            return idx
    for idx in indices:
        if not str(df.at[idx,'human_label']):
            return idx
    return indices[0] if indices else 0


def show_video_compact(st, video_path: Path, width: int = 430):
    try:
        st.video(str(video_path), width=width)
    except TypeError:
        st.video(str(video_path))


def main():
    args=parse_args()
    try:
        import streamlit as st
    except Exception as exc:
        raise SystemExit('Streamlit is not installed. Install with: pip install streamlit') from exc

    st.set_page_config(page_title='Manual QC 500', layout='wide')
    st.markdown(
        '''
<style>
  .block-container {
    padding-top: 1.15rem;
    padding-bottom: 0.4rem;
    padding-left: 0.65rem;
    padding-right: 0.65rem;
    max-width: 1680px;
  }
  [data-testid="stVerticalBlock"] { gap: 0.28rem; }
  [data-testid="stHorizontalBlock"] { gap: 0.45rem; }
  h1 { font-size: 1.25rem !important; margin: 0 0 0.15rem 0 !important; }
  h2, h3 { font-size: 0.95rem !important; margin: 0.15rem 0 !important; }
  p, li, label, .stMarkdown, .stCaption { font-size: 0.88rem !important; }
  div[data-testid="stMetric"] { padding: 0.05rem 0; }
  div[data-testid="stMetricValue"] { font-size: 1.0rem; }
  div[data-testid="stMetricLabel"] { font-size: 0.72rem; }
  .stButton button { padding: 0.22rem 0.45rem; min-height: 1.9rem; }
  div[data-baseweb="tab-list"] { gap: 0.25rem; }
  button[data-baseweb="tab"] { padding: 0.2rem 0.45rem; height: 1.9rem; }
  .stTextArea textarea { font-size: 0.82rem; }
</style>
''',
        unsafe_allow_html=True,
    )
    if 'df' not in st.session_state or st.session_state.get('csv_path') != str(args.csv) or st.session_state.get('out_path') != str(args.out):
        st.session_state.df = load_data(args.csv, args.out)
        st.session_state.csv_path = str(args.csv)
        st.session_state.out_path = str(args.out)
        st.session_state.idx = first_unlabeled_index(st.session_state.df, list(range(len(st.session_state.df))))
    df = st.session_state.df

    st.sidebar.header('Filters')
    only_unlabeled = st.sidebar.checkbox('show only unlabeled', value=False)
    sample_groups = ['all'] + sorted([x for x in df.get('sample_group', pd.Series(dtype=str)).astype(str).unique() if x])
    group = st.sidebar.selectbox('sample_group', sample_groups)
    tasks = ['all'] + sorted([x for x in df.get('task_family', pd.Series(dtype=str)).astype(str).unique() if x])
    task = st.sidebar.selectbox('task_family', tasks)
    qcs = ['all'] + sorted([x for x in df.get('current_qc_status', pd.Series(dtype=str)).astype(str).unique() if x])
    qc = st.sidebar.selectbox('current_qc_status', qcs)
    labels = ['all','unlabeled','PASS','REJECT']
    hl = st.sidebar.selectbox('human_label', labels)
    confs = ['all','1','2','3']
    hc = st.sidebar.selectbox('human_confidence', confs)
    with st.sidebar:
        st.markdown('---')
        with st.expander('Guideline', expanded=False):
            st.markdown(GUIDE)

    mask = pd.Series(True, index=df.index)
    if only_unlabeled: mask &= df['human_label'].astype(str).eq('')
    if group != 'all': mask &= df['sample_group'].astype(str).eq(group)
    if task != 'all': mask &= df['task_family'].astype(str).eq(task)
    if qc != 'all': mask &= df['current_qc_status'].astype(str).eq(qc)
    if hl == 'unlabeled': mask &= df['human_label'].astype(str).eq('')
    elif hl != 'all': mask &= df['human_label'].astype(str).eq(hl)
    if hc != 'all': mask &= df['human_confidence'].astype(str).eq(hc)
    filtered = df[mask]
    indices = filtered.index.tolist() or df.index.tolist()
    if st.session_state.idx not in indices:
        st.session_state.idx = indices[0] if indices else 0

    labeled = df['human_label'].astype(str).str.len().gt(0).sum()
    pass_n = df['human_label'].astype(str).eq('PASS').sum()
    reject_n = df['human_label'].astype(str).eq('REJECT').sum()
    current = df.loc[st.session_state.idx]

    st.markdown(f"### Manual QC 500 · `{current.get('episode_id','')}`")
    c1,c2,c3,c4,c5 = st.columns([1.05,0.7,0.75,0.8,0.9])
    c1.metric('Labeled / Total', f'{labeled} / {len(df)}')
    c2.metric('PASS', int(pass_n))
    c3.metric('REJECT', int(reject_n))
    c4.metric('Current index', int(st.session_state.idx))
    c5.metric('Group', str(current.get('sample_group','')))

    j1,j2,j3 = st.columns([1.4,0.35,0.55])
    jump_text = j1.text_input('Jump to sample_id / episode_id', value='')
    if j2.button('Jump') and jump_text.strip():
        m = df[(df['sample_id'].astype(str)==jump_text.strip()) | (df['episode_id'].astype(str)==jump_text.strip())]
        if not m.empty:
            st.session_state.idx = int(m.index[0]); st.rerun()
        else:
            st.warning('not found')
    if j3.button('Next unlabeled'):
        st.session_state.idx = first_unlabeled_index(df, indices, st.session_state.idx); st.rerun()

    default_label = str(current.get('human_label','')) if str(current.get('human_label','')) in ['PASS','REJECT'] else 'REJECT'
    existing_reasons = [x for x in str(current.get('human_reason','')).replace('|',',').replace(';',',').split(',') if x in REASONS]
    conf_val = str(current.get('human_confidence','')) if str(current.get('human_confidence','')) in ['1','2','3'] else '2'

    meta_col, media_col, sheet_col = st.columns([0.95, 1.0, 1.35])
    with meta_col:
        st.caption(f"{current.get('sample_id','')} · {current.get('sample_group','')}")
        st.markdown(f"**{current.get('task_family','')}** / `{current.get('robotwin_task_name','')}`")
        st.caption(
            f"T={current.get('T','')} · complexity={current.get('action_complexity_score','')} · "
            f"arm={current.get('dominant_arm','')}"
        )
        st.caption(
            f"QC={current.get('current_qc_status','')} · "
            f"{str(current.get('current_qc_reason',''))[:130]}"
        )
        labels_text = str(current.get('current_qc_labels',''))
        if labels_text:
            st.caption(f"labels: {labels_text[:160]}")
        st.text_area('Prompt', str(current.get('prompt_short','')), height=58, disabled=True)
        st.text_area('Risk', str(current.get('risk_summary','')), height=82, disabled=True)

        st.markdown('**Annotation**')
        label = st.radio('human_label', ['PASS','REJECT'], index=['PASS','REJECT'].index(default_label), horizontal=True, label_visibility='collapsed')
        reason = st.multiselect('human_reason', REASONS, default=existing_reasons, format_func=lambda x: REASON_LABELS_ZH.get(x, x), label_visibility='collapsed')
        confidence = st.select_slider(
            'human_confidence',
            options=['1','2','3'],
            value=conf_val,
            format_func=lambda x: {'1':'1 不确定','2':'2 基本确定','3':'3 很确定'}[x],
            label_visibility='collapsed',
        )
        notes = st.text_area('notes', value=str(current.get('notes','')), height=54, placeholder='notes', label_visibility='collapsed')
        annotator = st.text_input('annotator', value=str(current.get('annotator','')) or 'default', label_visibility='collapsed')

    with media_col:
        video_path = Path(str(current.get('video_path','')))
        if video_path.exists(): show_video_compact(st, video_path, width=430)
        else: st.error(f'video missing: {current.get("video_path","")}')
        fp = Path(str(current.get('first_frame_path','')))
        if fp.exists(): st.image(str(fp), caption='first frame', width=210)

    with sheet_col:
        tabs = st.tabs(['Overview','Motion Peak','Action Peak'])
        for tab, col in zip(tabs, ['overview_sheet','motion_peak_sheet','action_peak_sheet']):
            with tab:
                sheet_path=Path(str(current.get(col,'')))
                if sheet_path.exists(): st.image(str(sheet_path), width=590)
                else: st.warning(f'missing {col}: {sheet_path}')

    def save_current(skip=False):
        idx = st.session_state.idx
        if not skip:
            df.at[idx,'human_label'] = label
            df.at[idx,'human_reason'] = ','.join(reason)
            df.at[idx,'human_confidence'] = confidence
            df.at[idx,'notes'] = notes
            df.at[idx,'annotated_at'] = datetime.now().isoformat(timespec='seconds')
            df.at[idx,'annotator'] = annotator
        save_data(df, args.csv, args.out)

    with meta_col:
        b1,b2,b3,b4,b5 = st.columns(5)
        if b1.button('Save & Next', type='primary', use_container_width=True):
            save_current(); st.session_state.idx = first_unlabeled_index(df, indices, st.session_state.idx); st.rerun()
        if b2.button('Save Only', use_container_width=True):
            save_current(); st.success('saved')
        if b3.button('Previous', use_container_width=True):
            pos = indices.index(st.session_state.idx) if st.session_state.idx in indices else 0
            st.session_state.idx = indices[max(0,pos-1)]; st.rerun()
        if b4.button('Next', use_container_width=True):
            pos = indices.index(st.session_state.idx) if st.session_state.idx in indices else 0
            st.session_state.idx = indices[min(len(indices)-1,pos+1)]; st.rerun()
        if b5.button('Skip', use_container_width=True):
            save_current(skip=True)
            pos = indices.index(st.session_state.idx) if st.session_state.idx in indices else 0
            st.session_state.idx = indices[min(len(indices)-1,pos+1)]; st.rerun()

if __name__ == '__main__':
    main()
