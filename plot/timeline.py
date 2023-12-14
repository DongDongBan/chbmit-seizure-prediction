import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Sequence, Tuple
# from plotinfo_validate import pydantic_validate, plotinfo_schema_validate # TODO

class Record:
    file: str
    span: Tuple[datetime]
    info: str

class Seizure:
    span: Tuple[datetime]
    info: str

def default_preplot_hook(json_obj):
    # TODO Add validate func here
    for rec in json_obj["record_lst"]:
        rec["span"] = [datetime.strptime(rec["span"][0], dt_fmt), datetime.strptime(rec["span"][1], dt_fmt)]
    for rec in json_obj["seizure_lst"]:
        rec["span"] = [datetime.strptime(rec["span"][0], dt_fmt), datetime.strptime(rec["span"][1], dt_fmt)]

# TODO 现在默认是绿色-橙色-灰色，以后不指定具体颜色常量了，接入plotly主题
def _get_raw_timeline(title: str, record_seq: Sequence[Record], seizure_seq: Sequence[Seizure], unused_collection = {}, **kwargs) -> go.Figure:
    onset_points = go.Scatter(x=tuple(ost["span"][0] for ost in seizure_seq), y=[0] * len(seizure_seq), mode="markers", hovertext=tuple(ost["info"] for ost in seizure_seq), 
                    xaxis='x', yaxis='y', name="Onset", legendrank=1e1, 
                    marker_size=8, marker_color="orange")

    ### Generate Rects Attr
    base = [seg["span"][0] for seg in record_seq]
    length = [(seg["span"][1] - seg["span"][0]).total_seconds() * 1000 for seg in record_seq] # Unit(ms)
    text = [seg["info"] for seg in record_seq]
    color = ["MediumSeaGreen" if seg["file"] not in unused_collection else "DarkMagenta" for seg in record_seq]
    # opacity = [1.0] * len(record_seq)
    shape = []; 
    for seg in record_seq:
        if seg["file"] in unused_collection:    shape.append('x')
        elif "masked_collection" in kwargs and seg["file"] in kwargs["masked_collection"]:  shape.append('.')
        else:                                   shape.append('')


    offset=-4; width=4# ; ymin=offset+width
    recs_bar = go.Bar(
        base=base,
        x=length, 
        y=[0] * len(record_seq),
        hovertext=text, 
        orientation='h', 
        marker=dict(
            color=color,
            # opacity=opacity, 
            pattern_shape=shape, 
        ), 
        name="Record", 
        offset=offset, 
        width=[width] * len(record_seq), 
    )

    base = [seg["span"][0] for seg in seizure_seq]
    length = [(seg["span"][1] - seg["span"][0]).total_seconds() * 1000 for seg in seizure_seq]
    text = [seg["info"] for seg in seizure_seq]
    color = "Gold"

    offset=0; width=4# ; ymin=offset+width
    szs_bar = go.Bar(
        base=base,
        x=length, 
        y=[0] * len(seizure_seq),
        hovertext=text, 
        orientation='h', 
        marker=dict(
            color=color,
            # opacity=opacity, 
        ), 
        name="Seizure", 
        offset=offset, 
        width=[width] * len(seizure_seq), 
    )    

    fig = go.Figure(data=[onset_points, recs_bar, szs_bar], layout_title=title)
    if len(seizure_seq) == 0: fig.update_xaxes(type='date') # When a patient have an empty seizure-seq , it is necessary to set this axis attribute

    ### Emphasize xaxis & Onset Point
    fig.add_hline(y=0, line_color="gray", line_width=1)

    ### Ajust global Layout
    fig.update_layout(
        yaxis=dict(
            range=[-8, 8] if "yaxis_range" not in kwargs else kwargs["yaxis_range"], 
            fixedrange=False, 
        )
    )
    fig.update_xaxes(
        range=[record_seq[0]["span"][0] - timedelta(hours=1), record_seq[-1]["span"][1] + timedelta(hours=1)], 
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=1, label="对齐到天", step="day",  stepmode="todate"),
                dict(count=1, label="1d", step="day",  stepmode="backward"),
            ])
        )
    )

    return fig

def get_fig_from_obj(title, info_obj, preplot_hook = default_preplot_hook, afterplot_hook = lambda x: None):
    preplot_hook(info_obj)
    fig = _get_raw_timeline(title=title, record_seq=info_obj["record_lst"], seizure_seq=info_obj["seizure_lst"], unused_collection=set(fn for fn in info_obj["record_lst"]))
    afterplot_hook(fig)   
     
    return fig