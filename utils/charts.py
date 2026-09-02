"""Reusable chart helpers so gauges, radars, and hierarchy charts look
consistent across every PRISM page instead of each page hand-rolling Plotly
config."""

import plotly.graph_objects as go
import plotly.express as px

PRIMARY = "#0B3D66"
PRIMARY_LIGHT = "#3B82F6"
ACCENT = "#D97706"
SUCCESS = "#15803D"
DANGER = "#B91C1C"
NEUTRAL = "#94A3B8"

BAND_COLORS = {"Excellent": SUCCESS, "Good": PRIMARY, "Fair": ACCENT, "Needs Review": DANGER}


def gauge_chart(value: float, title: str, min_val=300, max_val=900,
                 band_edges=(550, 650, 750), height=240):
    """A speedometer-style gauge for the PRISM Score (or any bounded metric),
    with colored bands so the number's context is visible at a glance instead
    of needing a legend."""
    low, mid, high = band_edges
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 15}},
        number={"font": {"size": 34, "color": PRIMARY}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickwidth": 1, "tickcolor": NEUTRAL},
            "bar": {"color": PRIMARY, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [min_val, low], "color": "#FEE2E2"},
                {"range": [low, mid], "color": "#FEF3C7"},
                {"range": [mid, high], "color": "#DBEAFE"},
                {"range": [high, max_val], "color": "#DCFCE7"},
            ],
            "threshold": {"line": {"color": DANGER, "width": 3}, "thickness": 0.8, "value": value},
        },
    ))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def radar_chart(categories: list, values: list, title: str = "", color=PRIMARY, height=380,
                 range_max=100):
    """A spider/radar chart for multi-dimension breakdowns (e.g. the four
    PRISM Score checks) — makes the shape of strengths/weaknesses visible in
    one glance instead of four separate metric tiles."""
    cats_closed = categories + [categories[0]]
    vals_closed = values + [values[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed, fill="toself",
        line=dict(color=color, width=2), fillcolor=color, opacity=0.75,
        marker=dict(size=6),
    ))
    fig.update_traces(fillcolor=_hex_to_rgba(color, 0.25))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, range_max], tickfont=dict(size=9))),
        showlegend=False, title=title, height=height,
        margin=dict(l=40, r=40, t=40, b=20),
    )
    return fig


def multi_radar_chart(categories: list, series: dict, title: str = "", height=420, range_max=1.0):
    """Overlay multiple radar traces (e.g. one per investor risk profile) on
    the same axes for direct visual comparison."""
    fig = go.Figure()
    colors = [PRIMARY, ACCENT, SUCCESS, "#7C3AED", "#0EA5E9"]
    cats_closed = categories + [categories[0]]
    for i, (name, values) in enumerate(series.items()):
        vals_closed = list(values) + [values[0]]
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed, name=name,
            line=dict(color=color, width=2), opacity=0.85,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, range_max], tickfont=dict(size=9))),
        showlegend=True, legend=dict(orientation="h", y=-0.15), title=title, height=height,
        margin=dict(l=40, r=40, t=40, b=20),
    )
    return fig


def donut_chart(labels: list, values: list, title: str = "", height=340,
                 colors=None):
    colors = colors or [PRIMARY, PRIMARY_LIGHT, "#93C5FD", ACCENT, "#FCD34D", NEUTRAL, "#7C3AED"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=colors[:len(labels)]),
        textinfo="label+percent", textfont=dict(size=11),
    ))
    fig.update_layout(title=title, height=height, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def treemap_chart(df, path: list, values_col: str, title: str = "", height=460, color_col=None):
    fig = px.treemap(
        df, path=path, values=values_col, title=title, color=color_col,
        color_continuous_scale=[[0, "#DBEAFE"], [0.5, PRIMARY_LIGHT], [1, PRIMARY]] if color_col else None,
        color_discrete_sequence=[PRIMARY, PRIMARY_LIGHT, "#93C5FD", ACCENT] if not color_col else None,
    )
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_traces(textfont_size=12)
    return fig


def map_chart(df, lat_col="lat", lon_col="lon", color_col=None, size_col=None,
              hover_name=None, hover_data=None, title="", height=520, zoom=3.6,
              color_continuous_scale=None):
    """An actual India map (OpenStreetMap tiles, no token required) — for a
    product whose whole premise is hyperlocal geography, a real map reads as
    far more 'suitable' than another abstract chart."""
    fig = px.scatter_map(
        df, lat=lat_col, lon=lon_col, color=color_col, size=size_col,
        hover_name=hover_name, hover_data=hover_data, title=title,
        color_continuous_scale=color_continuous_scale or [[0, "#FEE2E2"], [0.5, "#FEF3C7"], [1, SUCCESS]],
        zoom=zoom, height=height, size_max=28,
        center={"lat": 22.5, "lon": 79.0},
    )
    fig.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=50, b=0))
    return fig


def network_graph_chart(G, ring_entities: set, title: str = "", height=520, max_nodes=200):
    """A force-directed buyer↔seller transaction network, with ring-member
    entities highlighted — the graph structure itself is the signal here
    (a cluster of entities trading only among themselves), which no
    per-transaction bar chart could show."""
    import networkx as nx

    if G.number_of_nodes() > max_nodes:
        # keep all ring members plus a random sample of the rest for legibility
        ring_nodes = [n for n in G.nodes if n in ring_entities]
        other_nodes = [n for n in G.nodes if n not in ring_entities]
        sample_size = max(0, max_nodes - len(ring_nodes))
        rng = __import__("numpy").random.default_rng(7)
        sampled = list(rng.choice(other_nodes, size=min(sample_size, len(other_nodes)), replace=False))
        keep = set(ring_nodes) | set(sampled)
        G = G.subgraph(keep).copy()

    pos = nx.spring_layout(G, seed=42, k=0.6)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.7, color="#CBD5E1"),
                              hoverinfo="none", mode="lines")

    node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        is_ring = node in ring_entities
        node_color.append(DANGER if is_ring else PRIMARY_LIGHT)
        node_size.append(14 if is_ring else 7)
        degree = G.degree(node)
        node_text.append(f"{node}<br>Connections: {degree}{'<br><b>Ring member</b>' if is_ring else ''}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers", hoverinfo="text", text=node_text,
        marker=dict(color=node_color, size=node_size, line=dict(width=1, color="white")),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title, showlegend=False, height=height,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


PROPERTY_ICONS = {
    "Apartment": "🏢", "Villa": "🏡", "Independent House": "🏠",
    "Penthouse": "🏙️", "Studio/1RK": "🚪", "Row House": "🏘️", "Plot/Land": "📐",
}


def property_icon_svg(property_type: str, color=PRIMARY, size=48):
    """Minimal, original line-art icons per property type (no stock imagery/
    copyright concerns) — used as small visual badges next to type selectors."""
    c = color
    icons = {
        "Apartment": f'<rect x="10" y="6" width="28" height="38" rx="1" fill="none" stroke="{c}" stroke-width="2"/><rect x="15" y="12" width="5" height="5" fill="{c}"/><rect x="24" y="12" width="5" height="5" fill="{c}"/><rect x="15" y="21" width="5" height="5" fill="{c}"/><rect x="24" y="21" width="5" height="5" fill="{c}"/><rect x="20" y="34" width="8" height="10" fill="{c}"/>',
        "Villa": f'<path d="M6 24 L24 8 L42 24" fill="none" stroke="{c}" stroke-width="2"/><rect x="10" y="24" width="28" height="20" fill="none" stroke="{c}" stroke-width="2"/><rect x="20" y="32" width="8" height="12" fill="{c}"/><rect x="13" y="28" width="5" height="5" fill="{c}"/><rect x="30" y="28" width="5" height="5" fill="{c}"/>',
        "Independent House": f'<path d="M8 22 L24 9 L40 22" fill="none" stroke="{c}" stroke-width="2"/><rect x="12" y="22" width="24" height="20" fill="none" stroke="{c}" stroke-width="2"/><rect x="20" y="30" width="8" height="12" fill="{c}"/><rect x="15" y="26" width="4" height="4" fill="{c}"/>',
        "Penthouse": f'<rect x="8" y="18" width="32" height="26" fill="none" stroke="{c}" stroke-width="2"/><rect x="14" y="8" width="20" height="12" fill="none" stroke="{c}" stroke-width="2"/><rect x="20" y="34" width="8" height="10" fill="{c}"/><line x1="8" y1="26" x2="40" y2="26" stroke="{c}" stroke-width="1.5"/>',
        "Studio/1RK": f'<rect x="10" y="14" width="28" height="30" fill="none" stroke="{c}" stroke-width="2"/><circle cx="24" cy="27" r="6" fill="none" stroke="{c}" stroke-width="1.5"/><rect x="20" y="35" width="8" height="9" fill="{c}"/>',
        "Row House": f'<rect x="4" y="20" width="14" height="24" fill="none" stroke="{c}" stroke-width="2"/><rect x="18" y="16" width="14" height="28" fill="none" stroke="{c}" stroke-width="2"/><rect x="32" y="20" width="12" height="24" fill="none" stroke="{c}" stroke-width="2"/>',
        "Plot/Land": f'<rect x="8" y="10" width="32" height="28" fill="none" stroke="{c}" stroke-width="2" stroke-dasharray="4,3"/><line x1="8" y1="10" x2="40" y2="38" stroke="{c}" stroke-width="1.5"/>',
    }
    body = icons.get(property_type, icons["Apartment"])
    return f'<svg width="{size}" height="{size}" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">{body}</svg>'



    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
