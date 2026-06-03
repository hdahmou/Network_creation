from __future__ import annotations

from pathlib import Path
from math import hypot
from typing import Any, Mapping


def graph_to_tikz(
    G: Any,
    scale: float,
    *,
    pos: Mapping[Any, tuple[float, float]] | None = None,
    node_size: float | Mapping[Any, float] | None = None,
    edge_width: float = 0.01,
    node_color: str = "white",
    edge_color: str = "black",
    node_border_color: str | None = "black",
    node_border_width: float = 0.003,
    arrows: bool | None = None,
    arrow_length_factor: float = 2.0,
    arrow_width_factor: float = 3.0,
    precision: int = 4,
) -> str:
    """
    Export either a netgraph.Graph object or a NetworkX graph to TikZ.

    For netgraph.Graph:
        - uses G.node_positions
        - uses G.node_size unless node_size is explicitly passed
        - uses G.edge_paths, preserving curved/routed edges

    For NetworkX:
        - uses `pos` if passed
        - otherwise uses node attribute "pos"
        - otherwise falls back to nx.spring_layout(G)
        - draws straight edges

    Parameters are in Netgraph/data units before multiplication by `scale`.
    """

    is_netgraph = all(
        hasattr(G, attr) for attr in ("node_positions", "node_size", "edge_paths")
    )
    is_networkx = (
        hasattr(G, "nodes") and hasattr(G, "edges") and hasattr(G, "is_directed")
    )

    if not (is_netgraph or is_networkx):
        raise TypeError("Expected either a netgraph.Graph object or a NetworkX graph.")

    # ---------- normalize graph geometry ----------

    if is_netgraph:
        nodes = list(G.node_positions)
        positions = dict(G.node_positions)
        sizes = _node_size_dict(
            node_size if node_size is not None else G.node_size, nodes, default=0.02
        )
        edge_paths = dict(G.edge_paths)

        if arrows is None:
            arrows = False

    else:
        import networkx as nx

        nodes = list(G.nodes)

        if pos is None:
            pos = nx.get_node_attributes(G, "pos")

        if not pos or len(pos) < len(nodes):
            pos = nx.spring_layout(G)

        positions = {node: tuple(pos[node]) for node in nodes}
        sizes = _node_size_dict(node_size, nodes, default=0.02)

        if G.is_multigraph():
            edges = list(G.edges(keys=True))
        else:
            edges = list(G.edges())

        edge_paths = {edge: [positions[edge[0]], positions[edge[1]]] for edge in edges}

        if arrows is None:
            arrows = bool(G.is_directed())

    # ---------- tiny formatting helpers ----------

    def fmt(x: float) -> str:
        x = float(x)
        if abs(x) < 10 ** (-precision):
            x = 0.0
        return f"{x:.{precision}f}".rstrip("0").rstrip(".") or "0"

    def cm(x: float) -> str:
        return f"{fmt(scale * x)}cm"

    def coord(p: Any) -> str:
        x, y = float(p[0]), float(p[1])
        return f"({fmt(scale * x)},{fmt(scale * y)})"

    def tikz_color(c: str) -> str:
        # Allows normal TikZ colors like "blue!20" and also HTML hex.
        c = str(c).strip()
        if c.startswith("#") and len(c) == 7:
            name = "ng" + c[1:].lower()
            return name
        return c

    def color_defs(*colors: str | None) -> list[str]:
        defs = []
        seen = set()
        for c in colors:
            if isinstance(c, str) and c.startswith("#") and len(c) == 7:
                name = "ng" + c[1:].lower()
                if name not in seen:
                    defs.append(rf"\definecolor{{{name}}}{{HTML}}{{{c[1:].upper()}}}")
                    seen.add(name)
        return defs

    node_fill = tikz_color(node_color)
    edge_draw = tikz_color(edge_color)
    node_draw = tikz_color(node_border_color) if node_border_color else None

    # ---------- path trimming, so arrows are visible ----------

    def path_len(path: list[tuple[float, float]]) -> float:
        return sum(
            hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(path, path[1:])
        )

    def trim_start(
        path: list[tuple[float, float]], d: float
    ) -> list[tuple[float, float]]:
        path = list(path)
        while d > 0 and len(path) >= 2:
            x0, y0 = path[0]
            x1, y1 = path[1]
            seg = hypot(x1 - x0, y1 - y0)

            if seg == 0:
                path.pop(0)
                continue

            if d < seg:
                t = d / seg
                path[0] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
                return path

            d -= seg
            path.pop(0)

        return path

    def trim_ends(
        path: list[Any], start: float, end: float
    ) -> list[tuple[float, float]]:
        path = [(float(x), float(y)) for x, y in path]

        if len(path) < 2:
            return path

        total = path_len(path)
        requested = start + end

        if requested > 0.95 * total and requested > 0:
            factor = 0.95 * total / requested
            start *= factor
            end *= factor

        path = trim_start(path, start)
        path = list(reversed(trim_start(list(reversed(path)), end)))
        return path

    # ---------- emit TikZ ----------

    lines = []
    lines += color_defs(node_color, edge_color, node_border_color)
    lines.append(r"% Requires: \usepackage{xcolor}")
    lines.append(r"% Requires: \usepackage{tikz}")
    if arrows:
        lines.append(r"% Requires: \usetikzlibrary{arrows.meta}")
    lines.append(r"\begin{tikzpicture}[x=1cm,y=1cm,line cap=round,line join=round]")

    # Edges first.
    for edge, raw_path in edge_paths.items():
        source, target = edge[0], edge[1]

        if source == target:
            x, y = positions[source]
            r = sizes[source]
            opts = _edge_options(
                edge_draw,
                cm(edge_width),
                arrows,
                cm(edge_width * arrow_length_factor),
                cm(edge_width * arrow_width_factor),
            )
            lines.append(
                rf"\draw[{opts}] {coord((x + r, y))} "
                rf"to[out=45,in=315,looseness=8] {coord((x - r, y))};"
            )
            continue

        path = trim_ends(
            raw_path,
            start=sizes[source],
            end=sizes[target],
        )

        if len(path) < 2:
            continue

        opts = _edge_options(
            edge_draw,
            cm(edge_width),
            arrows,
            cm(edge_width * arrow_length_factor),
            cm(edge_width * arrow_width_factor),
        )

        lines.append(rf"\draw[{opts}] " + " -- ".join(coord(p) for p in path) + ";")

    # Nodes second, so they mask edge interiors.
    for node in nodes:
        r = sizes[node]
        opts = [
            "circle",
            f"minimum size={cm(2 * r)}",
            "inner sep=0pt",
            f"fill={node_fill}",
        ]

        if node_draw is None:
            opts.append("draw=none")
        else:
            opts += [
                f"draw={node_draw}",
                f"line width={cm(node_border_width)}",
            ]

        lines.append(rf"\node[{','.join(opts)}] at {coord(positions[node])} {{}};")

    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def save_graph_tikz(G: Any, scale: float, filename: str | Path, **kwargs: Any) -> None:
    Path(filename).write_text(graph_to_tikz(G, scale, **kwargs), encoding="utf-8")


def _node_size_dict(
    node_size: float | Mapping[Any, float] | None,
    nodes: list[Any],
    *,
    default: float,
) -> dict[Any, float]:
    if node_size is None:
        return {node: default for node in nodes}

    if isinstance(node_size, Mapping):
        return {node: float(node_size.get(node, default)) for node in nodes}

    return {node: float(node_size) for node in nodes}


def _edge_options(
    edge_color: str,
    edge_width_cm: str,
    arrows: bool,
    arrow_length_cm: str,
    arrow_width_cm: str,
) -> str:
    opts = [
        f"draw={edge_color}",
        f"line width={edge_width_cm}",
    ]

    if arrows:
        opts.append(rf"-{{Latex[length={arrow_length_cm},width={arrow_width_cm}]}}")

    return ",".join(opts)
