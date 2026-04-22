import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from netgraph import Graph


def draw_network(graph: nx.DiGraph, ax: plt.Axes) -> Graph:
    node_labels = {n: str(n) for n in graph.nodes()}
    mpl.rcParams.update(mpl.rcParamsDefault)
    return Graph(
        graph=graph,
        node_labels=node_labels,
        edge_label_fontdict=dict(
            size=8,
            fontweight="normal",
            color="b",
            bbox=dict(
                boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0), alpha=0.9
            ),
        ),
        node_layout="spring",
        node_layout_kwargs={"pad_by": 0.01},
        edge_layout="curved",
        node_shape="o",
        node_label_fontdict=dict(size=8, fontweight="normal"),
        node_size=3,
        edge_width=0.5,
        edge_color="#d111b7",
        arrows=True,
        ax=ax,
    )
