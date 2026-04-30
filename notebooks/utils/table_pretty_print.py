import pandas as pd
import ipywidgets as widgets
from IPython.display import display, Markdown, HTML
from great_tables import GT


def _value_to_dataframe(value, *, dict_key_col="key", dict_value_col="value"):
    """
    Convert common notebook values into a DataFrame suitable for Great Tables.

    Plain dictionaries become one row per key:
        {"a": 1, "b": 2}
    becomes:
        key | value
        a   | 1
        b   | 2
    """

    if isinstance(value, pd.DataFrame):
        return value

    if isinstance(value, pd.Series):
        return value.reset_index(name=dict_value_col).rename(
            columns={"index": dict_key_col}
        )

    if isinstance(value, dict):
        if not value:
            return pd.DataFrame()

        # Dict of dicts:
        # {"theta_1": {"mean": 0.1, "sd": 0.2}}
        # becomes key | mean | sd
        if all(isinstance(v, dict) for v in value.values()):
            return pd.DataFrame.from_dict(value, orient="index").reset_index(
                names=dict_key_col
            )

        # Everything else becomes key | value rows.
        # This is the bit you want for artifact_summary.
        return pd.DataFrame(
            [
                {
                    dict_key_col: key,
                    dict_value_col: value_,
                }
                for key, value_ in value.items()
            ]
        )

    try:
        return pd.DataFrame(value)
    except ValueError:
        return pd.DataFrame([{dict_value_col: value}])


def _normalize_elements(
    elements,
    *,
    default_name="Summary",
    tab_key="name",
    title_key="name",
    value_key="value",
):
    """
    Accept either:
    - a list of element dicts:
        [{"name": "...", "value": ...}, ...]
    - a single plain dict:
        {"artifact_dir": "...", "created_at": "..."}
    - a single DataFrame / Series / scalar
    """

    # Single plain dictionary, not already an element spec.
    if isinstance(elements, dict) and value_key not in elements:
        return [
            {
                tab_key: default_name,
                title_key: default_name,
                value_key: elements,
            }
        ]

    # Single element dictionary:
    # {"name": "Artifact summary", "value": {...}}
    if isinstance(elements, dict) and value_key in elements:
        return [elements]

    # Already a list/tuple of element specs.
    if isinstance(elements, (list, tuple)):
        return list(elements)

    # Single DataFrame / Series / scalar.
    return [
        {
            tab_key: default_name,
            title_key: default_name,
            value_key: elements,
        }
    ]


def make_gt_tabs(
    elements,
    *,
    tab_key="name",
    title_key="name",
    value_key="value",
    empty_message="No data available.",
    use_raw_html=False,
    gt_transform=None,
    dict_key_col="key",
    dict_value_col="value",
    default_name="Summary",
):
    elements = _normalize_elements(
        elements,
        default_name=default_name,
        tab_key=tab_key,
        title_key=title_key,
        value_key=value_key,
    )

    tab_children = []

    for element in elements:
        title = element.get(title_key, "")
        tab_label = element.get(tab_key, title)
        value = element.get(value_key)

        out = widgets.Output()

        with out:
            if value is None:
                display(Markdown(f"_{empty_message}_"))
            else:
                df = _value_to_dataframe(
                    value,
                    dict_key_col=dict_key_col,
                    dict_value_col=dict_value_col,
                )

                if df.empty:
                    display(Markdown(f"_{empty_message}_"))
                else:
                    table = GT(df).tab_header(title=title)

                    if gt_transform is not None:
                        table = gt_transform(table, element)

                    if use_raw_html:
                        display(HTML(table.as_raw_html()))
                    else:
                        display(table)

        tab_children.append((tab_label, out))

    tabs = widgets.Tab(children=[child for _, child in tab_children])

    for i, (tab_label, _) in enumerate(tab_children):
        tabs.set_title(i, str(tab_label))

    return tabs


def default_style_gt_table(table, element):
    df = _value_to_dataframe(
        element["value"],
        dict_key_col="field",
        dict_value_col="value",
    )

    table = table.sub_missing(missing_text="").opt_row_striping()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Don't format label columns.
    numeric_cols = [col for col in numeric_cols if col not in {"key", "field", "name"}]

    if numeric_cols:
        table = table.fmt_number(
            columns=numeric_cols,
            decimals=4,
        )

    return table
