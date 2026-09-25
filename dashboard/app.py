"""CausalMech-Bio — interactive circuit inspector.

    python -m streamlit run dashboard/app.py

Loads circuit JSON files written by ``scripts/discover_circuits.py`` and renders
the causal residues on the 3D structure, coloured by direct causal effect.

Design notes
------------
The app reads *saved* circuits rather than running discovery live. Discovery
takes minutes and needs the model in memory; a dashboard that re-runs it on
every widget interaction would be unusable, and Streamlit's rerun-on-every-input
model makes that trap easy to fall into. Structures are cached across reruns for
the same reason.

The 3D view distinguishes two things that look alike on a static figure: the
mutated site (shown in a distinct colour) and the circuit residues the model
routes through (coloured by effect size). Seeing that the circuit sits *near but
not at* the mutation is usually the moment the result becomes legible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(
    page_title="CausalMech-Bio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CIRCUIT_DIRS = [ROOT / "outputs" / "circuits", ROOT / "outputs"]
PDB_DIR = ROOT / "assets" / "pdb"

# Colour ramp for effect magnitude: pale yellow -> deep red. Sequential, so it
# encodes one ordered quantity, and it stays distinguishable from the grey
# cartoon and the blue mutation marker.
EFFECT_COLORS = ["#fee08b", "#fdae61", "#f46d43", "#d73027", "#a50026"]
MUTATION_COLOR = "#2166ac"


# --------------------------------------------------------------------------- #
# Data loading                                                                  #
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def list_circuits() -> list[str]:
    seen: list[str] = []
    for directory in CIRCUIT_DIRS:
        if directory.exists():
            for path in sorted(directory.rglob("*.json"), key=lambda p: -p.stat().st_mtime):
                if path.name in ("config.json",):
                    continue
                try:
                    payload = json.loads(path.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if "nodes" in payload and "mutation" in payload:
                    if str(path) not in seen:
                        seen.append(str(path))
    return seen


@st.cache_data(show_spinner=False)
def load_circuit(path: str) -> dict:
    return json.loads(Path(path).read_text())


@st.cache_data(show_spinner="Reading structure…")
def read_pdb_text(pdb_id: str) -> str | None:
    """Local PDB text, or ``None``.

    Nothing is downloaded here. If the structure is missing, the app says which
    command fetches it rather than silently reaching for the network — a
    dashboard that downloads on demand is a dashboard that behaves differently
    depending on whether you have wifi.
    """
    for candidate in (f"{pdb_id.lower()}.pdb", f"{pdb_id.upper()}.pdb"):
        path = PDB_DIR / candidate
        if path.exists():
            return path.read_text()
    return None


def effect_color(value: float, vmax: float) -> str:
    if vmax <= 0:
        return EFFECT_COLORS[0]
    idx = min(int(abs(value) / vmax * len(EFFECT_COLORS)), len(EFFECT_COLORS) - 1)
    return EFFECT_COLORS[idx]


# --------------------------------------------------------------------------- #
# Sidebar                                                                       #
# --------------------------------------------------------------------------- #

st.sidebar.title("🧬 CausalMech-Bio")
st.sidebar.caption("Mechanistic circuit inspector")

circuits = list_circuits()
if not circuits:
    st.title("CausalMech-Bio — Mechanistic Circuit Inspector")
    st.warning("No discovered circuits found yet.")
    st.markdown(
        """
Run the pipeline first:

```bash
python scripts/fetch_assets.py --all
python scripts/train_sae.py
python scripts/discover_circuits.py --pdb 1A2Y --mutation A45T --layer 3
```

Circuit JSON files are picked up automatically from `outputs/circuits/`.
"""
    )
    st.stop()

selected = st.sidebar.selectbox(
    "Discovered circuit",
    circuits,
    format_func=lambda p: Path(p).stem,
)
circuit = load_circuit(selected)

nodes = circuit.get("nodes", [])
edges = circuit.get("edges", [])
geometry = circuit.get("geometry")
sequence = circuit.get("sequence", "")
pdb_id = circuit.get("pdb_id")

layers_present = sorted({n["layer"] for n in nodes})
layer_filter = st.sidebar.multiselect(
    "Layers", layers_present, default=layers_present, help="Restrict the view to these layers"
)
visible_nodes = [n for n in nodes if n["layer"] in layer_filter]

st.sidebar.markdown("### Discovered SAE features")
if visible_nodes:
    options = {
        f"L{n['layer']} · f{n['feature']} · DCE {n['dce']:+.3f}": n for n in visible_nodes
    }
    chosen_label = st.sidebar.selectbox("Feature", list(options), index=0)
    focus_node = options[chosen_label]
else:
    focus_node = None

show_all = st.sidebar.checkbox("Show all circuit residues", value=True)
style = st.sidebar.radio("Backbone style", ["cartoon", "stick", "line"], index=0)
highlight = st.sidebar.radio("Circuit style", ["sphere", "stick"], index=0)
show_surface = st.sidebar.checkbox("Translucent surface", value=False)

# --------------------------------------------------------------------------- #
# Header                                                                        #
# --------------------------------------------------------------------------- #

st.title("Mechanistic Circuit Inspector")
mutation = circuit.get("mutation", "?")
st.caption(
    f"{circuit.get('model_name', 'model')} · mutation **{mutation}**"
    + (f" · PDB **{pdb_id}**" if pdb_id else "")
)

cols = st.columns(4)
cols[0].metric(
    "Joint recovery",
    f"{circuit.get('recovered_fraction', 0.0):.1%}",
    help="Fraction of the mutation's effect recovered by patching all circuit features together",
)
cols[1].metric("Circuit nodes", len(nodes))
cols[2].metric("Circuit residues", len(circuit.get("residues", [])))
if geometry and geometry.get("mean_pairwise_distance") == geometry.get("mean_pairwise_distance"):
    cols[3].metric(
        "Mean Cα distance",
        f"{geometry['mean_pairwise_distance']:.1f} Å",
        delta=f"null {geometry['null_mean']:.1f} Å",
        delta_color="off",
    )
else:
    cols[3].metric("Mean Cα distance", "—")

# --------------------------------------------------------------------------- #
# Main layout                                                                   #
# --------------------------------------------------------------------------- #

left, right = st.columns([3, 2])

with left:
    st.subheader("Structure")
    if not pdb_id:
        st.info("This circuit was discovered from a raw sequence, so there is no structure to show.")
    else:
        pdb_text = read_pdb_text(pdb_id)
        if pdb_text is None:
            st.warning(
                f"`{pdb_id}` is not in `assets/pdb/`. Fetch it with:\n\n"
                f"```bash\npython scripts/fetch_assets.py --pdb {pdb_id}\n```"
            )
        else:
            try:
                import py3Dmol
                from stmol import showmol
            except ImportError:
                st.error("Install the viewer: `pip install py3Dmol stmol`")
                st.stop()

            residue_effects: dict[int, float] = {}
            source_nodes = visible_nodes if show_all else ([focus_node] if focus_node else [])
            for node in source_nodes:
                for residue, effect in zip(
                    node.get("residues", []), node.get("position_effects", []), strict=False
                ):
                    residue_effects[residue] = max(
                        residue_effects.get(residue, 0.0), abs(effect)
                    )

            vmax = max(residue_effects.values()) if residue_effects else 1.0

            view = py3Dmol.view(width=760, height=560)
            view.addModel(pdb_text, "pdb")
            view.setStyle({}, {style: {"color": "#d9d9d9"}})

            for residue, effect in residue_effects.items():
                view.addStyle(
                    {"resi": str(residue)},
                    {highlight: {"color": effect_color(effect, vmax), "radius": 0.9}},
                )

            # The mutated site, in a colour no effect value can take.
            mut_residue = "".join(c for c in mutation[1:-1] if c.isdigit())
            if mut_residue:
                view.addStyle(
                    {"resi": mut_residue},
                    {"sphere": {"color": MUTATION_COLOR, "radius": 1.1}},
                )
                view.addResLabels(
                    {"resi": mut_residue},
                    {"fontSize": 11, "backgroundOpacity": 0.6},
                )

            if show_surface:
                view.addSurface(2, {"opacity": 0.55, "color": "#f0f0f0"})

            view.zoomTo()
            showmol(view, height=560, width=760)

            legend = " ".join(
                f"<span style='background:{c};padding:2px 10px;margin-right:2px;"
                f"border-radius:2px;'></span>"
                for c in EFFECT_COLORS
            )
            st.markdown(
                f"<div style='font-size:0.85rem;color:#666'>"
                f"Effect size (|DCE|) low {legend} high &nbsp;&nbsp;"
                f"<span style='background:{MUTATION_COLOR};padding:2px 10px;"
                f"border-radius:2px;'></span> mutated site</div>",
                unsafe_allow_html=True,
            )

with right:
    st.subheader("Causal features")
    if visible_nodes:
        import pandas as pd

        table = pd.DataFrame(
            [
                {
                    "layer": n["layer"],
                    "feature": n["feature"],
                    "DCE": round(n["dce"], 4),
                    "norm.": f"{n['normalized_dce']:.1%}",
                    "residues": ", ".join(str(r) for r in n.get("residues", [])[:8]),
                }
                for n in sorted(visible_nodes, key=lambda n: abs(n["dce"]), reverse=True)
            ]
        )
        st.dataframe(table, hide_index=True, use_container_width=True)
    else:
        st.info("No nodes in the selected layers.")

    if focus_node:
        st.subheader(f"Feature {focus_node['feature']} · layer {focus_node['layer']}")
        residues = focus_node.get("residues", [])
        effects = focus_node.get("position_effects", [])
        if residues and effects:
            import pandas as pd

            st.bar_chart(
                pd.DataFrame({"|DCE|": [abs(e) for e in effects]}, index=[str(r) for r in residues]),
                height=220,
            )
            st.caption("Per-residue direct causal effect of this feature")
        else:
            st.caption("This feature has no localised positional effect.")

    if edges:
        st.subheader("Causal edges")
        import pandas as pd

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "from": f"L{e['src_layer']}·f{e['src_feature']}",
                        "to": f"L{e['dst_layer']}·f{e['dst_feature']}",
                        "weight": round(e["weight"], 3),
                    }
                    for e in sorted(edges, key=lambda e: abs(e["weight"]), reverse=True)
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

# --------------------------------------------------------------------------- #
# Validation                                                                    #
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("Biophysical validation")

if not geometry:
    st.info("No geometry test was run for this circuit (no structure, or fewer than two residues).")
else:
    g = st.columns(4)
    g[0].metric("Mean pairwise Cα", f"{geometry['mean_pairwise_distance']:.2f} Å")
    g[1].metric("Radius of gyration", f"{geometry['radius_of_gyration']:.2f} Å")
    g[2].metric("p-value", f"{geometry['p_value']:.4g}")
    g[3].metric("z-score", f"{geometry['z_score']:+.2f}")

    verdict = geometry.get("verdict", "")
    if geometry.get("significant") and geometry.get("passes_threshold"):
        st.success(f"**{verdict}** — more compact than size-matched random residue sets.")
    elif geometry.get("significant"):
        st.info(f"**{verdict}**")
    else:
        st.warning(f"**{verdict}**")

    st.caption(
        f"Compared against {geometry['n_permutations']:,} random residue sets of the same size "
        f"drawn from the same structure (null mean {geometry['null_mean']:.2f} ± "
        f"{geometry['null_std']:.2f} Å). The size-matched null is the meaningful test; the "
        f"{geometry['threshold_angstrom']:.0f} Å threshold alone depends as much on the protein "
        "as on the circuit."
    )
    if geometry.get("n_unmapped"):
        st.caption(
            f"{geometry['n_unmapped']} circuit residue(s) had no coordinates in the structure "
            "and were excluded."
        )

with st.expander("Sequence and metric details"):
    st.write(
        {
            "clean metric (wild type)": circuit.get("clean_metric"),
            "corrupted metric (mutant)": circuit.get("corrupted_metric"),
            "joint patched metric": circuit.get("joint_metric"),
            "sequence length": len(sequence),
            "discovered at": circuit.get("created_at"),
        }
    )
    if sequence:
        st.code("\n".join(sequence[i : i + 60] for i in range(0, len(sequence), 60)), language=None)
    st.json(circuit.get("config", {}))
