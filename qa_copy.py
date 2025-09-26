import streamlit as st
import ezdxf
import networkx as nx
import matplotlib.pyplot as plt
import tempfile
from matplotlib.patches import Patch

st.set_page_config(page_title="ELCAD Drawing QA Assistant", layout="wide")

# --- Global state ---
if "graph" not in st.session_state:
    st.session_state.graph = nx.Graph()
if "symbols" not in st.session_state:
    st.session_state.symbols = []
if "wires" not in st.session_state:
    st.session_state.wires = []
if "labels" not in st.session_state:
    st.session_state.labels = []
if "qa_results" not in st.session_state:
    st.session_state.qa_results = {
        "graph_queries": [],
        "connectivity": [],
        "labels": [],
        "compliance": [],
        "layout": [],
        "documentation": []
    }

# --- Tabs ---
tabs = st.tabs([
    "1. Upload DWG/DXF",
    "2. Parsed Symbols",
    "3. Knowledge Graph",
    "4. Enrich Graph with Semantics",
    "5. Run QA as Graph Queries",
    "6. Connectivity Issues",
    "7. Symbol & Labelling Errors",
    "8. Compliance & Standards",
    "9. Layout & Design Rules",
    "10. Documentation Lists",
    "11. QA Reporting"
])

import tempfile
import ezdxf
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

with tabs[0]:
    st.header("Upload & Preview DXF File")
    uploaded_file = st.file_uploader("Upload an ELCAD drawing (.dxf only)", type=["dxf"])

    if uploaded_file:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load DXF
        doc = ezdxf.readfile(tmp_path)
        msp = doc.modelspace()

        # 🔹 Parse entities immediately (for later tabs)
        st.session_state.symbols = [
            {"pos": (e.dxf.insert.x, e.dxf.insert.y), "name": e.dxf.name}
            for e in msp.query("INSERT")
        ]
        st.session_state.labels = [
            {"pos": (e.dxf.insert.x, e.dxf.insert.y), "text": e.dxf.text}
            for e in msp.query("TEXT MTEXT")
        ]
        st.session_state.wires = [
            {"start": (e.dxf.start.x, e.dxf.start.y), "end": (e.dxf.end.x, e.dxf.end.y)}
            for e in msp.query("LINE")
        ]

        # 🔹 Render with Matplotlib (simple static preview)
        fig, ax = plt.subplots(figsize=(8, 8))
        ctx = RenderContext(doc)
        backend = MatplotlibBackend(ax)
        frontend = Frontend(ctx, backend)
        frontend.draw_layout(msp, finalize=True)
        ax.set_title("DXF Preview")
        st.pyplot(fig)

        st.success(f"DXF parsed ✅: {len(st.session_state.symbols)} symbols, "
                   f"{len(st.session_state.labels)} labels, "
                   f"{len(st.session_state.wires)} wires")

# --------------------------------------------------
# Tab 2: Parsed Symbols
# --------------------------------------------------
with tabs[1]:
    st.header("Parsed Entities from DXF")

    if st.session_state.symbols:
        # Plot symbols
        fig, ax = plt.subplots()
        for i, s in enumerate(st.session_state.symbols):
            x, y = s["pos"]
            ax.scatter(x, y, marker="o", color="blue")
            ax.text(x, y, s["name"], fontsize=8, color="red")
        ax.set_title("Parsed Symbols (positions from DXF)")
        st.pyplot(fig)

        st.subheader("Symbols (JSON)")
        st.json(st.session_state.symbols)
    else:
        st.info("No symbols parsed yet. Upload a DXF in Tab 1.")

    st.subheader("Wires (first 20 shown)")
    st.write(st.session_state.wires[:20])

    st.subheader("Labels")
    st.write(st.session_state.labels)



# --------------------------------------------------
# Tab 3: Knowledge Graph (Symbols, Labels, Wires)
# --------------------------------------------------
# --------------------------------------------------
# Tab 3: Knowledge Graph (Symbols, Labels, Wires)
# --------------------------------------------------
# --------------------------------------------------
# Tab 3: Knowledge Graph (Symbols, Labels, Wires) - Readable Static View
# --------------------------------------------------
from matplotlib.patches import Patch

with tabs[2]:
    st.header("Knowledge Graph (Symbols, Labels, Wires)")

    G = nx.Graph()

    # --- Add symbols
    for i, s in enumerate(st.session_state.symbols):
        try:
            pos = (float(s["pos"][0]), float(s["pos"][1]))
            node_id = f"S{i}"
            G.add_node(node_id, type="symbol", label=s["name"], pos=pos)
        except Exception:
            continue

    # --- Add labels
    for i, l in enumerate(st.session_state.labels):
        try:
            pos = (float(l["pos"][0]), float(l["pos"][1]))
            node_id = f"L{i}"
            G.add_node(node_id, type="label", label=l["text"], pos=pos)
        except Exception:
            continue

    # --- Symbol–Label association (nearest)
    for i, lbl in enumerate(st.session_state.labels):
        try:
            lx, ly = map(float, lbl["pos"])
        except Exception:
            continue

        nearest_symbol, nearest_dist = None, float("inf")
        for j, sym in enumerate(st.session_state.symbols):
            try:
                sx, sy = map(float, sym["pos"])
                dist = ((lx - sx) ** 2 + (ly - sy) ** 2) ** 0.5
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_symbol = f"S{j}"
            except Exception:
                continue
        if nearest_symbol and nearest_dist < 200:
            G.add_edge(f"L{i}", nearest_symbol, type="label_association")

    # --- Add wires as nodes (simplified)
    for i, (start, end) in enumerate(st.session_state.wires):
        try:
            start = (float(start[0]), float(start[1]))
            end = (float(end[0]), float(end[1]))
        except Exception:
            continue

        wire_node = f"W{i}"
        G.add_node(wire_node, type="wire", label=f"Wire{i}")

        for j, sym in enumerate(st.session_state.symbols):
            try:
                sx, sy = map(float, sym["pos"])
                if (min(abs(sx - start[0]), abs(sx - end[0])) < 50 and
                    min(abs(sy - start[1]), abs(sy - end[1])) < 50):
                    G.add_edge(f"S{j}", wire_node)
            except Exception:
                continue

    st.session_state.graph = G

    # --- Colors
    colors = []
    for node, data in G.nodes(data=True):
        if data["type"] == "symbol":
            colors.append("blue")
        elif data["type"] == "label":
            colors.append("green")
        else:
            colors.append("grey")

    # --- Use Kamada-Kawai for readability
    pos_graph = nx.kamada_kawai_layout(G)

    # --- Draw graph with smaller labels/nodes
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    nx.draw(
        G, pos_graph,
        with_labels=True,
        labels={n: n for n in G.nodes()},  # show only IDs, not full names
        node_color=colors,
        node_size=200,
        font_size=6,
        ax=ax2
    )

    legend_elements = [
        Patch(facecolor='blue', label='Symbol'),
        Patch(facecolor='green', label='Label'),
        Patch(facecolor='grey', label='Wire/Connection')
    ]
    ax2.legend(handles=legend_elements, loc='best')
    st.pyplot(fig2)

    # --- Show full names separately
    st.subheader("📋 Node Details")
    details = [
        {"ID": n, "Type": d["type"], "Full Label": d.get("label", "")}
        for n, d in G.nodes(data=True)
    ]
    st.table(details)
# --------------------------------------------------
# Tab 4: Enrich Graph with Semantics (Hybrid GPT + Rules) - Fixed
# --------------------------------------------------
with tabs[3]:
    st.header("Symbols with Semantic Subtypes")

    if not st.session_state.symbols:
        st.warning("No symbols parsed yet. Upload a DXF in Tab 1.")
    else:
        # ---- Rule-based classifier
        def rule_based_classify(name):
            name = name.upper()
            if "CT" in name or "CURRENT" in name:
                return "current_transformer"
            if "TV" in name or "TF" in name or "TRANSF" in name:
                return "transformer"
            if "R" in name or "RELAY" in name:
                return "relay"
            if "CB" in name or "BR" in name or "BREAKER" in name:
                return "breaker"
            if "GND" in name or "GROUND" in name:
                return "ground"
            return "generic"

        # ---- Try GPT classification (fallback to rules if fails)
        import os, json, re, openai
        from dotenv import load_dotenv
        load_dotenv()

        gpt_map = {}
        try:
            client = openai.AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),  # 2024-12-01-preview
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )

            with st.spinner("Classifying symbols with GPT…"):
                chat_resp = client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),  # gpt-4o-raj
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in electrical CAD QA. "
                                       "Always return valid JSON only."
                        },
                        {
                            "role": "user",
                            "content": """
                            Classify each DXF symbol into one of these:
                            Relay, Current Transformer, Breaker, Ground, Transformer, Generic.
                            Input: symbol names from the drawing.
                            Output: ONLY valid JSON array.
                            Example:
                            [
                              {"name": "X1", "type": "relay"},
                              {"name": "CT1", "type": "current_transformer"}
                            ]
                            """
                        }
                    ],
                    temperature=0
                )

            gpt_raw = (chat_resp.choices[0].message.content or "").strip()

            # Remove markdown fences if present
            if gpt_raw.startswith("```"):
                gpt_raw = re.sub(r"^```(json)?", "", gpt_raw, flags=re.IGNORECASE).strip()
                gpt_raw = re.sub(r"```$", "", gpt_raw).strip()

            if gpt_raw:
                try:
                    gpt_labels = json.loads(gpt_raw)
                    gpt_map = {
                        s["name"]: s["type"].lower()
                        for s in gpt_labels if "name" in s and "type" in s
                    }
                except Exception as e:
                    st.warning(f"⚠️ GPT output was not valid JSON, using rules only. Error: {e}")
                    st.text_area("GPT Raw Output", gpt_raw, height=200)
                    gpt_map = {}
            else:
                st.warning("⚠️ GPT returned empty output → using rule-based classification only.")
                gpt_map = {}

        except Exception as e:
            st.error(f"GPT classification failed completely. Error: {e}")
            gpt_map = {}

        # ---- Merge GPT + Rules
        categories = {}
        for s in st.session_state["symbols"]:
            name = s["name"]
            gpt_type = gpt_map.get(name, "generic")
            if gpt_type == "generic":
                categories[name] = rule_based_classify(name)
            else:
                categories[name] = gpt_type

        # ---- Build semantic graph (with IDs only)
        G = nx.Graph()
        id_map = {}
        for i, s in enumerate(st.session_state["symbols"]):
            node_id = f"S{i}"
            id_map[node_id] = s["name"]  # keep mapping for table
            G.add_node(node_id, category=categories.get(s["name"], "generic"))

        # ---- Layout
        pos = nx.kamada_kawai_layout(G)

        # ---- Color map
        color_map = {
            "relay": "red",
            "current_transformer": "blue",
            "breaker": "orange",
            "ground": "green",
            "transformer": "purple",
            "generic": "grey",
        }
        node_colors = [
            color_map.get(G.nodes[node]["category"], "grey")
            for node in G.nodes
        ]

        # ---- Draw
        fig, ax = plt.subplots(figsize=(10, 6))
        nx.draw(
            G, pos,
            with_labels=True,
            labels={n: n for n in G.nodes()},  # show only IDs (S1, S2…)
            node_color=node_colors,
            node_size=500,
            font_size=6,
            ax=ax
        )

        # ---- Legend
        legend_elements = [
            Patch(facecolor=color_map[k], label=k.capitalize())
            for k in color_map
        ]
        ax.legend(handles=legend_elements, loc="lower left")
        ax.set_title("Symbols with Semantic Subtypes (via GPT + Rules)")
        st.pyplot(fig)

        # ---- Details Table
        st.subheader("📋 Symbol Classification Details")
        details = [
            {"ID": nid, "Original Name": id_map[nid], "Category": G.nodes[nid]["category"]}
            for nid in G.nodes
        ]
        st.table(details)

# --------------------------------------------------
# Tab 5: Run QA as Graph Queries
# --------------------------------------------------
# --------------------------------------------------
# Tab 5: Run QA as Graph Queries (structure + semantics)
# --------------------------------------------------
with tabs[4]:
    st.header("Run QA as Graph Queries")

    issues = []

    for node, data in st.session_state.graph.nodes(data=True):
        if data["type"] == "symbol":
            category = data.get("category", "generic")  # semantic type if available

            # --- Check if symbol has a linked label
            linked_labels = [
                n for n in st.session_state.graph.neighbors(node)
                if st.session_state.graph.nodes[n]["type"] == "label"
            ]
            if not linked_labels:
                issues.append(f"⚠️ {category.capitalize()} '{node}' has no label")

    # --- Check dangling wires
    for node, data in st.session_state.graph.nodes(data=True):
        if data["type"] == "wire" and st.session_state.graph.degree(node) < 2:
            issues.append(f"⚠️ Wire {node} is dangling")

    if not issues:
        issues.append("✅ No graph query issues found")

    # Deduplicate
    unique_issues = list(dict.fromkeys(issues))
    st.session_state.qa_results["graph_queries"] = unique_issues
    st.write(unique_issues)


# --------------------------------------------------
# Tab 6: Connectivity Issues (extended)
# --------------------------------------------------
with tabs[5]:
    st.header("Connectivity Issues")

    issues = []
    for n, d in st.session_state.graph.nodes(data=True):
        if d["type"] == "symbol":
            category = d.get("category", "generic")

            # Check if the symbol has any wire connections
            connected = any(
                st.session_state.graph.nodes[nb]["type"] == "wire"
                for nb in st.session_state.graph.neighbors(n)
            )
            if not connected:
                issues.append(f"⚠️ {category.capitalize()} '{n}' is not connected to any wire")

    # --- Wiring continuity checks
    for node, data in st.session_state.graph.nodes(data=True):
        if data["type"] == "wire":
            deg = st.session_state.graph.degree(node)
            if deg < 2:
                issues.append(f"⚠️ Wire {node} is dangling (no continuity)")
            elif deg > 4:
                issues.append(f"⚠️ Wire {node} has too many connections (possible short circuit)")

    if not issues:
        issues.append("✅ No connectivity/continuity issues found")

    unique_issues = list(dict.fromkeys(issues))
    st.session_state.qa_results["connectivity"] = unique_issues
    st.write(unique_issues)
# --------------------------------------------------
# Tab 7: Symbol & Labelling Errors (extended)
# --------------------------------------------------
with tabs[6]:
    st.header("Symbol & Labelling Errors")

    results = []

    # --- Existing symbol-to-wire checks
    for sym in st.session_state.symbols:
        try:
            sx, sy = map(float, sym["pos"])
            connected = False
            for w in st.session_state.wires:
                start = tuple(map(float, w["start"]))
                end   = tuple(map(float, w["end"]))
                if (min(abs(sx - start[0]), abs(sx - end[0])) < 50 and
                    min(abs(sy - start[1]), abs(sy - end[1])) < 50):
                    connected = True
                    break
            if not connected:
                category = "generic"
                if sym["name"] in st.session_state.graph.nodes:
                    category = st.session_state.graph.nodes[sym["name"]].get("category", "generic")
                results.append(f"⚠️ {category.capitalize()} '{sym['name']}' is not connected to any wire")
        except Exception as e:
            results.append(f"⚠️ Could not check connectivity for {sym.get('name','?')} (error: {e})")

    # --- Advanced Label QA
    seen_labels = set()
    for i, l in enumerate(st.session_state.labels):
        text = l.get("text", "").strip()
        pos = l.get("pos", (0, 0))
        if text in seen_labels:
            results.append(f"⚠️ Duplicate label '{text}' detected")
        else:
            seen_labels.add(text)

        # Check for distant labels (>200 units from nearest symbol)
        nearest = min(
            (((pos[0]-s["pos"][0])**2 + (pos[1]-s["pos"][1])**2)**0.5, s["name"])
            for s in st.session_state.symbols
        )
        if nearest[0] > 200:
            results.append(f"⚠️ Label '{text}' is too far from nearest symbol '{nearest[1]}'")

    if results:
        st.subheader("Labelling Findings")
        for r in dict.fromkeys(results):
            st.write(r)
    else:
        st.success("✅ No labelling issues found")

    st.session_state.qa_results["labels"] = results
# --------------------------------------------------
# Tab 8: Compliance & Standards (extended)
# --------------------------------------------------
with tabs[7]:
    st.header("Compliance & Standards")

    issues = []

    # --- Drawing-wide checks
    if not any("title" in n.lower() for n in st.session_state.graph.nodes):
        issues.append("⚠️ Missing title block")

    if not any("rev" in d.get("text","").lower() for _, d in st.session_state.graph.nodes(data=True)):
        issues.append("⚠️ Missing revision info")

    # --- Type-specific rules
    for n, d in st.session_state.graph.nodes(data=True):
        if d["type"] == "symbol":
            category = d.get("category", "generic")

            if category == "breaker":
                linked_labels = [
                    nb for nb in st.session_state.graph.neighbors(n)
                    if st.session_state.graph.nodes[nb]["type"] == "label"
                ]
                if not linked_labels:
                    issues.append(f"⚠️ Breaker '{n}' missing mandatory label")

            if category == "transformer":
                connected = any(
                    st.session_state.graph.nodes[nb]["type"] == "wire"
                    for nb in st.session_state.graph.neighbors(n)
                )
                if not connected:
                    issues.append(f"⚠️ Transformer '{n}' not wired — violates standards")

            if category == "ground":
                # At least one ground must exist globally
                pass

    # --- Safety checks
    if not any(d.get("category") == "ground" for _, d in st.session_state.graph.nodes(data=True)):
        issues.append("⚠️ No ground connection found (safety violation)")

    # --- Metadata checks
    if not any("project" in d.get("text","").lower() for _, d in st.session_state.graph.nodes(data=True)):
        issues.append("⚠️ Missing project reference metadata")

    if not issues:
        issues.append("✅ No compliance/safety issues found")

    unique_issues = list(dict.fromkeys(issues))
    st.session_state.qa_results["compliance"] = unique_issues
    st.write(unique_issues)
# --------------------------------------------------
# Tab 9: Layout & Design Rules (extended)
# --------------------------------------------------
with tabs[8]:
    st.header("Layout & Design Rules")

    issues = []
    positions = {}

    for n, d in st.session_state.graph.nodes(data=True):
        if d["type"] == "symbol":
            pos = d.get("pos", (0, 0))
            category = d.get("category", "generic")

            # Overlap detection
            for other, (opos, ocat) in positions.items():
                if abs(pos[0] - opos[0]) < 10 and abs(pos[1] - opos[1]) < 10:
                    issues.append(
                        f"⚠️ {category.capitalize()} '{n}' overlaps with {ocat.capitalize()} '{other}'"
                    )
            positions[n] = (pos, category)

            # --- Symbol misuse check
            if category == "generic":
                issues.append(f"⚠️ Symbol '{n}' not recognized → possible misuse or unknown symbol")

    if not issues:
        issues.append("✅ No layout/misuse issues found")

    st.session_state.qa_results["layout"] = list(dict.fromkeys(issues))
    st.write(st.session_state.qa_results["layout"])

# --------------------------------------------------
# Tab 10: Documentation Lists (extended)
# --------------------------------------------------
with tabs[9]:
    st.header("Documentation Lists")

    issues = []

    if not st.session_state.symbols:
        issues.append("⚠️ No device list extracted from drawing")

    if len(st.session_state.labels) < len(st.session_state.symbols):
        issues.append("⚠️ Fewer labels than symbols overall, possible missing entries")

    category_counts = {}
    for n, d in st.session_state.graph.nodes(data=True):
        if d["type"] == "symbol":
            cat = d.get("category", "generic")
            category_counts[cat] = category_counts.get(cat, 0) + 1

    if st.session_state.labels and category_counts:
        label_count = len(st.session_state.labels)
        for cat, count in category_counts.items():
            if count > label_count:
                issues.append(f"⚠️ More {cat.capitalize()} symbols ({count}) than labels ({label_count})")

    # --- Revision control (placeholder: requires prior baseline)
    # Compare against stored baseline if available
    if "previous_symbols" in st.session_state:
        if len(st.session_state.symbols) != len(st.session_state.previous_symbols):
            issues.append("⚠️ Device list changed since last revision")

    if not issues:
        issues.append("✅ No documentation/revision issues found")

    st.session_state.qa_results["documentation"] = list(dict.fromkeys(issues))
    st.write(st.session_state.qa_results["documentation"])


# --------------------------------------------------
# Tab 11: Final QA Reporting (grouped by QA type + category)
# --------------------------------------------------
with tabs[10]:
    st.header("Final QA Reporting")

    report_lines = []

    for section, findings in st.session_state.qa_results.items():
        report_lines.append(f"\n=== {section.upper()} ===")

        # --- Organize findings by category if possible
        categorized = {}
        for f in findings:
            # Try to extract category name from the issue text
            matched = None
            for cat in ["Relay", "Breaker", "Transformer", "Ground", "Current_transformer", "Generic"]:
                if cat.lower() in f.lower():
                    matched = cat.capitalize()
                    break
            if not matched:
                matched = "General"

            categorized.setdefault(matched, []).append(f)

        # Write grouped results
        for cat, cat_findings in categorized.items():
            report_lines.append(f"\n-- {cat} --")
            for cf in dict.fromkeys(cat_findings):  # dedupe
                report_lines.append(cf)

    report_text = "\n".join(report_lines)

    st.text_area("Final QA Report", report_text, height=500)
    st.download_button(
        "⬇️ Download QA Report",
        report_text.encode("utf-8"),
        "QA_Report.txt"
    )

