import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

import costs

ROOT = Path.cwd()
ART_DIR, WEIGHTS = ROOT / "artifacts", ROOT / "weights" / "hardhat_yolo11n.pt"
EXAMPLES_DIR = ROOT / "examples"
if not WEIGHTS.exists() or not (ART_DIR / "config.json").exists():
    raise SystemExit("weights/ or artifacts/ missing: run `uv run python main.py train` first.")

config = json.loads((ART_DIR / "config.json").read_text())
OP = config["operating_point"]
CUT, LOW0, HIGH0 = OP["single_cut"], OP["low"], OP["high"]

model = YOLO(str(WEIGHTS))              # loaded, never trained
HEAD = [k for k, v in model.names.items() if v == "head"][0]


def load_scores(split):
    df = pd.read_csv(ART_DIR / f"scores_{split}.csv")
    df["scene"] = np.select(
        [df["positive"] == 0, df["n_helmet"] == 0, df["n_head"] == 1],
        ["negative (all helmets)", "positive: nobody wears a helmet", "positive: 1 head among helmets"],
        default="positive: several heads among helmets")
    return df


SCORES = {"valid": load_scores("valid"), "test": load_scores("test")}
POSITIVES = {
    "all positives": lambda d: d,
    "positives among helmets (site-like)": lambda d: d[(d["positive"] == 0) | (d["n_helmet"] > 0)],
    "one head among helmets (hardest)": lambda d: d[(d["positive"] == 0) | ((d["n_helmet"] > 0) & (d["n_head"] == 1))],
}


def read_csv(name, **kw):
    path = ART_DIR / name
    return pd.read_csv(path, **kw) if path.exists() else pd.DataFrame({"missing": [name]})


def image_or_none(name):
    path = ART_DIR / name
    return str(path) if path.exists() else None


print("loaded", WEIGHTS.name, model.names, "| operating point", OP)

VERDICT = {"pass": ("PASS", "#2e7d32"), "review": ("SEND TO REVIEW", "#ef6c00"), "alert": ("ALERT", "#c62828")}


def zone(score, low, high):
    return "pass" if score < low else ("review" if score < high else "alert")


def predict_image(image, low, high):
    if image is None:
        return None, "Upload an image or pick an example."
    image = image.convert("RGB")
    r = model.predict(image, conf=0.01, imgsz=640, device="cpu", verbose=False)[0]
    cls = r.boxes.cls.numpy().astype(int)
    conf = r.boxes.conf.numpy()
    xyxy = r.boxes.xyxy.numpy()
    heads = cls == HEAD
    score = float(conf[heads].max()) if heads.any() else 0.0

    out = image.copy()
    d = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=max(12, image.size[1] // 45))
    n_heads = n_helmets = 0
    for c, cf, box in zip(cls, conf, xyxy):
        if c == HEAD and cf >= low:
            color, n_heads = "red", n_heads + 1
        elif c != HEAD and cf >= 0.25:
            color, n_helmets = "lime", n_helmets + 1
        else:
            continue
        d.rectangle(box.tolist(), outline=color, width=3)
        d.text((box[0] + 2, box[1] - font.size - 2), f"{model.names[int(c)]} {cf:.2f}", fill=color, font=font)

    z = zone(score, low, high)
    label, color = VERDICT[z]
    text = (f"<h2 style='color:{color};margin:0'>{label}</h2>"
            f"<p>Snapshot score (most confident <code>head</code>): <b>{score:.3f}</b> &nbsp;·&nbsp; "
            f"pass &lt; {low:.2f} ≤ review &lt; {high:.2f} ≤ alert</p>"
            f"<p>{n_heads} head box(es) ≥ low, {n_helmets} helmet box(es) ≥ 0.25.</p>")
    return out, text


EXAMPLES = sorted(str(p) for p in EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.exists() else []

def subset(split, positives):
    return POSITIVES[positives](SCORES[split])


def three_way_matrix(df, low, high):
    zones = np.where(df["score"] < low, "pass", np.where(df["score"] < high, "review", "alert"))
    truth = np.where(df["positive"] == 1, "someone without helmet", "all with helmet")
    cm = pd.crosstab(pd.Series(truth, name="truth"), pd.Series(zones, name="zone"))
    return cm.reindex(index=["someone without helmet", "all with helmet"], columns=["pass", "review", "alert"], fill_value=0)


def matrix_fig(cm, low, high):
    names = [["FN (missed)", "review", "TP (caught)"], ["TN", "review", "FP (false alarm)"]]
    text = [[f"{names[i][j]}<br><b>{cm.iat[i, j]}</b>" for j in range(3)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=cm.values, x=list(cm.columns), y=list(cm.index), text=text, texttemplate="%{text}",
                               colorscale="Blues", showscale=False))
    fig.update_layout(title=f"Snapshots by zone, low = {low:.2f}, high = {high:.2f}", yaxis_autorange="reversed",
                      height=330, margin=dict(t=50, b=20))
    return fig


def designs_table(df, low, high, p, c_fn):
    pos, neg = costs.split_scores(df)
    rows = {
        "A. do nothing": costs.design_cost(pos, neg, costs.NEVER, costs.NEVER, p=p, c_fn=c_fn),
        "B. a person checks every snapshot": costs.design_cost(pos, neg, 0.0, costs.NEVER, p=p, c_fn=c_fn),
        f"C. model alone (chosen cut {CUT:.2f})": costs.design_cost(pos, neg, CUT, CUT, p=p, c_fn=c_fn),
        f"D. current sliders [{low:.2f}, {high:.2f})": costs.design_cost(pos, neg, low, high, p=p, c_fn=c_fn),
    }
    t = pd.DataFrame(rows).T[["eur_per_1000", "eur_per_day", "fn_per_1000", "fp_per_1000", "recall_system",
                              "review_share", "reviews_per_day", "review_hours_per_day"]]
    t.columns = ["€ / 1,000", "€ / day", "FN / 1,000", "FP / 1,000", "recall", "% to review", "reviews / day", "review h / day"]
    t["% to review"] = t["% to review"] * 100
    return t.round(2).reset_index(names="design")


def operating_point(low, high, split, positives, p, c_fn):
    if high < low:
        low, high = high, low
    df = subset(split, positives)
    pos, neg = costs.split_scores(df)
    r = costs.design_cost(pos, neg, low, high, p=p, c_fn=c_fn)
    fits = r["review_hours_per_day"] <= costs.REVIEW_HOURS_AVAILABLE
    summary = (
        f"### {r['eur_per_1000']:.0f} € per 1,000 snapshots · {r['eur_per_day']:.0f} € per day\n"
        f"- **{r['review_share']:.1%}** of snapshots go to review → **{r['reviews_per_day']:.0f} a day** × "
        f"{costs.SECONDS_PER_REVIEW} s = **{r['review_hours_per_day'] * 60:.0f} min/day** "
        f"({'fits in' if fits else '**does not fit** in'} the {costs.REVIEW_HOURS_AVAILABLE:.0f} h available; "
        f"{r['review_hours_per_day'] / 8:.2f} full-time people)\n"
        f"- recall of the whole system **{r['recall_system']:.1%}** · missed violations **{r['fn_per_1000']:.1f}** and "
        f"false alarms **{r['fp_per_1000']:.1f}** per 1,000 snapshots · precision of alerts **{r['alert_precision']:.1%}**\n"
        f"- {len(pos)} positive and {len(neg)} negative images in this {split} subset, re-weighted to a {p:.0%} base rate"
    )
    return matrix_fig(three_way_matrix(df, low, high), low, high), summary, designs_table(df, low, high, p, c_fn)

def cost_curve(low, high, split, positives, p, c_fn):
    fig = go.Figure()
    cuts = np.round(np.arange(0, 1.0001, 0.01), 2)
    for name, dash in [("valid", None), ("test", "dash")]:
        pos, neg = costs.split_scores(subset(name, positives))
        y = [costs.design_cost(pos, neg, c, c, p=p, c_fn=c_fn)["eur_per_1000"] for c in cuts]
        fig.add_trace(go.Scatter(x=cuts, y=y, name=f"model alone, {name}", line=dict(dash=dash, width=3 if not dash else 2)))
        if name == split:
            chosen_y = y[int(round(CUT * 100))]
    pos, neg = costs.split_scores(subset(split, positives))
    for label, lo, hi, color in [("do nothing", costs.NEVER, costs.NEVER, "grey"),
                                 ("person checks everything", 0.0, costs.NEVER, "purple")]:
        fig.add_hline(y=costs.design_cost(pos, neg, lo, hi, p=p, c_fn=c_fn)["eur_per_1000"], line_dash="dot",
                      line_color=color, annotation_text=label, annotation_position="top right")
    band = costs.design_cost(pos, neg, min(low, high), max(low, high), p=p, c_fn=c_fn)["eur_per_1000"]
    fig.add_hline(y=band, line_color="orange", annotation_text=f"three-way at the sliders ({band:.0f} €)",
                  annotation_position="bottom right")
    fig.add_trace(go.Scatter(x=[CUT], y=[chosen_y], mode="markers", marker=dict(size=14, color="red"),
                             name=f"chosen on validation: {CUT:.2f}"))
    for v in (low, high):
        fig.add_vline(x=v, line_dash="dash", line_color="orange")
    fig.update_layout(xaxis_title="threshold on the snapshot score", yaxis_title="€ per 1,000 snapshots",
                      yaxis_rangemode="tozero", height=480,
                      title=f"Cost against threshold ({positives}, base rate {p:.0%}, FN {c_fn:.0f} €)")
    return fig

per_class = read_csv("per_class_test.csv").round(3)
det_baselines = read_csv("baselines_detection.csv").round(3)
img_baselines = read_csv("baselines_image_level.csv").round(3)
box_cm = read_csv("confusion_matrix_test.csv").rename(columns={"Unnamed: 0": ""})
heads = read_csv("head_recall_by_size_test.csv")
if "found" in heads:
    heads["size"] = pd.cut(heads["side_px_640"], [0, 16, 32, 64, 10_000], labels=["<16 px", "16-32 px", "32-64 px", ">64 px"])
    recall_size = heads.groupby("size", observed=False)["found"].agg(recall="mean", heads="count").round(3).reset_index()
else:
    recall_size = heads


def scene_recall(low, high):
    df = SCORES["test"]
    pos = df[df["positive"] == 1]
    t = pos.groupby("scene").agg(images=("image", "count"),
                                 alert=("score", lambda s: (s >= high).mean()),
                                 review=("score", lambda s: ((s >= low) & (s < high)).mean()),
                                 missed=("score", lambda s: (s < low).mean()))
    return t.round(3).reset_index()

ds_inst = read_csv("dataset_instances.csv").round(1)
ds_img = read_csv("dataset_image_level.csv").round(3)
ds_size = read_csv("dataset_object_size.csv").round(3)
scenes = pd.concat([SCORES[s]["scene"].value_counts().rename(s) for s in ("valid", "test")], axis=1).reset_index(names="scene")

CONTROLS = []
with gr.Blocks(title="Hard hat check") as demo:
    gr.Markdown(
        "# Is anyone on site without a hard hat?\n"
        f"`yolo11n` finetuned on Hard Hat Workers v5 ({config['dataset']['url']}). One decision per camera snapshot: its score is "
        "the confidence of the most confident `head` (a head without a helmet). Thresholds chosen on validation: "
        f"model alone at **{CUT:.2f}**, three-way band **[{LOW0:.2f}, {HIGH0:.2f})**. Nothing is retrained here.")
    with gr.Row():
        low = gr.Slider(0, 1, value=LOW0, step=0.01, label="low: below this the snapshot passes")
        high = gr.Slider(0, 1, value=HIGH0, step=0.01, label="high: at or above this the supervisor gets an alert")

    with gr.Tab("Try an image"):
        with gr.Row():
            img_in = gr.Image(type="pil", sources=["upload", "clipboard", "webcam"], label="Camera snapshot")
            img_out = gr.Image(label="Prediction (red = head without helmet ≥ low, green = helmet ≥ 0.25)")
        verdict = gr.HTML()
        if EXAMPLES:
            gr.Examples(EXAMPLES, inputs=img_in, label="Test-set examples (from the clearest cases to the worst miss and false alarm)")
        for trigger in (img_in.change, low.release, high.release):
            trigger(predict_image, [img_in, low, high], [img_out, verdict])

    with gr.Tab("Operating point"):
        with gr.Row():
            split = gr.Radio(["valid", "test"], value="test", label="Split")
            positives = gr.Radio(list(POSITIVES), value="all positives", label="Which positive images count")
        with gr.Row():
            p = gr.Slider(0.005, 0.30, value=costs.BASE_RATE, step=0.005, label="Base rate (share of snapshots with a violation)")
            c_fn = gr.Slider(1, 200, value=costs.C_FN, step=1, label="Cost of a missed violation, FN (€)")
        gr.Markdown(f"TP = FP = {costs.C_TP:.0f} € (a supervisor's walk), review = {costs.C_REVIEW:.2f} € "
                    f"({costs.SECONDS_PER_REVIEW} s at {costs.REVIEWER_EUR_PER_HOUR:.0f} €/h), reviewer recall "
                    f"{costs.HUMAN_RECALL:.0%} / specificity {costs.HUMAN_SPECIFICITY:.0%}, "
                    f"{costs.SNAPSHOTS_PER_DAY:,} snapshots a day, {costs.REVIEW_HOURS_AVAILABLE:.0f} h of review available.")
        with gr.Row():
            cm_plot = gr.Plot()
            summary = gr.Markdown()
        designs = gr.Dataframe(label="The four designs at these assumptions")

    with gr.Tab("Cost curve"):
        curve = gr.Plot()

    with gr.Tab("Per-class results"):
        gr.Markdown("**Positive class: `head`** (a head without a hard hat). The headline number is `head` AP50 and recall; "
                    "`helmet` has 2.6× more instances and lifts the mean.")
        gr.Dataframe(per_class, label="Test, per class (precision and recall at Ultralytics' best-F1 confidence)")
        gr.Dataframe(det_baselines, label="Against the baselines, box level")
        gr.Dataframe(img_baselines, label="Baselines, image level (test)")
        with gr.Row():
            gr.Dataframe(box_cm, label="Box confusion matrix at conf 0.25 (rows = predicted, columns = true)")
            if image_or_none("confusion_matrix_normalized.png"):
                gr.Image(image_or_none("confusion_matrix_normalized.png"), label="Normalised confusion matrix")
        with gr.Row():
            gr.Dataframe(recall_size, label="Head recall by size (conf 0.25, IoU 0.5)")
            scene_table = gr.Dataframe(label="Positive test images by type of scene, at the current sliders")
        with gr.Row():
            for name, label in [("failures_missed.png", "Worst misses"), ("failures_false_alarms.png", "Most confident false alarms"),
                                ("BoxPR_curve.png", "Precision-recall curve (test)")]:
                if image_or_none(name):
                    gr.Image(image_or_none(name), label=label)

    with gr.Tab("Dataset"):
        gr.Markdown(f"Roboflow Universe: `{config['dataset']['workspace']} / {config['dataset']['project']} / "
                    f"{config['dataset']['version']}`. Training: {config['training']}.")
        gr.Dataframe(ds_inst, label="Instances per class and split")
        gr.Dataframe(ds_img, label="Image level: positive = at least one head without a helmet")
        gr.Dataframe(scenes, label="Type of scene")
        gr.Dataframe(ds_size, label="Object size (share of the image, and side in px at 640)")
        for name, label in [("dataset_size_and_scenes.png", "Object size and scenes"), ("results.png", "Training curves")]:
            if image_or_none(name):
                gr.Image(image_or_none(name), label=label)

    inputs = [low, high, split, positives, p, c_fn]
    for ctrl in inputs:
        (ctrl.release if isinstance(ctrl, gr.Slider) else ctrl.change)(
            operating_point, inputs, [cm_plot, summary, designs])
        (ctrl.release if isinstance(ctrl, gr.Slider) else ctrl.change)(cost_curve, inputs, curve)
    for ctrl in (low, high):
        ctrl.release(scene_recall, [low, high], scene_table)
    demo.load(operating_point, inputs, [cm_plot, summary, designs])
    demo.load(cost_curve, inputs, curve)
    demo.load(scene_recall, [low, high], scene_table)

demo.launch(inbrowser=os.environ.get("CI") is None)
