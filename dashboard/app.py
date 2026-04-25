import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

st.set_page_config(page_title="IPL RL Dashboard", layout="wide", page_icon="🏏")

TEAM_NAMES = ["MI", "CSK", "RCB", "KKR", "DC", "RR", "PBKS", "SRH"]
TEAM_COLORS = {
    "MI": "#004BA0",
    "CSK": "#FFCC00",
    "RCB": "#EC1C24",
    "KKR": "#3A225D",
    "DC": "#00008B",
    "RR": "#EA1A85",
    "PBKS": "#AFBED1",
    "SRH": "#F7A721",
}


# Safe loader — always returns default, never crashes
def load_json(fname, default):
    try:
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _team_label_from_winner(raw):
    if isinstance(raw, int) and 0 <= raw < len(TEAM_NAMES):
        return TEAM_NAMES[raw]
    raw_str = str(raw)
    if raw_str.isdigit():
        idx = int(raw_str)
        if 0 <= idx < len(TEAM_NAMES):
            return TEAM_NAMES[idx]
    if raw_str in TEAM_NAMES:
        return raw_str
    return "--"


def _latest_team_metrics_from_rewards():
    path = "training/logs/rewards.csv"
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        if df.empty or "episode" not in df.columns or "team_id" not in df.columns:
            return {}
        latest_ep = int(df["episode"].max())
        latest = df[df["episode"] == latest_ep].copy()
        metrics = {}
        for _, row in latest.iterrows():
            tid = str(row.get("team_id", "")).strip()
            if tid not in TEAM_NAMES:
                continue
            metrics[tid] = {
                "total_reward": float(row.get("TOTAL", 0.0) or 0.0),
                "final_position": int(float(row.get("final_position", 8) or 8)),
                "budget_wasted_cr": float(row.get("budget_wasted_cr", 0.0) or 0.0),
            }
        return metrics
    except Exception:
        return {}


def _extract_lot_events(raw_auction_log):
    lots = []
    for item in raw_auction_log:
        if isinstance(item, dict) and "player_name" in item:
            lots.append(item)
            continue
        # Logger can append episode wrappers: {"episode": N, "data": [...]}
        if isinstance(item, dict) and isinstance(item.get("data"), list):
            for inner in item["data"]:
                if isinstance(inner, dict) and "player_name" in inner:
                    lots.append(inner)
    return lots


def _load_rewards_df():
    path = "training/logs/rewards.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _build_team_aggregates(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame()
    required = {"team_id", "final_position", "budget_wasted_cr"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    agg = (
        df.groupby("team_id", dropna=False)
        .agg(
            champion_count=("final_position", lambda x: int((pd.to_numeric(x, errors="coerce") == 1).sum())),
            avg_final_rank=("final_position", lambda x: float(pd.to_numeric(x, errors="coerce").mean())),
            avg_budget_wasted_cr=("budget_wasted_cr", lambda x: float(pd.to_numeric(x, errors="coerce").mean())),
        )
        .reset_index()
        .rename(columns={"team_id": "team"})
    )
    agg["team"] = agg["team"].astype(str)
    return agg


def _build_behavior_metrics(behaviors):
    if isinstance(behaviors, dict):
        behaviors = [behaviors]
    if not isinstance(behaviors, list) or not behaviors:
        return {}

    team_stats = {t: {"overbid_rate": [], "block_rate": [], "patience_score": [], "bluff_success_rate": [], "labels": []} for t in TEAM_NAMES}
    for ep in behaviors:
        if not isinstance(ep, dict):
            continue
        for tid, team in enumerate(TEAM_NAMES):
            blob = ep.get(str(tid), {})
            if not isinstance(blob, dict):
                continue
            for k in ["overbid_rate", "block_rate", "patience_score", "bluff_success_rate"]:
                team_stats[team][k].append(float(blob.get(k, 0.0) or 0.0))
            team_stats[team]["labels"].append(str(blob.get("label", "?")))

    summary = {}
    for team, vals in team_stats.items():
        summary[team] = {
            "overbid_rate": sum(vals["overbid_rate"]) / max(1, len(vals["overbid_rate"])),
            "block_rate": sum(vals["block_rate"]) / max(1, len(vals["block_rate"])),
            "patience_score": sum(vals["patience_score"]) / max(1, len(vals["patience_score"])),
            "bluff_success_rate": sum(vals["bluff_success_rate"]) / max(1, len(vals["bluff_success_rate"])),
            "labels": vals["labels"],
        }
    return summary


panel = st.sidebar.radio(
    "Panel",
    [
        "Live Auction",
        "Team Panels",
        "Learning Graphs",
        "Season Results",
        "Before vs After",
        "Strategy Insights",
    ],
)

auto_refresh = st.sidebar.checkbox("Auto-refresh (2s)", value=False)
if auto_refresh:
    import time

    time.sleep(2)
    st.rerun()

# PANEL 1: LIVE AUCTION
if panel == "Live Auction":
    st.title("Live Auction Feed")
    auction_log = load_json("training/logs/auction_log.json", [])
    lot_events = _extract_lot_events(auction_log if isinstance(auction_log, list) else [])
    latest_team_metrics = _latest_team_metrics_from_rewards()
    if not lot_events:
        st.warning("No auction data yet. Run: python training/train.py")
    else:
        last = lot_events[-1]
        col1, col2 = st.columns([4, 6])
        with col1:
            st.metric("Player", last.get("player_name", "--"))
            st.metric("Current Bid", f"Rs.{last.get('price', 0):.1f} Cr")
            st.metric("Leader", _team_label_from_winner(last.get("winner", 0)))
            if last.get("bluff"):
                st.warning("Bluff detected!")
        with col2:
            squads = load_json("training/logs/squads.json", {})
            budgets = {}
            y_label = "Budget Remaining (Cr)"
            for i, t in enumerate(TEAM_NAMES):
                team_data = squads.get(str(i), {})
                if isinstance(team_data, dict):
                    budgets[t] = float(team_data.get("budget_remaining", 90))
                elif t in latest_team_metrics:
                    # Fallback: when only reward logs are available, show end-of-episode unspent budget.
                    budgets[t] = float(latest_team_metrics[t].get("budget_wasted_cr", 0.0))
                    y_label = "Unspent Budget (Cr)"
                else:
                    budgets[t] = 90.0
            fig = px.bar(
                x=list(budgets.keys()),
                y=list(budgets.values()),
                color=list(budgets.values()),
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(yaxis_title=y_label)
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Last 20 Lots")
        for lot in reversed(lot_events[-20:]):
            w = _team_label_from_winner(lot.get("winner", 0))
            st.write(f"{lot.get('player_name', '?')} -> {w} @ Rs.{lot.get('price', 0):.1f}Cr")

# PANEL 2: TEAM PANELS
elif panel == "Team Panels":
    st.title("IPL Team Panels")
    squads = load_json("training/logs/squads.json", {})
    latest_team_metrics = _latest_team_metrics_from_rewards()
    cols = st.columns(4)
    for i, team in enumerate(TEAM_NAMES):
        with cols[i % 4]:
            data = squads.get(str(i), {})
            players = data.get("players", []) if isinstance(data, dict) else []
            budget = float(data.get("budget_remaining", 90)) if isinstance(data, dict) else 90.0
            st.markdown(f"**{team}**")
            st.caption(f"Budget: Rs.{budget:.1f}Cr  | Squad: {len(players)}")
            if players:
                df = pd.DataFrame(
                    [{"Name": p.get("name", "?"), "Role": p.get("role", "?"), "Paid": p.get("price_paid", 0)} for p in players[:8]]
                )
                st.dataframe(df, hide_index=True, use_container_width=True)
            elif team in latest_team_metrics:
                m = latest_team_metrics[team]
                st.caption(
                    f"Latest Total Reward: {m['total_reward']:.2f} | "
                    f"Final Pos: {m['final_position']} | "
                    f"Unspent: {m['budget_wasted_cr']:.1f}Cr"
                )
            else:
                st.info("No squad data yet.")

# PANEL 3: LEARNING GRAPHS
elif panel == "Learning Graphs":
    st.title("Learning Graphs")
    curve = load_json("training/logs/reward_curve.json", {})
    if not curve.get("episodes"):
        st.info("No training data yet. Run training/train.py to populate.")
    else:
        episodes = curve["episodes"]
        teams_data = curve.get("teams", {})
        fig1 = go.Figure()
        for team, metrics in teams_data.items():
            fig1.add_trace(
                go.Scatter(
                    x=episodes,
                    y=metrics.get("rewards", []),
                    name=team,
                    line=dict(color=TEAM_COLORS.get(team, "gray")),
                )
            )
        fig1.update_layout(title="Reward per Episode", xaxis_title="Episode")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        for team, metrics in teams_data.items():
            fig2.add_trace(
                go.Scatter(
                    x=episodes,
                    y=metrics.get("win_rate", []),
                    name=team,
                    line=dict(color=TEAM_COLORS.get(team, "gray")),
                )
            )
        fig2.update_layout(title="Win Rate vs Episodes", xaxis_title="Episode")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = go.Figure()
        for team, metrics in teams_data.items():
            fig3.add_trace(
                go.Scatter(
                    x=episodes,
                    y=metrics.get("budget_efficiency", []),
                    name=team,
                    line=dict(color=TEAM_COLORS.get(team, "gray")),
                )
            )
        fig3.update_layout(title="Budget Efficiency (Rolling) vs Episodes", xaxis_title="Episode")
        st.plotly_chart(fig3, use_container_width=True)

        if os.path.exists("training/logs/reward_curve.png"):
            st.image("training/logs/reward_curve.png", caption="Reward curve image")
        elif os.path.exists("training/logs/comparison_curve.png"):
            st.image("training/logs/comparison_curve.png", caption="Reward curve image")

        from training.reward_logger import RewardLogger

        proof = RewardLogger().get_learning_proof()
        c1, c2, c3 = st.columns(3)
        c1.metric("reward_improvement_pct", f"{proof.get('reward_improvement_pct', 0):.2f}%")
        c2.metric("win_rate_improvement_pct", f"{proof.get('win_rate_improvement_pct', 0):.2f}%")
        c3.metric("budget_efficiency_improvement_pct", f"{proof.get('budget_efficiency_improvement_pct', 0):.2f}%")

        rewards_df = _load_rewards_df()
        agg_df = _build_team_aggregates(rewards_df)
        if not agg_df.empty:
            st.subheader("Champion Distribution (from episodes)")
            fig4 = px.bar(agg_df, x="team", y="champion_count", color="team")
            st.plotly_chart(fig4, use_container_width=True)
            st.subheader("Average Final Rank / Budget Wasted")
            st.dataframe(
                agg_df.sort_values("avg_final_rank", ascending=True),
                use_container_width=True,
                hide_index=True,
            )

elif panel == "Season Results":
    st.title("Season Standings")
    season = load_json("training/logs/season_results.json", {})
    if not season.get("standings"):
        st.info("Season not yet simulated. Run at least 1 training episode.")
    else:
        standings = season["standings"]
        rows = []
        for team_id, stats in standings.items():
            tid = int(team_id) if str(team_id).isdigit() else TEAM_NAMES.index(team_id) if team_id in TEAM_NAMES else 0
            rows.append(
                {
                    "Team": TEAM_NAMES[tid],
                    "Wins": stats.get("wins", 0),
                    "Losses": stats.get("losses", 0),
                    "NRR": round(stats.get("nrr", 0), 3),
                }
            )
        df = pd.DataFrame(rows).sort_values("Wins", ascending=False)
        champion = season.get("champion")
        if champion is not None:
            champ_name = TEAM_NAMES[int(champion)] if str(champion).isdigit() else str(champion)
            st.success(f"Champion: {champ_name}")
        st.dataframe(df, use_container_width=True, hide_index=True)
        bracket = season.get("bracket", {})
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Q1**")
            st.write(bracket.get("q1", bracket.get("Q1", "TBD")))
        with c2:
            st.write("**Eliminator + Q2**")
            elim = bracket.get("eliminator", bracket.get("Eliminator", "TBD"))
            q2 = bracket.get("q2", bracket.get("Q2", "TBD"))
            st.write(f"{elim}\n\n{q2}")
        with c3:
            st.write("**FINAL**")
            st.write(bracket.get("final", bracket.get("Final", "TBD")))

# PANEL 5: BEFORE vs AFTER
elif panel == "Before vs After":
    st.title("Before vs After Training")
    behaviors = load_json("training/logs/behavior_summaries.json", [])
    if isinstance(behaviors, dict):
        # Support current logger format where latest summary may be a dict.
        behaviors = [behaviors]
    if len(behaviors) < 20:
        st.info("Need at least 20 training episodes.")
    else:
        n = len(behaviors)
        slider = st.slider("Compare episodes:", 10, n, (10, n), step=10)
        early = behaviors[: slider[0]]
        late = behaviors[slider[1] - 10 : slider[1]]

        def avg_metric(eps, metric):
            vals = [ep.get(str(i), {}).get(metric, 0) for ep in eps for i in range(8)]
            return sum(vals) / len(vals) if vals else 0

        metrics = ["overbid_rate", "block_rate", "patience_score", "bluff_success_rate"]
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"EARLY (ep 1-{slider[0]})")
            for m in metrics:
                st.metric(m.replace("_", " ").title(), f"{avg_metric(early, m):.2%}")
        with c2:
            st.subheader(f"LATE (ep {slider[1]-10}-{slider[1]})")
            for m in metrics:
                ev, lv = avg_metric(early, m), avg_metric(late, m)
                st.metric(m.replace("_", " ").title(), f"{lv:.2%}", delta=f"{lv-ev:+.2%}")

# PANEL 6: STRATEGY INSIGHTS
elif panel == "Strategy Insights":
    st.title("Strategy Insights")
    insights = load_json("training/logs/emergent_insights.json", [])
    behaviors = load_json("training/logs/behavior_summaries.json", [])
    if isinstance(behaviors, dict):
        behaviors = [behaviors]

    if isinstance(insights, dict):
        insights_list = [f"{k}: {v}" for k, v in insights.items()]
    else:
        insights_list = insights

    if not insights_list:
        st.info("No insights yet. Need 25+ training episodes.")
    else:
        st.subheader("What the Agents Learned")
        for insight in insights_list:
            st.write(f"• {insight}")

    behavior_summary = _build_behavior_metrics(behaviors)
    if behavior_summary:
        st.subheader("Behavior Metrics (Emergent)")
        rows = []
        for team in TEAM_NAMES:
            b = behavior_summary.get(team, {})
            rows.append(
                {
                    "Team": team,
                    "Overbid rate (panic)": round(100 * b.get("overbid_rate", 0.0), 2),
                    "Block rate": round(100 * b.get("block_rate", 0.0), 2),
                    "Patience score": round(b.get("patience_score", 0.0), 4),
                    "Bluff success rate": round(100 * b.get("bluff_success_rate", 0.0), 2),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if behaviors:
        st.subheader("Agent Strategy Evolution")
        for tid, team in enumerate(TEAM_NAMES):
            labels = [ep.get(str(tid), {}).get("label", "?") for ep in behaviors[::10]]
            st.write(f"**{team}**: " + " → ".join(labels[-8:]))
    with st.sidebar:
        st.markdown("---")
        st.subheader("Recent Lots")
        auction_log = load_json("training/logs/auction_log.json", [])
        for lot in auction_log[-10:]:
            st.write(f"{lot.get('player_name', '?')} @ Rs.{lot.get('price', 0):.1f}Cr")
