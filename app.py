import gradio as gr
import json
import os
import argparse
import csv
from collections import defaultdict
import plotly.graph_objects as go
import pandas as pd

TEAM_NAMES = ["MI", "CSK", "RCB", "KKR", "DC", "RR", "PBKS", "SRH"]
PERSONALITIES = [
    "aggressive",
    "conservative",
    "aggressive",
    "balanced",
    "role_filler",
    "conservative",
    "balanced",
    "role_filler",
]


def _team_label(raw):
    raw_str = str(raw)
    if raw_str in TEAM_NAMES:
        return raw_str
    if raw_str.isdigit():
        idx = int(raw_str)
        if 0 <= idx < len(TEAM_NAMES):
            return TEAM_NAMES[idx]
    return raw_str


def _format_rosters_rows(squads):
    rows = []
    for team in TEAM_NAMES:
        squad = squads.get(team, [])
        if not squad:
            # Some env variants store by numeric index.
            idx = TEAM_NAMES.index(team)
            squad = squads.get(idx, squads.get(str(idx), []))
        player_names = ", ".join([p.get("name", "?") for p in squad]) if isinstance(squad, list) else ""
        rows.append([team, player_names])
    return rows


def _format_season_text(season_data):
    if not isinstance(season_data, dict) or not season_data.get("standings"):
        return "Season simulation not available for this run."
    lines = []
    champion = season_data.get("champion", "--")
    lines.append(f"Champion: {_team_label(champion)}")
    lines.append("")
    lines.append("Standings:")
    standings = season_data.get("standings", {})
    ordered = sorted(
        standings.items(),
        key=lambda x: x[1].get("rank", 99) if isinstance(x[1], dict) else 99,
    )
    for team_id, stats in ordered:
        team = _team_label(team_id)
        if not isinstance(stats, dict):
            continue
        lines.append(
            f"- {team}: W {stats.get('wins', 0)} | L {stats.get('losses', 0)} | NRR {float(stats.get('nrr', 0.0)):.3f}"
        )
    return "\n".join(lines)


def _format_transfer_text(transfer_log):
    players_file = os.path.join(os.path.dirname(__file__), "data", "players.json")
    id_to_name = {}
    try:
        if os.path.exists(players_file):
            with open(players_file, "r", encoding="utf-8") as f:
                players = json.load(f)
            id_to_name = {int(p.get("id")): p.get("name", f"Player#{p.get('id')}") for p in players}
    except Exception:
        id_to_name = {}

    def _name_from_id(pid):
        try:
            return id_to_name.get(int(pid), f"Player#{pid}")
        except Exception:
            return f"Player#{pid}"

    if not transfer_log:
        return "No transfer activity recorded."
    lines = ["Recent Transfers:"]
    for t in transfer_log[-20:]:
        if isinstance(t, dict):
            from_team = _team_label(t.get("from_team", t.get("from", "?")))
            to_team = _team_label(t.get("to_team", t.get("to", "?")))
            # Transfer log uses player IDs; resolve to names for readable Phase 3 output.
            give_id = t.get("give_player_id")
            want_id = t.get("want_player_id")
            if give_id is not None and want_id is not None:
                give_name = _name_from_id(give_id)
                want_name = _name_from_id(want_id)
                status = "ACCEPTED" if t.get("accepted", False) else "REJECTED"
                lines.append(
                    f"- {status}: {from_team} offered {give_name} for {want_name} from {to_team}"
                )
            else:
                reason = str(t.get("reason", "No player-level details"))
                status = "ACCEPTED" if t.get("accepted", False) else "REJECTED"
                lines.append(f"- {status}: {from_team} -> {to_team} ({reason})")
        else:
            lines.append(f"- {str(t)}")
    return "\n".join(lines)


def _extract_auction_events_from_env(env):
    logs = []
    if getattr(env, "auction_engine", None) is None:
        return logs
    for ev in getattr(env.auction_engine, "auction_log", []):
        if not isinstance(ev, dict):
            continue
        if ev.get("event") != "sold":
            continue
        player_id = ev.get("player_id")
        player_name = f"Player #{player_id}"
        try:
            if player_id is not None:
                player = env._player_from_id(int(player_id))
                player_name = player.get("name", player_name)
        except Exception:
            pass
        team = _team_label(ev.get("team_id", "?"))
        price = float(ev.get("price", 0.0) or 0.0)
        logs.append(f"{player_name} -> {team} @ Rs.{price:.1f}Cr")
    return logs


def _build_transfer_actions(env):
    actions = {}
    if getattr(env, "transfer_market", None) is None:
        return {team: ("skip", None) for team in TEAM_NAMES}

    for idx, team_id in enumerate(TEAM_NAMES):
        obs = env.transfer_market.get_transfer_observation(team_id)
        trades_remaining = int(obs.get("trades_remaining", 0))
        weak_players = obs.get("own_weak_players", [])
        if trades_remaining <= 0 or not weak_players:
            actions[team_id] = ("skip", None)
            continue

        # Target next team in ring that has at least one player.
        target_team = None
        target_player = None
        for j in range(1, len(TEAM_NAMES)):
            cand_team = TEAM_NAMES[(idx + j) % len(TEAM_NAMES)]
            cand_squad = env.team_squads.get(cand_team, [])
            if cand_squad:
                target_team = cand_team
                target_player = cand_squad[0]
                break

        if target_team is None or target_player is None:
            actions[team_id] = ("skip", None)
            continue

        give_player_id = int(weak_players[0].get("id"))
        want_player_id = int(target_player.get("id"))
        actions[team_id] = (
            "trade",
            {
                "to_team": target_team,
                "give_player_id": give_player_id,
                "want_player_id": want_player_id,
                "cash": 0.0,
            },
        )
    return actions


def run_full_simulation_cycle(fast_mode=False):
    # Lazy import to keep startup fast on Colab.
    from env.ipl_env import IPLAuctionEnv
    from agents.base_agent import BaseIPLAgent

    env = IPLAuctionEnv()
    agents = {}
    for i in range(8):
        team_id = TEAM_NAMES[i]
        agent = BaseIPLAgent(team_id, PERSONALITIES[i])
        agent.team_name = TEAM_NAMES[i]
        agents[team_id] = agent

    import time

    obs = env.reset()
    done = False
    last_info = {}
    is_colab = "COLAB_RELEASE_TAG" in os.environ
    # Full mode gives correct phase outputs; fast mode is optional.
    max_steps = 90 if (is_colab and fast_mode) else 10000
    max_seconds = 12 if (is_colab and fast_mode) else 120
    start_time = time.time()
    steps = 0

    while not done:
        phase = obs.get(TEAM_NAMES[0], {}).get("phase", "auction")
        actions = {}
        if phase == "auction":
            for team_id in TEAM_NAMES:
                decision = agents[team_id].decide_bid(obs.get(team_id, {}))
                if decision.get("action") == "bid":
                    actions[team_id] = ("bid", decision.get("amount", 0.5), decision.get("bluff", False))
                else:
                    actions[team_id] = ("pass", None)
        elif phase == "transfer":
            actions = _build_transfer_actions(env)
        else:
            # Season phase does not require explicit actions in this environment.
            actions = {team_id: ("skip", None) for team_id in TEAM_NAMES}

        obs, rewards, done, info = env.step(actions)
        del rewards
        last_info = info
        steps += 1
        if steps >= max_steps or (time.time() - start_time) >= max_seconds:
            break

    squads = last_info.get("final_squads", env.team_squads)
    roster_rows = _format_rosters_rows(squads if isinstance(squads, dict) else {})

    season_data = env.last_season_results if hasattr(env, "last_season_results") else {}
    season_text = _format_season_text(season_data)

    transfer_log = []
    if getattr(env, "transfer_market", None) is not None:
        transfer_log = getattr(env.transfer_market, "trade_log", []) or []
    transfer_text = _format_transfer_text(transfer_log)

    metrics_lines = ["Team Reward Snapshot:"]
    for team in TEAM_NAMES:
        try:
            total = float(env.compute_reward(team))
            metrics_lines.append(f"- {team}: {total:+.2f}")
        except Exception:
            metrics_lines.append(f"- {team}: N/A")
    metrics_text = "\n".join(metrics_lines)

    log = _extract_auction_events_from_env(env)
    auction_text = "\n".join(log[-30:]) if log else "No sold lots captured."
    if not done:
        auction_text = (
            f"Simulation stopped early ({steps} steps in {time.time() - start_time:.1f}s).\n"
            f"Enable full run by turning off Fast Mode.\n\n"
            + auction_text
        )
    return auction_text, roster_rows, season_text, transfer_text, metrics_text


def load_results():
    try:
        if not os.path.exists("training/logs/reward_curve.json"):
            return "No training data yet. Run train.py first."
        with open("training/logs/reward_curve.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        teams = data.get("teams", {})
        lines = []
        for team, metrics in teams.items():
            rewards = metrics.get("rewards", [])
            if rewards:
                window = rewards[-10:] if len(rewards) >= 10 else rewards
                avg = sum(window) / max(1, len(window))
                lines.append(f"{team}: Avg Reward (last 10 eps) = {avg:.1f}")
        return "\n".join(lines) or "No training data yet. Run train.py first."
    except Exception:
        return "No training data yet. Run train.py first."


APP_CSS = """
body { background: #070f25 !important; }
.gradio-container { background: #070f25 !important; color: #e6ebff !important; }
.panel { border: 1px solid #2b3768; border-radius: 10px; }
"""


def _safe_load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _build_line_fig(title, episodes, teams_payload, key):
    fig = go.Figure()
    for team in TEAM_NAMES:
        metrics = teams_payload.get(team, {})
        y = metrics.get(key, [])
        if y:
            fig.add_trace(go.Scatter(x=episodes[: len(y)], y=y, mode="lines", name=team))
    fig.update_layout(title=title, xaxis_title="Episode", yaxis_title=key.replace("_", " ").title(), template="plotly_dark")
    return fig


def _load_analytics():
    # Build accurate curves directly from rewards.csv (episode-level means), not pre-aggregated json.
    rewards_path = "training/logs/rewards.csv"
    teams_payload = {t: {"episodes": [], "rewards": [], "win_rate": [], "budget_efficiency": []} for t in TEAM_NAMES}

    if os.path.exists(rewards_path):
        df = pd.read_csv(rewards_path)
        if not df.empty:
            df["team_id"] = df["team_id"].astype(str).map(_team_label)
            df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
            df["TOTAL"] = pd.to_numeric(df.get("TOTAL"), errors="coerce")
            df["final_position"] = pd.to_numeric(df.get("final_position"), errors="coerce")
            df["budget_wasted_cr"] = pd.to_numeric(df.get("budget_wasted_cr"), errors="coerce")
            df = df.dropna(subset=["episode", "team_id"])
            df = df[df["team_id"].isin(TEAM_NAMES)]
            df["episode"] = df["episode"].astype(int)

            grouped = (
                df.groupby(["team_id", "episode"], as_index=False)
                .agg(
                    reward=("TOTAL", "mean"),
                    win=("final_position", lambda x: float((x <= 4).mean())),
                    budget_eff=("budget_wasted_cr", lambda x: float((1.0 - (x / 90.0)).clip(lower=0.0, upper=1.0).mean())),
                )
                .sort_values(["team_id", "episode"])
            )

            for team in TEAM_NAMES:
                tdf = grouped[grouped["team_id"] == team].copy()
                if tdf.empty:
                    continue
                tdf["reward_roll"] = tdf["reward"].rolling(window=10, min_periods=1).mean()
                tdf["win_roll"] = tdf["win"].rolling(window=10, min_periods=1).mean()
                tdf["budget_roll"] = tdf["budget_eff"].rolling(window=10, min_periods=1).mean()
                teams_payload[team] = {
                    "episodes": tdf["episode"].tolist(),
                    "rewards": tdf["reward_roll"].tolist(),
                    "win_rate": tdf["win_roll"].tolist(),
                    "budget_efficiency": tdf["budget_roll"].tolist(),
                }

    # Build figures with per-team episode axes.
    reward_fig = go.Figure()
    win_fig = go.Figure()
    budget_fig = go.Figure()
    for team in TEAM_NAMES:
        data = teams_payload.get(team, {})
        x = data.get("episodes", [])
        if not x:
            continue
        reward_fig.add_trace(go.Scatter(x=x, y=data.get("rewards", []), mode="lines", name=team))
        win_fig.add_trace(go.Scatter(x=x, y=data.get("win_rate", []), mode="lines", name=team))
        budget_fig.add_trace(go.Scatter(x=x, y=data.get("budget_efficiency", []), mode="lines", name=team))

    reward_fig.update_layout(title="Reward Curve per Team", xaxis_title="Episode", yaxis_title="Rewards", template="plotly_dark")
    win_fig.update_layout(title="Win-rate Curve (Rolling) per Team", xaxis_title="Episode", yaxis_title="Win Rate", template="plotly_dark")
    budget_fig.update_layout(title="Budget-efficiency Curve (Rolling) per Team", xaxis_title="Episode", yaxis_title="Budget Efficiency", template="plotly_dark")

    # Headline metrics computed from first 10 vs last 10 rolling points.
    def _pct(first, last):
        if abs(first) < 1e-9:
            return 0.0 if abs(last) < 1e-9 else 100.0
        return ((last - first) / abs(first)) * 100.0

    reward_impr, win_impr, budget_impr = [], [], []
    for team in TEAM_NAMES:
        d = teams_payload.get(team, {})
        r = d.get("rewards", [])
        w = d.get("win_rate", [])
        b = d.get("budget_efficiency", [])
        if len(r) >= 20 and len(w) >= 20 and len(b) >= 20:
            reward_impr.append(_pct(sum(r[:10]) / 10.0, sum(r[-10:]) / 10.0))
            win_impr.append(_pct(sum(w[:10]) / 10.0, sum(w[-10:]) / 10.0))
            budget_impr.append(_pct(sum(b[:10]) / 10.0, sum(b[-10:]) / 10.0))

    headline = (
        f"reward_improvement_pct: {sum(reward_impr)/max(1,len(reward_impr)):.2f}%\n"
        f"win_rate_improvement_pct: {sum(win_impr)/max(1,len(win_impr)):.2f}%\n"
        f"budget_efficiency_improvement_pct: {sum(budget_impr)/max(1,len(budget_impr)):.2f}%"
    )

    # Aggregates from rewards.csv
    champion_counts = defaultdict(int)
    rank_sum = defaultdict(float)
    budget_sum = defaultdict(float)
    rows_count = defaultdict(int)
    try:
        with open("training/logs/rewards.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                team = str(row.get("team_id", "")).strip()
                if team not in TEAM_NAMES:
                    continue
                final_pos = float(row.get("final_position", 8) or 8)
                budget_wasted = float(row.get("budget_wasted_cr", 0) or 0)
                if int(final_pos) == 1:
                    champion_counts[team] += 1
                rank_sum[team] += final_pos
                budget_sum[team] += budget_wasted
                rows_count[team] += 1
    except Exception:
        pass

    aggregate_lines = ["Champion distribution / avg final rank / avg budget wasted"]
    for team in TEAM_NAMES:
        n = max(1, rows_count[team])
        aggregate_lines.append(
            f"- {team}: champions={champion_counts[team]}, avg_rank={rank_sum[team]/n:.2f}, avg_budget_wasted={budget_sum[team]/n:.2f} Cr"
        )
    aggregate_text = "\n".join(aggregate_lines)

    # Behavior insights
    behaviors = _safe_load_json("training/logs/behavior_summaries.json", [])
    if isinstance(behaviors, dict):
        behaviors = [behaviors]
    metrics_acc = {t: {"overbid_rate": [], "block_rate": [], "patience_score": [], "bluff_success_rate": [], "labels": []} for t in TEAM_NAMES}
    for ep in behaviors:
        if not isinstance(ep, dict):
            continue
        for i, team in enumerate(TEAM_NAMES):
            b = ep.get(str(i), {})
            if not isinstance(b, dict):
                continue
            for key in ["overbid_rate", "block_rate", "patience_score", "bluff_success_rate"]:
                metrics_acc[team][key].append(float(b.get(key, 0.0) or 0.0))
            metrics_acc[team]["labels"].append(str(b.get("label", "?")))

    behavior_lines = ["Behavior insights (emergent)"]
    label_lines = ["Strategy label evolution"]
    for team in TEAM_NAMES:
        vals = metrics_acc[team]
        n = max(1, len(vals["overbid_rate"]))
        behavior_lines.append(
            f"- {team}: overbid={100*sum(vals['overbid_rate'])/n:.2f}% | "
            f"block={100*sum(vals['block_rate'])/n:.2f}% | "
            f"patience={sum(vals['patience_score'])/n:.3f} | "
            f"bluff_success={100*sum(vals['bluff_success_rate'])/n:.2f}%"
        )
        labels = [x for x in vals["labels"] if x and x != "?"]
        if labels:
            short = labels[:: max(1, len(labels) // 5)]
            label_lines.append(f"- {team}: " + " -> ".join(short[:6]))
        else:
            label_lines.append(f"- {team}: no labels yet")

    # Reward curve image path fallback
    curve_img = None
    if os.path.exists("training/logs/reward_curve.png"):
        curve_img = "training/logs/reward_curve.png"
    elif os.path.exists("training/logs/comparison_curve.png"):
        curve_img = "training/logs/comparison_curve.png"

    return (
        reward_fig,
        win_fig,
        budget_fig,
        headline,
        aggregate_text,
        "\n".join(behavior_lines),
        "\n".join(label_lines),
        curve_img,
    )


with gr.Blocks(title="IPL RL Auction") as demo:
    gr.Markdown(
        """
        # IPL Multi-Agent RL Auction Environment
        Teaching 8 AI agents to draft, manage, and optimize championship-winning squads across 3 phases.
        """
    )

    run_btn = gr.Button("Run Full Simulation Cycle", variant="primary")
    fast_mode = gr.Checkbox(label="Fast Mode (quicker, may show partial phases)", value=False)

    with gr.Tabs():
        with gr.Tab("Phase 1: Auction"):
            with gr.Row():
                auction_log = gr.Textbox(
                    label="Live Auction Bidding (Last 30 Events)",
                    lines=16,
                    elem_classes=["panel"],
                )
                rosters_out = gr.Dataframe(
                    headers=["Team", "Final Rosters (All Players)"],
                    datatype=["str", "str"],
                    row_count=(8, "fixed"),
                    col_count=(2, "fixed"),
                    interactive=False,
                    wrap=True,
                )
        with gr.Tab("Phase 2: Season results"):
            season_out = gr.Textbox(label="Season Summary", lines=18, elem_classes=["panel"])
        with gr.Tab("Phase 3: Transfer Window"):
            transfer_out = gr.Textbox(label="Transfer Activity", lines=18, elem_classes=["panel"])
        with gr.Tab("Training Metrics"):
            metrics_out = gr.Textbox(label="Simulation Metrics", lines=14, elem_classes=["panel"])
            results_out = gr.Textbox(label="Logged Training Results", lines=8)
            reward_fig = gr.Plot(label="Reward curve per team across episodes")
            win_fig = gr.Plot(label="Win-rate curve (rolling) per team")
            budget_fig = gr.Plot(label="Budget-efficiency curve (rolling) per team")
            learning_headline = gr.Textbox(label="Learning-proof headline metrics", lines=4)
            team_aggregates = gr.Textbox(label="Champion/rank/budget aggregates", lines=10)
            behavior_insights = gr.Textbox(label="Behavior insights", lines=10)
            strategy_evolution = gr.Textbox(label="Strategy label evolution", lines=10)
            reward_image = gr.Image(label="Reward Curve Image", type="filepath")
            gr.Button("Load Analytics").click(
                fn=_load_analytics,
                outputs=[
                    reward_fig,
                    win_fig,
                    budget_fig,
                    learning_headline,
                    team_aggregates,
                    behavior_insights,
                    strategy_evolution,
                    reward_image,
                ],
            )
            gr.Button("Load Training Logs").click(fn=load_results, outputs=results_out)
        with gr.Tab("About"):
            gr.Markdown("Multi-agent RL for IPL team building with auction, season, and transfer phases.")

    run_btn.click(
        fn=run_full_simulation_cycle,
        inputs=[fast_mode],
        outputs=[auction_log, rosters_out, season_out, transfer_out, metrics_out],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch IPL RL Gradio app.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    parser.add_argument("--port", type=int, default=7860, help="Port to launch the app on.")
    args = parser.parse_args()

    is_colab = "COLAB_RELEASE_TAG" in os.environ
    share_enabled = args.share or is_colab
    # Colab sessions often have occupied ports from old runs; retry nearby ports.
    for candidate_port in range(args.port, args.port + 20):
        try:
            demo.launch(
                share=share_enabled,
                server_name="0.0.0.0",
                server_port=candidate_port,
            )
            break
        except OSError:
            if candidate_port == args.port + 19:
                raise
