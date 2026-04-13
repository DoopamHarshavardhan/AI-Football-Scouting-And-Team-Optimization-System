import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, FancyArrowPatch
import os
import groq
from dotenv import load_dotenv
from groq import Groq
import markdown

load_dotenv()  # loads .env file

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# HELPER FUNCTIONS

def expand_position(pos):
    mapping = {
        "ST": "Striker", "CF": "Centre Forward",
        "LW": "Left Winger", "RW": "Right Winger",
        "LM": "Left Midfielder", "RM": "Right Midfielder",
        "CAM": "Attacking Midfielder", "CM": "Central Midfielder",
        "CDM": "Defensive Midfielder", "CB": "Centre Back",
        "LB": "Left Back", "RB": "Right Back",
        "LWB": "Left Wing Back", "RWB": "Right Wing Back",
        "GK": "Goalkeeper"
    }
    if pd.isna(pos):
        return pos
    pos = str(pos).replace("/", ",")
    positions = [p.strip() for p in pos.split(",")]
    expanded = [mapping.get(p, p) for p in positions]
    return ", ".join(expanded)


def position_coordinates_image(pos):
    mapping = {
        "GK":  (8,  50),   "CB":  (22, 50),  "LB":  (22, 21),
        "RB":  (22, 76.5), "LWB": (40, 32),  "RWB": (40, 68),
        "CDM": (36, 50),   "CM":  (48, 50),  "LM":  (48, 21),
        "RM":  (48, 76.5), "CAM": (61, 50),  "LW":  (75, 27),
        "RW":  (75, 71),   "CF":  (70, 50),  "ST":  (61, 50)
    }
    return mapping.get(pos, (50, 50))


def get_player_skills(player):
    pace      = (player["acceleration"] + player["sprint_speed"]) / 2
    shooting  = (player["finishing"] + player["shot_power"] + player["long_shots"]) / 3
    passing   = (player["short_passing"] + player["long_passing"] + player["vision"]) / 3
    dribbling = (player["dribbling"] + player["ball_control"] + player["agility"]) / 3
    defending = (player["standing_tackle"] + player["sliding_tackle"] + player["interceptions"]) / 3
    physical  = (player["strength"] + player["stamina"] + player["balance"]) / 3
    return {
        "Pace": round(pace, 1), "Shooting": round(shooting, 1),
        "Passing": round(passing, 1), "Dribbling": round(dribbling, 1),
        "Defending": round(defending, 1), "Physical": round(physical, 1)
    }


def build_similarity_features(player):
    return pd.DataFrame([{
        "Pace":      (player["acceleration"] + player["sprint_speed"]) / 2,
        "Shooting":  (player["finishing"] + player["shot_power"] + player["long_shots"]) / 3,
        "Passing":   (player["short_passing"] + player["long_passing"] + player["vision"]) / 3,
        "Dribbling": (player["dribbling"] + player["ball_control"] + player["agility"]) / 3,
        "Defending": (player["standing_tackle"] + player["sliding_tackle"] + player["interceptions"]) / 3,
        "Physical":  (player["strength"] + player["stamina"] + player["balance"]) / 3
    }])


def build_player_features(player, feature_list):
    data = {}
    for feature in feature_list:
        if feature == "growth_gap":
            data[feature] = player["potential"] - player["overall_rating"]
        elif feature == "age_penalty":
            data[feature] = max(0, player["age"] - 27)
        else:
            data[feature] = player[feature]
    return pd.DataFrame([data])


def build_archetype_features(player):
    return pd.DataFrame([{
        "Pace":      (player["acceleration"] + player["sprint_speed"]) / 2,
        "Shooting":  (player["finishing"] + player["shot_power"] + player["long_shots"]) / 3,
        "Passing":   (player["short_passing"] + player["long_passing"] + player["vision"]) / 3,
        "Dribbling": (player["dribbling"] + player["ball_control"] + player["agility"]) / 3,
        "Defending": (player["standing_tackle"] + player["sliding_tackle"] + player["interceptions"]) / 3,
        "Physical":  (player["strength"] + player["stamina"] + player["balance"]) / 3
    }])


def get_prediction_confidence(model, X_row):
    tree_preds = np.array([tree.predict(X_row)[0] for tree in model.estimators_])
    std_dev = np.std(tree_preds)
    if std_dev < 1.5:
        return "High"
    elif std_dev < 3.0:
        return "Medium"
    else:
        return "Low"

# TACTICAL FIT  (Tab 2)

POSITION_DEMANDS = {
    "GK":  {"Pace": 10, "Shooting": 5,  "Passing": 30, "Dribbling": 10, "Defending": 80, "Physical": 60},
    "CB":  {"Pace": 40, "Shooting": 10, "Passing": 40, "Dribbling": 20, "Defending": 90, "Physical": 80},
    "LB":  {"Pace": 75, "Shooting": 20, "Passing": 55, "Dribbling": 55, "Defending": 70, "Physical": 60},
    "RB":  {"Pace": 75, "Shooting": 20, "Passing": 55, "Dribbling": 55, "Defending": 70, "Physical": 60},
    "LWB": {"Pace": 80, "Shooting": 25, "Passing": 60, "Dribbling": 65, "Defending": 60, "Physical": 60},
    "RWB": {"Pace": 80, "Shooting": 25, "Passing": 60, "Dribbling": 65, "Defending": 60, "Physical": 60},
    "CDM": {"Pace": 45, "Shooting": 30, "Passing": 65, "Dribbling": 50, "Defending": 80, "Physical": 75},
    "CM":  {"Pace": 55, "Shooting": 50, "Passing": 80, "Dribbling": 65, "Defending": 55, "Physical": 60},
    "LM":  {"Pace": 70, "Shooting": 55, "Passing": 65, "Dribbling": 70, "Defending": 40, "Physical": 55},
    "RM":  {"Pace": 70, "Shooting": 55, "Passing": 65, "Dribbling": 70, "Defending": 40, "Physical": 55},
    "CAM": {"Pace": 60, "Shooting": 65, "Passing": 80, "Dribbling": 75, "Defending": 25, "Physical": 45},
    "LW":  {"Pace": 85, "Shooting": 65, "Passing": 60, "Dribbling": 85, "Defending": 25, "Physical": 45},
    "RW":  {"Pace": 85, "Shooting": 65, "Passing": 60, "Dribbling": 85, "Defending": 25, "Physical": 45},
    "CF":  {"Pace": 70, "Shooting": 80, "Passing": 65, "Dribbling": 75, "Defending": 20, "Physical": 55},
    "ST":  {"Pace": 75, "Shooting": 90, "Passing": 50, "Dribbling": 65, "Defending": 20, "Physical": 65},
}


def compute_position_fit(player_skills: dict, position: str) -> float:
    demands      = POSITION_DEMANDS[position]
    total_weight = sum(demands.values())
    score = sum(min(player_skills.get(s, 0) / 100, 1.0) * d for s, d in demands.items())
    return round((score / total_weight) * 100, 1)


def fit_colour(score: float):
    if score >= 72:
        return "#2ecc71"
    elif score >= 52:
        return "#f1c40f"
    else:
        return "#e74c3c"


def fit_label(score: float):
    if score >= 72:
        return "Good fit"
    elif score >= 52:
        return "Playable"
    else:
        return "Poor fit"


def top_missing_skills(player_skills: dict, position: str) -> str:
    demands = POSITION_DEMANDS[position]
    gaps = {s: round(d - player_skills.get(s, 0), 1)
            for s, d in demands.items() if d >= 50 and d > player_skills.get(s, 0)}
    top2 = sorted(gaps, key=gaps.get, reverse=True)[:2]
    if not top2:
        return "Meets all key demands"
    return "Needs: " + ", ".join([f"{s} (+{gaps[s]:.0f})" for s in top2])

# REALISTIC PITCH DRAWING


def draw_realistic_pitch(ax, fig):
    """
    Draw a realistic football pitch with alternating grass stripes,
    proper markings, penalty areas, goal areas, centre circle and arcs.
    """
    # --- Alternating grass stripes (8 vertical bands) ---
    stripe_colors = ["#3a8c3f", "#2e7d32"]
    n_stripes = 8
    stripe_w  = 100 / n_stripes
    for i in range(n_stripes):
        ax.add_patch(plt.Rectangle(
            (i * stripe_w, 0), stripe_w, 100,
            color=stripe_colors[i % 2], zorder=0
        ))

    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#2e7d32")

    lw   = 1.4   # line width
    lc   = "white"

    # --- Outer boundary ---
    ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color=lc, lw=lw, zorder=3)

    # --- Halfway line ---
    ax.axvline(50, color=lc, lw=lw, zorder=3)

    # --- Centre circle ---
    centre_circle = plt.Circle((50, 50), 9.15, fill=False, color=lc, lw=lw, zorder=3)
    ax.add_patch(centre_circle)
    ax.scatter(50, 50, s=18, color=lc, zorder=4)

    # --- Left penalty area ---
    ax.plot([0, 16, 16, 0], [21, 21, 79, 79], color=lc, lw=lw, zorder=3)
    # Left goal area (6-yard box)
    ax.plot([0, 6, 6, 0], [37, 37, 63, 63], color=lc, lw=lw, zorder=3)
    # Left penalty spot
    ax.scatter(11, 50, s=12, color=lc, zorder=4)
    # Left penalty arc
    left_arc = Arc((11, 50), width=18.3, height=18.3,
                   angle=0, theta1=307, theta2=53,
                   color=lc, lw=lw, zorder=3)
    ax.add_patch(left_arc)
    # Left goal
    ax.plot([0, -2, -2, 0], [44, 44, 56, 56], color=lc, lw=lw, zorder=3)

    # --- Right penalty area ---
    ax.plot([100, 84, 84, 100], [21, 21, 79, 79], color=lc, lw=lw, zorder=3)
    # Right goal area (6-yard box)
    ax.plot([100, 94, 94, 100], [37, 37, 63, 63], color=lc, lw=lw, zorder=3)
    # Right penalty spot
    ax.scatter(89, 50, s=12, color=lc, zorder=4)
    # Right penalty arc
    right_arc = Arc((89, 50), width=18.3, height=18.3,
                    angle=0, theta1=127, theta2=233,
                    color=lc, lw=lw, zorder=3)
    ax.add_patch(right_arc)
    # Right goal
    ax.plot([100, 102, 102, 100], [44, 44, 56, 56], color=lc, lw=lw, zorder=3)

    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_aspect("equal")
    ax.axis("off")

# GROQ AI  — works locally (env var) AND on Streamlit Cloud (secrets)

def call_groq(prompt: str) -> str:
    # Use a supported Groq model by default.
    # Valid options include: llama-3.3-70b-versatile, llama-3.1-8b-instant, compound-beta, gemma2-9b-it, openai/gpt-oss-20b, qwen/qwen3-32b, etc.
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
            timeout=30
        )
        choices = getattr(response, "choices", None)
        if choices and len(choices) > 0 and getattr(choices[0], "message", None):
            return choices[0].message.content
        return "⚠️ AI analysis unavailable: received an empty response from Groq."
    except groq.BadRequestError as e:
        body = getattr(e, "body", None)
        details = body if body else str(e)
        return (
            "⚠️ Groq Bad Request: model not found or invalid request. "
            f"Model={model}. {details}"
        )
    except groq.APIError as e:
        status = getattr(e, "status_code", None)
        body = getattr(e, "body", None)
        detail_text = body if body else str(e)
        return f"⚠️ Groq API error{f' (status {status})' if status else ''}: {detail_text}"
    except groq.GroqError as e:
        return f"⚠️ Groq error: {str(e)}"
    except Exception as e:
        return f"⚠️ AI analysis unavailable: {str(e)}"

# AI PROMPTS

def prompt_player_scout(player, skills, archetype, predicted_peak, confidence):
    return f"""You are an elite football scout. Write a concise professional scouting report for this player.

Player: {player['name']}
Age: {int(player['age'])} | Position: {player.get('Position_Full', player['positions'])}
Overall: {int(player['overall_rating'])}/100 | Potential: {int(player['potential'])}/100
Predicted Peak (ML model): {predicted_peak} | Confidence: {confidence}
Archetype: {archetype}
Skills — Pace: {skills['Pace']}, Shooting: {skills['Shooting']}, Passing: {skills['Passing']}, Dribbling: {skills['Dribbling']}, Defending: {skills['Defending']}, Physical: {skills['Physical']}
Attacking Score: {round(player['attacking_score'], 1)} | Midfield Score: {round(player['midfield_score'], 1)} | Defensive Score: {round(player['defensive_score'], 1)}
Growth Index: {round(player['growth_index'], 1)} | Recommended Role: {player['recommended_role']}

Write 4-5 bullet points covering:
1. This player's standout quality and what makes them special
2. Their biggest weakness and what they need to improve
3. Which tactical system and formation suits them best
4. Whether they are worth signing now or one to watch for the future

Use markdown-style bullets (start each item with "-"). Use actual stats to justify every point. Professional scout tone."""


def prompt_compare(player_a, skills_a, archetype_a, player_b, skills_b, archetype_b):
    return f"""You are an elite football scout comparing two players for a club's recruitment team.

**{player_a['name']}**:
Age: {int(player_a['age'])} | Position: {player_a.get('Position_Full', player_a['positions'])}
Overall: {int(player_a['overall_rating'])} | Potential: {int(player_a['potential'])} | Growth Index: {round(player_a['growth_index'], 1)}
Archetype: {archetype_a}
Skills — Pace: {skills_a['Pace']}, Shooting: {skills_a['Shooting']}, Passing: {skills_a['Passing']}, Dribbling: {skills_a['Dribbling']}, Defending: {skills_a['Defending']}, Physical: {skills_a['Physical']}

**{player_b['name']}**:
Age: {int(player_b['age'])} | Position: {player_b.get('Position_Full', player_b['positions'])}
Overall: {int(player_b['overall_rating'])} | Potential: {int(player_b['potential'])} | Growth Index: {round(player_b['growth_index'], 1)}
Archetype: {archetype_b}
Skills — Pace: {skills_b['Pace']}, Shooting: {skills_b['Shooting']}, Passing: {skills_b['Passing']}, Dribbling: {skills_b['Dribbling']}, Defending: {skills_b['Defending']}, Physical: {skills_b['Physical']}

Write a concise 5-6 bullet-point scouting report comparing {player_a['name']} and {player_b['name']}, focusing on key recommendations:

- **Standout Quality**: Highlight each player's key strength with stats
- **Current Performance**: Who performs better now and why
- **Future Potential**: Who has higher ceiling and why
- **Tactical Fit**: Best system for each player
- **Recommendation**: Who to sign and why (be decisive)

Keep each bullet to 1-2 sentences. Use bold names like **{player_a['name']}** and **{player_b['name']}**."""


def prompt_squad(team_data: list):
    players_text = "\n".join([
        f"- {r['role']}: {r['name']} (OVR {r['overall']}, Pos: {r['positions']})"
        for r in team_data
    ])
    return f"""You are a football tactics analyst reviewing a squad for a club manager.

Current XI:
{players_text}

Write a 5-6 bullet-point squad analysis:
1. The squad's biggest strength as a unit
2. The weakest position that urgently needs upgrading and why
3. The recommended formation that suits this squad best
4. The key player who makes this team tick
5. One transfer priority — what type of player to sign next

Use markdown-style bullets (start each item with "-"). Be specific. Reference positions and ratings in your reasoning."""


# PAGE CONFIG

st.set_page_config(
    page_title="AI-Based Football Scouting ",
    page_icon="⚽",
    layout="wide"
)

st.markdown(
    "<h2 style='margin-top:0;'>⚽ AI-Based Football Player Scouting & Team Optimization System</h2>",
    unsafe_allow_html=True
)

st.markdown("""
<style>
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 0.8rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}
div[data-testid="stMetric"] {
    padding: 0.3rem;
}
.ai-box {
    background-color: #ffffff;
    border-left: 4px solid #1a73e8;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    font-size: 15px;
    line-height: 1.75;
    color: #000000;
    margin-top: 0.5rem;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# DATA & MODEL LOADING

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

@st.cache_resource
def load_knn_model():
    knn      = joblib.load("similar_player_knn.joblib")
    scaler   = joblib.load("similarity_scaler.joblib")
    features = joblib.load("similarity_features.joblib")
    return knn, scaler, features

@st.cache_resource
def load_ml_model():
    model    = joblib.load("future_overall_rf_model.joblib")
    features = joblib.load("model_features.joblib")
    return model, features

@st.cache_resource
def load_archetype_model():
    kmeans   = joblib.load("player_archetype_kmeans.joblib")
    scaler   = joblib.load("archetype_scaler.joblib")
    features = joblib.load("archetype_features.joblib")
    return kmeans, scaler, features

knn_model,       knn_scaler,       knn_features       = load_knn_model()
ml_model,        ml_features                          = load_ml_model()
archetype_model, archetype_scaler, archetype_features = load_archetype_model()

for col in ["Position", "Pos", "player_positions", "positions", "Position(s)"]:
    if col in df.columns:
        df["Position_Full"] = df[col].apply(expand_position)
        break
else:
    df["Position_Full"] = "Unknown"

ARCHETYPE_MAP = {
    0: "Goal Poacher", 1: "Playmaker", 2: "Defensive Anchor",
    3: "Pace Winger",  4: "Box-to-Box Midfielder"
}

# SIDEBAR

st.sidebar.header("🔍 Player Selection")
selected_player_name = st.sidebar.selectbox(
    "Select Player", sorted(df["name"].unique())
)
selected_player = df[df["name"] == selected_player_name].iloc[0]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔁 Similar Players")

try:
    sim_input  = build_similarity_features(selected_player)
    sim_scaled = knn_scaler.transform(sim_input[knn_features])
    distances, indices = knn_model.kneighbors(sim_scaled)
    similar_df = df.iloc[indices[0][1:]][["name", "overall_rating"]]
    for _, row in similar_df.iterrows():
        st.sidebar.markdown(f"• **{row['name']}** (OVR {int(row['overall_rating'])})")
except Exception:
    st.sidebar.info("Similar players unavailable")

# TABS

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧑 Player Profile",
    "🎯 Tactical Fit",
    "🆚 Compare Players",
    "🏆 Team Builder",
    "🔍 Scouting Hub"
])

# TAB 1 — PLAYER PROFILE

with tab1:
    player = selected_player

    c1, c2, c3, c4, c5 = st.columns(5)
    overall   = int(player["overall_rating"])
    potential = int(player["potential"])
    growth    = potential - overall
    primary_pos_full = player["Position_Full"].split(",")[0].strip()

    c1.metric("Age",       int(player["age"]))
    c5.metric("Position",  primary_pos_full)
    c2.metric("Overall",   f"{overall}/100")
    c3.metric("Potential", f"{potential}/100")
    c4.metric("Growth",    f"+{growth}")

    pace      = (player["acceleration"] + player["sprint_speed"]) / 2
    shooting  = (player["finishing"] + player["shot_power"] + player["long_shots"]) / 3
    passing   = (player["short_passing"] + player["long_passing"] + player["vision"]) / 3
    dribbling = (player["dribbling"] + player["ball_control"] + player["agility"]) / 3
    defending = (player["standing_tackle"] + player["sliding_tackle"] + player["interceptions"]) / 3
    physical  = (player["strength"] + player["stamina"] + player["balance"]) / 3

    skills = {
        "Pace": pace, "Shoot": shooting, "Pass": passing,
        "Dribble": dribbling, "Defend": defending, "Physical": physical
    }

    col_left, col_mid, col_right = st.columns([1.9, 1.3, 1.6])

    with col_left:
        fig, ax = plt.subplots(figsize=(3.2, 2.2))
        ax.barh(list(skills.keys()), list(skills.values()))
        ax.set_xlim(0, 100)
        ax.tick_params(labelsize=7)
        ax.set_title("Skills", fontsize=8)
        st.pyplot(fig)

    with col_mid:
        attack   = skills["Shoot"] + skills["Pace"]
        midfield = skills["Pass"] + skills["Dribble"]
        defense  = skills["Defend"] + skills["Physical"]

        fig2, ax2 = plt.subplots(figsize=(2.6, 2.2))
        ax2.pie(
            [attack, midfield, defense],
            labels=["Att", "Mid", "Def"],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 7}
        )
        ax2.set_title("Style", fontsize=8)
        st.pyplot(fig2)

    with col_right:
        if shooting > 70 and pace > 70:
            role = "Attacking Forward"
        elif passing > 70 and dribbling > 70:
            role = "Playmaking Midfielder"
        elif defending > 70 and physical > 70:
            role = "Defensive Specialist"
        else:
            role = "Balanced Utility Player"

        st.markdown("**🧠 AI Role**")
        st.success(role)

        archetype_input  = build_archetype_features(player)
        archetype_scaled = archetype_scaler.transform(archetype_input[archetype_features])
        cluster_id       = archetype_model.predict(archetype_scaled)[0]
        player_archetype = ARCHETYPE_MAP.get(cluster_id, "Balanced Profile")

        st.markdown("**🧬 Player Archetype**")
        st.info(player_archetype)

        strongest = max(skills, key=skills.get)
        weakest   = min(skills, key=skills.get)

        st.markdown(
            f"""
            <div style="font-size:20px; line-height:1.4;">
            <b>📌 Insights</b><br>
            • <b>Strongest:</b> {strongest}<br>
            • <b>Weakest:</b> {weakest}<br>
            • <b>Versatility:</b> High<br>
            • <b>Growth Level:</b> {'High' if growth >= 8 else 'Moderate'}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🔮 ML Prediction")

    player_features = build_player_features(player, ml_features)
    predicted_peak  = int(round(ml_model.predict(player_features)[0]))
    expected_growth = predicted_peak - overall
    confidence      = get_prediction_confidence(ml_model, player_features)

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Peak Overall", predicted_peak)
    m2.metric("Expected Growth",        f"+{expected_growth}")
    m3.metric("Model Confidence",       confidence)

    # AI SCOUT REPORT
    st.markdown("---")
    st.markdown("### 🤖 AI Scout Report")

    if st.button("🧠 Generate Scout Report", key="scout_tab1"):
        with st.spinner("Analysing player..."):
            skills_for_ai = get_player_skills(player)
            report = call_groq(
                prompt_player_scout(player, skills_for_ai, player_archetype,
                                    predicted_peak, confidence)
            )
        html = markdown.markdown(report)
        st.markdown(f'<div class="ai-box">{html}</div>', unsafe_allow_html=True)
    else:
        st.caption("Click above to get a full AI-generated scouting report for this player.")



# TAB 2 — TACTICAL FIT  (realistic pitch)

with tab2:
    player = selected_player
    st.subheader("🎯 Tactical Position Fit Analysis")
    st.caption("How well does this player's stats match what each position demands?")

    player_skills = get_player_skills(player)
    primary_pos   = player["positions"].split(",")[0].strip()
    fit_scores    = {pos: compute_position_fit(player_skills, pos) for pos in POSITION_DEMANDS}

    col_pitch, col_table = st.columns([1.6, 1.4])

    with col_pitch:
        fig, ax = plt.subplots(figsize=(6.0, 5.5))
        draw_realistic_pitch(ax, fig)

        for pos, score in fit_scores.items():
            x, y   = position_coordinates_image(pos)
            colour = fit_colour(score)

            # Outer glow ring for natural position
            if pos == primary_pos:
                ax.scatter(x, y, s=520, c="gold",
                           edgecolors="white", linewidths=0, zorder=4, alpha=0.4)

            ax.scatter(x, y, s=260 + score * 1.5, c=colour,
                       edgecolors="white", linewidths=1.2, zorder=5, alpha=0.92)
            ax.text(x, y + 0.5, pos,
                    ha="center", va="center",
                    fontsize=5.5, color="white", weight="bold", zorder=6)
            ax.text(x, y - 5.2, f"{score:.0f}%",
                    ha="center", va="top",
                    fontsize=5, color="white", zorder=6)

        legend_elements = [
            mpatches.Patch(color="#2ecc71", label="Good fit  (>=72%)"),
            mpatches.Patch(color="#f1c40f", label="Playable  (52-71%)"),
            mpatches.Patch(color="#e74c3c", label="Poor fit  (<52%)"),
            mpatches.Patch(color="gold",    label="Natural position", alpha=0.6),
        ]
        ax.legend(handles=legend_elements, loc="lower right",
                  fontsize=6, framealpha=0.5,
                  facecolor="#111", labelcolor="white")

        ax.set_title(f"{player['name']} — Position Fit Heatmap",
                     color="white", fontsize=9, pad=8)
        st.pyplot(fig)

    with col_table:
        st.markdown("**Position Fit Scores**")
        rows = []
        for pos, score in sorted(fit_scores.items(), key=lambda x: x[1], reverse=True):
            rows.append({
                "Position":   f"{pos} ({expand_position(pos)})",
                "Fit %":      f"{score:.0f}%",
                "Assessment": fit_label(score),
                "Key Gap":    top_missing_skills(player_skills, pos)
            })
        fit_df = pd.DataFrame(rows)
        st.dataframe(fit_df, use_container_width=True, height=500, hide_index=True)

    st.markdown("---")
    best_pos  = max(fit_scores, key=fit_scores.get)
    worst_pos = min(fit_scores, key=fit_scores.get)

    b1, b2, b3 = st.columns(3)
    b1.metric("Natural Position",  f"{primary_pos} — {fit_scores[primary_pos]:.0f}%")
    b2.metric("Best Suited Role",  f"{best_pos} — {fit_scores[best_pos]:.0f}%")
    b3.metric("Worst Suited Role", f"{worst_pos} — {fit_scores[worst_pos]:.0f}%")


# TAB 3 — COMPARE PLAYERS

with tab3:
    st.subheader("🆚 Player vs Player Comparison")

    sel1, sel2 = st.columns(2)
    with sel1:
        player_a_name = st.selectbox("Player A", sorted(df["name"].unique()), key="cmp_a")
    with sel2:
        player_b_name = st.selectbox("Player B", sorted(df["name"].unique()), index=1, key="cmp_b")

    player_a = df[df["name"] == player_a_name].iloc[0]
    player_b = df[df["name"] == player_b_name].iloc[0]

    m1, m2, m3, m4 = st.columns(4)

    a_overall   = int(player_a["overall_rating"])
    b_overall   = int(player_b["overall_rating"])
    a_potential = int(player_a["potential"])
    b_potential = int(player_b["potential"])
    a_growth    = a_potential - a_overall
    b_growth    = b_potential - b_overall
    pos_a       = player_a["positions"].split(",")[0].strip()
    pos_b       = player_b["positions"].split(",")[0].strip()

    m1.metric("Overall (A | B)",   f"{a_overall}| {b_overall}")
    m2.metric("Potential (A | B)", f"{a_potential}| {b_potential}")
    m3.metric("Growth (A | B)",    f"+{a_growth} | +{b_growth}")
    m4.metric("Position (A | B)",  f"{pos_a} | {pos_b}",
              help=f"{expand_position(pos_a)} | {expand_position(pos_b)}")

    st.markdown("---")

    skills_a = get_player_skills(player_a)
    skills_b = get_player_skills(player_b)

    c1, c2, c3 = st.columns([2, 1.4, 1.4])

    with c1:
        labels   = list(skills_a.keys())
        values_a = list(skills_a.values())
        values_b = list(skills_b.values())
        y        = range(len(labels))

        fig, ax = plt.subplots(figsize=(3.2, 2.2))
        ax.barh([i + 0.18 for i in y], values_a, height=0.35, label="A")
        ax.barh([i - 0.18 for i in y], values_b, height=0.35, label="B")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_title("Skills", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        plt.tight_layout(pad=0.6)
        st.pyplot(fig)

    with c2:
        a_attack = skills_a["Shooting"] + skills_a["Pace"]
        a_mid    = skills_a["Passing"]  + skills_a["Dribbling"]
        a_def    = skills_a["Defending"]+ skills_a["Physical"]

        fig_a, ax_a = plt.subplots(figsize=(2.2, 2.2))
        ax_a.pie([a_attack, a_mid, a_def], labels=["Att","Mid","Def"],
                 autopct="%1.0f%%", startangle=90, radius=0.8,
                 textprops={"fontsize": 7})
        ax_a.set_title("Player A", fontsize=9)
        plt.tight_layout(pad=0.6)
        st.pyplot(fig_a)

    with c3:
        b_attack = skills_b["Shooting"] + skills_b["Pace"]
        b_mid    = skills_b["Passing"]  + skills_b["Dribbling"]
        b_def    = skills_b["Defending"]+ skills_b["Physical"]

        fig_b, ax_b = plt.subplots(figsize=(2.2, 2.2))
        ax_b.pie([b_attack, b_mid, b_def], labels=["Att","Mid","Def"],
                 autopct="%1.0f%%", startangle=90, radius=0.8,
                 textprops={"fontsize": 7})
        ax_b.set_title("Player B", fontsize=9)
        plt.tight_layout(pad=0.6)
        st.pyplot(fig_b)

    # AI VERDICT
    st.markdown("---")

    arch_input_a  = build_archetype_features(player_a)
    arch_scaled_a = archetype_scaler.transform(arch_input_a[archetype_features])
    archetype_a   = ARCHETYPE_MAP.get(archetype_model.predict(arch_scaled_a)[0], "Balanced Profile")

    arch_input_b  = build_archetype_features(player_b)
    arch_scaled_b = archetype_scaler.transform(arch_input_b[archetype_features])
    archetype_b   = ARCHETYPE_MAP.get(archetype_model.predict(arch_scaled_b)[0], "Balanced Profile")

    if st.button("🧠 Generate AI Verdict", key="ai_compare"):
        with st.spinner("Comparing players..."):
            verdict = call_groq(
                prompt_compare(player_a, skills_a, archetype_a,
                               player_b, skills_b, archetype_b)
            )
        html = markdown.markdown(verdict)
        st.markdown(f'<div class="ai-box">{html}</div>', unsafe_allow_html=True)
    else:
        better_now    = player_a_name if a_overall   > b_overall   else player_b_name
        better_future = player_a_name if a_potential > b_potential else player_b_name
        st.markdown(
            f"""
            <div style="font-size:19px; line-height:1.4; margin-top:6px;">
            🧠 <b>AI Verdict</b><br>
            • Better current performer: <b>{better_now}</b><br>
            • Higher future potential: <b>{better_future}</b><br>
            • Skill profiles suggest different tactical suitability<br>
            <span style="font-size:14px; color:#888;">
              Click "Generate AI Verdict" for a full in-depth analysis
            </span>
            </div>
            """,
            unsafe_allow_html=True
        )


# TAB 4 — TEAM BUILDER

with tab4:
    st.subheader("🏆 Team Builder – Squad Optimization")

    TEAM_ROLES = ["GK","LB","CB","RB","CDM","CM","LM","RM","LW","RW","ST"]

    POSITION_SKILLS = {
        "GK":  ["gk_diving","gk_reflexes"],
        "CB":  ["standing_tackle","interceptions","strength"],
        "LB":  ["pace","standing_tackle"],
        "RB":  ["pace","standing_tackle"],
        "CDM": ["interceptions","short_passing"],
        "CM":  ["short_passing","dribbling"],
        "LM":  ["pace","dribbling","short_passing","standing_tackle"],
        "RM":  ["pace","dribbling","short_passing","standing_tackle"],
        "LW":  ["pace","dribbling"],
        "RW":  ["pace","dribbling"],
        "ST":  ["finishing","shot_power"]
    }

    if "team" not in st.session_state:
        st.session_state.team = {}
        used_players = set()
        for role in TEAM_ROLES:
            eligible = df[df["positions"].str.contains(role, na=False)]
            eligible = eligible.sort_values("overall_rating", ascending=False)
            for _, row in eligible.iterrows():
                if row["name"] not in used_players:
                    st.session_state.team[role] = row
                    used_players.add(row["name"])
                    break

    team = st.session_state.team

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot([0,100,100,0,0], [0,0,100,100,0], color="black")
    ax.axhline(50, color="gray", linewidth=0.4)
    ax.plot([0,16,16,0],    [21,21,79,79], color="black")
    ax.plot([100,84,84,100],[21,21,79,79], color="black")
    ax.add_patch(plt.Circle((50,50), 9.15, fill=False))
    for role, row in team.items():
        x, y = position_coordinates_image(role)
        ax.scatter(x, y, s=100, color="green", zorder=5)
        ax.text(x, y+2.5, row["name"], ha="center", va="bottom", fontsize=6)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    st.pyplot(fig)

    col_left, col_right = st.columns([1.4, 1.6])

    with col_left:
        st.markdown("### 👥Team details")
        current_xi_df = pd.DataFrame([
            {"Role": role, "Player": row["name"], "Overall": row["overall_rating"]}
            for role, row in team.items()
        ])
        st.dataframe(current_xi_df, use_container_width=True, height=380)

    with col_right:
        st.markdown("### 🔁 Replacement Suggestions")
        replace_role   = st.selectbox("Select position to replace", TEAM_ROLES)
        current_player = team[replace_role]
        current_rating = current_player["overall_rating"]

        candidates = df[
            df["positions"].str.contains(replace_role, na=False) &
            (~df["name"].isin([p["name"] for p in team.values()]))
        ].copy()

        def fit_score(row):
            skills = POSITION_SKILLS.get(replace_role, [])
            vals = [row[s] for s in skills if s in row]
            return sum(vals) / len(vals) if vals else 0

        candidates["FitScore"] = candidates.apply(fit_score, axis=1)
        candidates["Upgrade"]  = candidates["overall_rating"] - current_rating

        top5 = candidates.sort_values(
            ["Upgrade","FitScore"], ascending=False
        ).head(5)

        for _, row in top5.iterrows():
            c1, c2, c3, c4 = st.columns([3, 1.2, 1.2, 1.5])
            c1.markdown(f"**{row['name']}**")
            c2.markdown(f"{row['overall_rating']}")
            c3.markdown(f"+{row['Upgrade']}")
            if c4.button("Replace", key=f"replace_{replace_role}_{row['name']}"):
                st.session_state.team[replace_role] = row
                st.rerun()

    # AI SQUAD ANALYSIS
    st.markdown("---")
    st.markdown("### 🤖 AI Squad Analysis")

    if st.button("🧠 Analyse My Squad", key="ai_squad"):
        with st.spinner("Analysing your squad..."):
            team_data = [
                {"role": role, "name": row["name"],
                 "overall": int(row["overall_rating"]), "positions": row["positions"]}
                for role, row in team.items()
            ]
            analysis = call_groq(prompt_squad(team_data))
        html = markdown.markdown(analysis)
        st.markdown(f'<div class="ai-box">{html}</div>', unsafe_allow_html=True)
    else:
        st.caption("Click above to get an AI tactical analysis of your current XI.")

# TAB 5 — SCOUTING HUB

with tab5:
    st.subheader("🔍 Scouting Hub – Find Your Player")
    st.caption("Filter the full database to find players that match your exact scouting criteria.")

    f1, f2, f3 = st.columns(3)
    f4, f5, f6 = st.columns(3)

    all_positions = sorted(set(
        p.strip()
        for positions in df["positions"].dropna()
        for p in str(positions).split(",")
    ))
    all_nationalities = sorted(df["nationality"].dropna().unique())
    all_roles         = sorted(df["recommended_role"].dropna().unique())

    with f1:
        pos_filter = st.multiselect("Position", all_positions)
    with f2:
        nat_filter = st.multiselect("Nationality", all_nationalities)
    with f3:
        role_filter = st.multiselect("Recommended Role", all_roles)
    with f4:
        age_range = st.slider("Age Range", int(df["age"].min()), int(df["age"].max()), (16, 30))
    with f5:
        ovr_range = st.slider("Min Overall", 40, 99, 65)
    with f6:
        pot_range = st.slider("Min Potential", 40, 99, 70)

    filtered = df.copy()
    if pos_filter:
        filtered = filtered[
            filtered["positions"].apply(
                lambda x: any(p in str(x).split(",") for p in pos_filter)
            )
        ]
    if nat_filter:
        filtered = filtered[filtered["nationality"].isin(nat_filter)]
    if role_filter:
        filtered = filtered[filtered["recommended_role"].isin(role_filter)]

    filtered = filtered[
        (filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1]) &
        (filtered["overall_rating"] >= ovr_range) &
        (filtered["potential"] >= pot_range)
    ]

    sort_col = st.selectbox(
        "Sort by",
        ["overall_rating", "potential", "growth_index", "value_euro", "age"],
        index=0
    )

    filtered_sorted = filtered.sort_values(sort_col, ascending=False)
    display_cols = ["name","age","nationality","positions",
                    "overall_rating","potential","growth_index",
                    "value_euro","recommended_role","potential_level"]
    display_cols = [c for c in display_cols if c in filtered_sorted.columns]

    st.markdown(f"**{len(filtered_sorted)} players found**")
    st.dataframe(
        filtered_sorted[display_cols].reset_index(drop=True),
        use_container_width=True, height=420
    )

    # HIDDEN GEM FINDER
    st.markdown("---")
    st.markdown("### 💎 Hidden Gem Finder")
    st.caption("High potential, low value — players worth signing before anyone else notices.")

    gems = df[
        (df["growth_index"] >= 10) &
        (df["potential"] >= 78) &
        (df["age"] <= 23) &
        (df["overall_rating"] <= 72)
    ].sort_values("growth_index", ascending=False).head(10)

    gem_cols = ["name","age","nationality","positions",
                "overall_rating","potential","growth_index","value_euro"]
    gem_cols = [c for c in gem_cols if c in gems.columns]

    st.dataframe(gems[gem_cols].reset_index(drop=True),
                 use_container_width=True, height=360)
