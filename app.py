import streamlit as st
import pickle
import pandas as pd
import os

st.title('LeagueOracle — Match-Win-Predictor')

# --- Hard-coded single model (LightGBM) 
@st.cache_resource
def load_lgbm_model(path='models/m_lgbm_model.pkl'):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

# Load the single model once for the whole app
lgb_model = load_lgbm_model()
best_model_name = 'm_lgbm_model.pkl' if lgb_model is not None else None

# --- Model selection / loading utilities ----------------------------------
def list_model_files(models_dir='models'):
   
    models = []
    try:
        if not os.path.exists(models_dir):
            return models
        for fname in os.listdir(models_dir):
            if fname.lower().endswith('.pkl'):
                models.append(os.path.join(models_dir, fname))
    except Exception:
        return []
    return sorted(models)


def load_model_from_path(path):

    try:
        with open(path, 'rb') as f:
            return pickle.load(f), None
    except Exception as e:
        return None, str(e)


def predict_row(model, df):
    """Run prediction and return (pred_label, confidence) or raise."""
    pred = model.predict(df)
    try:
        prob = float(model.predict_proba(df)[:, 1][0])
    except Exception:
        # fallback to decision_function where available
        prob = float(model.decision_function(df)[0])
    return int(pred[0]), prob


# Sidebar: page selector (place above model selection for easier navigation)
page = st.sidebar.radio('Page', ['Home', 'Methodology', 'Prediction', 'Takeaways & Credits'])


# Hard-coded model: use the single LightGBM model
model = lgb_model
st.sidebar.header('Model')
if model is not None:
    st.sidebar.success(f'Loaded built-in model: {best_model_name}')
else:
    st.sidebar.error('Built-in model `m_lgbm_model.pkl` not found. Place the file in the app folder.')

# Page selector (moved earlier above model selection)


def show_home():
    st.header('🏠 Home')
    st.markdown(f"""
    ### League of Legends Match Predictor

    This demo lets you explore how data from a single match can help predict the outcome — whether **Team 1** wins or not.

    - 🧮 **Prediction:** Build inputs and run the model.  
    - 📊 **Methodology:** Learn how the model analyzes match data.  
    - 🧠 **Insights:** Understand what features drive victory.

    **Current loaded model:** `{best_model_name or 'None'}`
    """)

    st.image("images/summoners-rift.webp", caption="Summoner’s Rift Map", width=800)

    st.markdown("""
    ### 🌍 The Battlefield: Summoner's Rift

    A **League of Legends** match is a 5v5 strategic battle fought on **Summoner's Rift** — a map split into three **lanes** and a **jungle**.

    - 🏰 **Nexus:** Each team's main base — destroy it to win.  
    - 🧱 **Inhibitors:** Unlock *super minions* when destroyed.  
    - 🗼 **Towers:** Defensive checkpoints protecting each lane.  
    - 🐉 **Dragons:** Elemental monsters giving permanent team buffs.  
    - 🐍 **Baron Nashor:** A late-game boss granting powerful buffs.  
    - ⚔️ **Top Lane:** Solo bruisers and duelists dominate here.  
    - 🛡️ **Mid Lane:** Fast-paced duels between mages and assassins.  
    - 🏹 **Bot Lane:** ADCs and Supports fight for scaling advantage.  
    - 🌲 **Jungle:** The roaming zone connecting all lanes.

    Every decision — from when to fight Baron to defending a tower — builds toward the ultimate goal: **destroying the enemy Nexus.**
    """)

    st.write("💡 Explore the Prediction tab to see how the model interprets these elements statistically!")



def show_methodology():
    st.markdown(
        """
        <style>
        .lol-container {
            background-color: rgba(10, 10, 25, 0.92);
            color: #E0E6ED;
            padding: 25px;
            border-radius: 18px;
            border: 2px solid #1f51ff;
            box-shadow: 0px 0px 15px #0055ff66;
            font-family: 'Segoe UI', sans-serif;
        }
        .lol-header {
            color: #FFD700;
            text-shadow: 0px 0px 10px #FFD700;
            font-size: 32px;
            text-align: center;
            margin-bottom: 10px;
        }
        .lol-subheader {
            color: #8AB4F8;
            font-size: 22px;
            margin-top: 25px;
        }
        .lol-table {
            background-color: rgba(30, 30, 50, 0.8);
            border-radius: 10px;
            padding: 10px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown('<div class="lol-container">', unsafe_allow_html=True)
    st.markdown('<div class="lol-header">⚙️ Methodology — Decoding the Summoner’s Rift</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Why Predicting League of Legends Matches Is Challenging

    Predicting the outcome of a **League of Legends** match means turning chaos into data.  
    Every skirmish, every objective, every minute can change the tide. The challenge lies in **complex, interacting events** and **human unpredictability**.

    **Why it’s hard:**
    -  Teams constantly **adapt strategies** mid-match — no two games flow the same.
    - Objectives have **nonlinear effects** — a Baron at 25 minutes is not the same as a Baron at 40.
    - **Team coordination** and **champion synergy** are intangible factors no dataset fully captures.
    - Momentum swings and comebacks make even leading teams vulnerable until the Nexus falls.

    The battlefield may look simple — but beneath it lies a web of **probabilities and momentum shifts**.
    """)

    st.markdown('<div class="lol-subheader"> Features & Their Impact on Victory</div>', unsafe_allow_html=True)

    st.markdown("""
    The model studies **in-game objectives** and **event sequences** to understand what drives a win.
    Below are the main features extracted from the dataset and how they influence a team’s chances:
    """)

    st.markdown("""
    <div class='lol-table'>
    <table style='width:100%; color:#E0E6ED;'>
    <tr><th> Feature</th><th> Description</th><th>Influence on Win</th></tr>
    <tr><td> gameDuration</td><td>Match length (seconds)</td><td>Short games favor early snowball comps; long ones favor scaling teams.</td></tr>
    <tr><td> firstBlood</td><td>Team securing first kill</td><td>Sets early momentum and tempo control.</td></tr>
    <tr><td> firstTower</td><td>First destroyed tower</td><td>Massive gold swing and early map control.</td></tr>
    <tr><td> firstInhibitor</td><td>First inhibitor taken</td><td>Indicates near-victory and base pressure dominance.</td></tr>
    <tr><td> firstDragon</td><td>First Dragon secured</td><td>Grants buffs and long-term scaling advantage.</td></tr>
    <tr><td> firstBaron</td><td>First Baron Nashor slain</td><td>Huge mid/late power spike and siege potential.</td></tr>
    <tr><td> firstRiftHerald</td><td>First Rift Herald secured</td><td>Push advantage; early tempo gain.</td></tr>
    <tr><td> t1_towerKills</td><td>Blue towers destroyed</td><td>Reflects map dominance for Blue Team.</td></tr>
    <tr><td> t1_inhibitorKills</td><td>Blue inhibitors destroyed</td><td>Major end-game strength indicator.</td></tr>
    <tr><td> t1_baronKills</td><td>Blue Barons slain</td><td>Often signals a near win for Blue.</td></tr>
    <tr><td> t1_dragonKills</td><td>Blue Dragons slain</td><td>Shows objective consistency and buff control.</td></tr>
    <tr><td> t1_riftHeraldKills</td><td>Blue Heralds taken</td><td>Push advantage in early lanes.</td></tr>
    <tr><td> t2_towerKills</td><td>Red towers destroyed</td><td>Mirror measure of map pressure.</td></tr>
    <tr><td> t2_inhibitorKills</td><td>Red inhibitors destroyed</td><td>Endgame control indicator for Red.</td></tr>
    <tr><td> t2_baronKills</td><td>Red Barons slain</td><td>Momentum swing or comeback factor.</td></tr>
    <tr><td> t2_dragonKills</td><td>Red Dragons slain</td><td>Scaling and map control over time.</td></tr>
    <tr><td> t2_riftHeraldKills</td><td>Red Heralds taken</td><td>Early game objective strength.</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ---
    ### 🧹 Data Wrangling & Preprocessing

    Raw game data was cleansed and transformed into a machine-readable format:
    -  Removed unnecessary or missing fields (e.g., champion IDs, bans, summoner spells).
    -  Encoded binary events (e.g., `firstBlood`, `firstTower`) as 1 or 2.
    -  Split data into **train/test** sets for unbiased evaluation.
    - Balanced class distribution between Blue and Red wins.
    """)

    st.markdown("""
    ---
    ### 📊 Exploratory Data Analysis (EDA)

    Before model training, visual exploration revealed game patterns:
    - **Baron** and **Inhibitor kills** showed strongest correlation with victory.
    - **First Blood** on its own had modest impact, but together with **First Tower**, improved win prediction.
    - Match durations clustered around **25–35 minutes**, where most decisive outcomes occur.

    🔹 **Visuals included:**
    - Objective frequency vs. win rates.
    - Correlation heatmaps.
    """)

    st.markdown("""
    ---
    ### 🤖 Model Development & Evaluation

    Multiple ML algorithms were deployed and tuned for performance:

    | Model | Type | Strengths |
    |--------|------|------------|
    |  **Decision Tree** | Single tree classifier | Easy to visualize and interpret. |
    |  **Random Forest** | Ensemble of trees | Handles complex interactions, reduces overfitting. |
    |  **LightGBM** | Gradient boosting | Extremely efficient, handles large datasets well. |

    Evaluation metrics used:
    - Accuracy, Precision, Recall, F1-score.
    - Confidence score for each match prediction.
    - ROC-AUC to measure overall separability.

    The **best performing model** was stored and integrated into the **Streamlit GUI** for real-time predictions.
    """)

    st.markdown("""
    ---
    ### 🧠 Summary

    The project transforms match stats into insights, predicting who will triumph — **Team Blue**  or **Team Red** .  
    It merges the **strategic essence of LoL** with the **power of machine learning**, bringing data science to the Summoner’s Rift.
    """)

    st.markdown("</div>", unsafe_allow_html=True)



def show_prediction():
    # --- Load Model ---
    @st.cache_resource
    def load_model():
        with open("models/m_lgbm_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model

    lgb_model = load_model()

    # --- App Title (page-level) ---
    st.title("⚔️ Match Winner Prediction App (LightGBM)")

    st.write("Enter match statistics below to predict the winning team:")

    # --- Define feature inputs ---
    features = [
        'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron',
        'firstDragon', 'firstRiftHerald',
        'tower_diff', 'inhib_diff', 'baron_diff', 'dragon_diff', 'rift_diff'
    ]

    # Create input fields dynamically
    inputs = {}
    st.subheader("Match Features")
    for feature in features:
        inputs[feature] = st.number_input(f"{feature}", value=0, step=1)

    # --- Predict Button ---
    if st.button("Predict Winner"):
        # Convert input into DataFrame
        input_df = pd.DataFrame([inputs])
        
        # Get probabilities & prediction
        probabilities = lgb_model.predict_proba(input_df)
        winner = lgb_model.predict(input_df)[0]
        
        # Display results
        st.success(f"🏆 **Predicted Winner: Team {int(winner)}**")
        st.metric("Team 1 Win Probability", f"{probabilities[0][0]*100:.3f}%")
        st.metric("Team 2 Win Probability", f"{probabilities[0][1]*100:.3f}%")

        # Show input data for clarity
        st.write("### Input Summary", input_df)



def show_takeaways():
    st.header('Takeaways & Credits')
    st.markdown(
        """
        - Key takeaways from modelling and results.

        So total 5 models were trained and evaluated; performance summary:

        | Model | Accuracy |
        |---|---:|
        | Decision Tree | 95.73% |
        | Random Forest | 95.75% |
        | LightGBM | 96.27% |

        ---
        **Limitations and next steps**
        - The model doesn't take into account the skill level of the player on either side which could affect the game.
        - The model also doesn't consider champion picks and bans which can significantly influence match outcomes.
        - The model assumes all matches are played on Summoner's Rift and may not generalize to other maps or game modes.
        - All games are from season 9 and may not reflect current meta or balance changes.

        **Improvements could include:**
        - Incorporating player skill metrics (rank, win rate, etc.) into the feature set.
        - Adding champion pick/ban data to capture team composition effects.
        - Using time-series data to model in-game momentum shifts rather than static end-of-game stats.


        **Credits**
        - Project author: Ankit Kolhe
        - Data source: Kaggle: https://www.kaggle.com/datasets/datasnaek/league-of-legends/data
        - Thanks to the open source community for tools like Streamlit, scikit-learn, and pandas.
        - Academic Mentors: Abhinay Gudadhe, Dr. Nisarg Gandhewar
        """
    )


# Route to the selected page
if page == 'Home':
    show_home()
elif page == 'Methodology':
    show_methodology()
elif page == 'Prediction':
    show_prediction()
elif page == 'Takeaways & Credits':
    show_takeaways()
