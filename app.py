import os
import sys
import time
import numpy as np
import streamlit as st
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure




PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

try:
    from smart_hvac_env import SmartHVACEnv
except ImportError:
    st.error("⚠️ Could not import 'SmartHVACEnv'. Please ensure smart_hvac_env.py is in the directory.")
    st.stop()

st.set_page_config(
    page_title="Smart HVAC Twin",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        /* Reduce top padding to pull everything up */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        /* METRIC CARD STYLING */
        div[data-testid="stMetric"] {
            background-color: #f0f2f6; /* Light grey background */
            padding: 10px;
            border-radius: 5px;
            border-left: 5px solid #4e8cff;
            color: black; /* FORCE TEXT COLOR TO BLACK */
        }
        
        /* Force label colors inside metrics to black */
        div[data-testid="stMetric"] label {
            color: #000000 !important;
        }
        
        /* Force value colors inside metrics to black */
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #000000 !important;
        }

        /* Compact title */
        h1 { font-size: 1.8rem !important; margin-bottom: 0rem !important; }
        h3 { font-size: 1.2rem !important; margin-top: 0rem !important; }
    </style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_model():
    model_path = os.path.join(PROJECT_ROOT, "dqn_smart_hvac_all_improvements.zip")
    if not os.path.exists(model_path):
        return None 
    try:
        model = DQN.load(model_path)
        
        new_logger = configure("./logs", ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
model = load_model()


if "history" not in st.session_state:
    st.session_state.history = {"steps": [], "T_in": [], "T_out": []}

if "env" not in st.session_state:
    st.session_state.env = SmartHVACEnv() 
    st.session_state.env.max_occupancy = 100
    st.session_state.obs, _ = st.session_state.env.reset()
    st.session_state.prev_occupancy = st.session_state.env.occupancy
    st.session_state.total_energy = 0.0
    st.session_state.last_info = {"ideal_temp": 24.0}

if "total_time_min" not in st.session_state:
    st.session_state.total_time_min = 0.0
    st.session_state.total_ac_on_min = 0.0

env = st.session_state.env

if not hasattr(env, "forced_ac_state"):
    env.forced_ac_state = None  
if not hasattr(env, "manual_override"):
    env.manual_override = False 




def run_simulation_step(model):
    r = np.random.random()
    delta = np.random.randint(-3, 4) if r < 0.90 else np.random.randint(-8, 9)
    env.occupancy = max(0, min(env.occupancy + delta, env.max_occupancy))
    st.session_state.prev_occupancy = env.occupancy


    if model:
        action, _ = model.predict(st.session_state.obs, deterministic=False) 
    else:
        action = env.action_space.sample()

    next_obs, reward, done, _, info = env.step(action)

    if model:
        model.replay_buffer.add(
            st.session_state.obs,  
            next_obs,              
            action,                
            reward,                
            done,                  
            [info]                 
        )

        if model.replay_buffer.size() > model.batch_size:
            model.train(gradient_steps=1, batch_size=model.batch_size)
    
    st.session_state.history["steps"].append(st.session_state.total_time_min)
    st.session_state.history["T_in"].append(env.T_in)
    st.session_state.history["T_out"].append(env.T_out)
    
    if len(st.session_state.history["steps"]) > 100:
        st.session_state.history["steps"].pop(0)
        st.session_state.history["T_in"].pop(0)
        st.session_state.history["T_out"].pop(0)
    
    st.session_state.total_time_min = info.get("total_time_min", st.session_state.total_time_min + 3)
    st.session_state.total_ac_on_min = info.get("total_ac_on_min", st.session_state.total_ac_on_min)

    st.session_state.total_energy += info.get("energy_kwh", 0.0)
    st.session_state.last_info = info
    st.session_state.obs = next_obs  

    if done:
        st.session_state.obs, _ = env.reset()
        st.session_state.prev_occupancy = env.occupancy
        st.session_state.total_energy = 0.0
        st.session_state.total_time_min = 0.0
        st.session_state.total_ac_on_min = 0.0



with st.sidebar:
    st.markdown("### 🎮 Control Panel")
    
    col_play, col_reset = st.columns(2)
    with col_play:
        auto_run = st.toggle("🔄 Auto Run", value=False)
    with col_reset:
        if st.button("🔁 Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    speed = st.slider("Speed (Delay)", 0.0, 1.0, 0.1)

    if not auto_run:
        if st.button("▶ Step Once", use_container_width=True):
            run_simulation_step(model)
            st.rerun()

    st.markdown("---")
    st.markdown("### 🍂 Season Control")
    selected_season = st.selectbox(
        "Current Season", 
        ["Summer", "Monsoon", "Autumn", "Winter"],
        index=0
    )
    env.set_season(selected_season)
    
    st.markdown("### 🌡️ Temperature Overrides")
    
    is_manual = getattr(env, "manual_override", False)
    
    if is_manual:
        st.warning("⚠️ Manual Mode Active")
        if st.button("🧠 Resume ML Auto-Control", use_container_width=True):
            env.manual_override = False
            st.rerun()
    else:
        st.success("🤖 ML Control Active")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("🔥 +1°C", key="btn_plus", use_container_width=True):
            env.manual_override = True   
            env.setpoint = min(30.0, env.setpoint + 1.0)
            st.rerun()
    with col_t2:
        if st.button("❄️ -1°C", key="btn_minus", use_container_width=True):
            env.manual_override = True   
            env.setpoint = max(16.0, env.setpoint - 1.0)
            st.rerun()
            
    st.markdown("---")
    st.markdown("### ⚡ AC Master Control")

    col_ac1, col_ac2 = st.columns(2)

    with col_ac1:
        if st.button("🟥 Force AC OFF", use_container_width=True):
            env.forced_ac_state = False
            st.rerun()

    with col_ac2:
        if st.button("🟩 Force AC ON", use_container_width=True):
            env.forced_ac_state = True
            st.rerun()

    if st.button("🤖 Return Control to AI", use_container_width=True):
        env.forced_ac_state = None
        st.rerun()
    st.markdown("### 🚀 Power Settings")
    
    if not hasattr(env, "turbo_mode"):
        env.turbo_mode = False
        
    turbo_on = st.toggle("Turbo Cooling", value=env.turbo_mode)
    
    env.turbo_mode = turbo_on
    
    if turbo_on:
        st.caption("✅ **Active:** AC will force temp down regardless of heat.")
    else:
        st.caption("⚠️ **Inactive:** AC may struggle in high heat/crowds.")



def render_room_svg(env, prev_occ, ac_used):
    curr_occ = env.occupancy
    width, height = 800, 450 
    sx, sy = width / 10, height / 6

    svg = []
    svg.append(f'<rect width="{width}" height="{height}" fill="#f8f9fa" stroke="#333" stroke-width="2" rx="15"/>')

    xs = np.linspace(1, 9, 10)
    ys = np.linspace(1, 5, 10)
    for i, (x, y) in enumerate([(x, y) for y in ys for x in xs]):
        px, py = x * sx, height - y * sy
        svg.append(f'<text x="{px}" y="{py}" font-size="20" text-anchor="middle" fill="#ccc">🪑</text>')
        if i < curr_occ:
            svg.append(f'<text x="{px}" y="{py - 10}" font-size="22" text-anchor="middle">👤</text>')


    forced_state = getattr(env, "forced_ac_state", None)
    
    if forced_state is False:
        ac_color = "#e74c3c"
        ac_text = "AC FORCED OFF 🟥"
    elif forced_state is True:
        ac_color = "#27ae60"
        ac_text = "AC FORCED ON 🟩"
    else:
        ac_color = "#2ecc71" if ac_used else "#95a5a6"
        ac_text = "AC ON ❄️" if ac_used else "AC OFF 💤"

    svg.append(f'<rect x="10" y="10" width="160" height="40" fill="{ac_color}" rx="5" opacity="0.9"/>')
    svg.append(f'<text x="90" y="38" font-family="Arial" font-weight="bold" font-size="20" fill="white" text-anchor="middle">{ac_text}</text>')

    return f'<svg viewBox="0 0 {width} {height}" style="width:100%; max-height:400px;">{"".join(svg)}</svg>'


st.title("🏢 Digital Twin: Smart HVAC Controller")

ideal_temp = st.session_state.last_info.get("ideal_temp", 24.0)
err = env.T_in - ideal_temp

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric(
        label="🌡️ Room Temp", 
        value=f"{env.T_in:.2f} °C", 
        delta=f"{err:+.2f} from ideal"
    )

with kpi2:
    st.metric(label="🌤️ Outside Temp", value=f"{env.T_out:.1f} °C")

with kpi3:
    st.metric(label="🧠 ML Target", value=f"{ideal_temp:.1f} °C")

with kpi4:
    ac_used = st.session_state.last_info.get("ac_on", False)
    
    forced_state = getattr(env, "forced_ac_state", None)

    if forced_state is False:
        ac_state = "FORCED OFF 🟥"
    elif forced_state is True:
        ac_state = "FORCED ON 🟩"
    else:
        ac_state = "RUNNING ❄️" if ac_used else "IDLE 💤"
    st.metric(label="⚙️ System State", value=ac_state)

with kpi5:
    st.metric(label="⚡ Energy Used", value=f"{st.session_state.total_energy:.2f} kWh")

col_visual, col_stats = st.columns([2, 1])

with col_visual:
    st.markdown("### Live Occupancy Map")
    st.markdown(
        render_room_svg(
            env,
            st.session_state.prev_occupancy,
            st.session_state.last_info.get("ac_on", False)
        ),
        unsafe_allow_html=True
    )

with col_stats:
    st.markdown("### ⏱️ Performance")
    
    total_min = st.session_state.total_time_min
    ac_on_min = st.session_state.total_ac_on_min
    
    h, m = divmod(int(total_min), 60)
    ah, am = divmod(int(ac_on_min), 60)
    
    duty = (ac_on_min / total_min * 100) if total_min > 0 else 0.0

    st.info(f"**Simulated Time:** {h}h {m}m")
    
    st.write(f"**AC Duty Cycle:** {duty:.1f}%")
    st.progress(int(duty))
    
    st.write(f"**Active Cooling:** {ah}h {am}m")
    
    st.markdown("---")
    st.markdown(f"**Occupancy:** {env.occupancy} / {env.max_occupancy}")
    st.progress(env.occupancy / env.max_occupancy)

if auto_run:
    time.sleep(speed)
    run_simulation_step(model)
    st.rerun()