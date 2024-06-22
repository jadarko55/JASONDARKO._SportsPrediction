import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError as e:
    st.error(f'File not found error: {e}')
except Exception as e:
    st.error(f'Error loading model/scaler: {e}')

st.title("Football Predictions")

def features():
    potential = st.number_input("Player's Potential")
    value_eur = st.number_input("Player's Value in Euros")
    wage_eur = st.number_input("Player's Wage in Euros")
    age = st.number_input("Player's Age")
    height_cm = st.number_input("Player's Height (cm)")
    weight_kg = st.number_input("Player's Weight (kg)")
    league_level = st.number_input("Player's League Level")
    weak_foot = st.number_input("Player's Weak Foot")
    skill_moves = st.number_input("Player's Skill Moves")
    international_reputation = st.number_input("Player's International Reputation")
    pace = st.number_input("Player's Pace")
    shooting = st.number_input("Player's Shooting")
    passing = st.number_input("Player's Passing")
    dribbling = st.number_input("Player's Dribbling")
    defending = st.number_input("Player's Defending")
    physic = st.number_input("Player's Physic")
    attacking_crossing = st.number_input("Player's Attacking Crossing")
    attacking_finishing = st.number_input("Player's Attacking Finishing")
    attacking_heading_accuracy = st.number_input("Player's Attacking Heading Accuracy")
    attacking_short_passing = st.number_input("Player's Attacking Short Passing")
    attacking_volleys = st.number_input("Player's Attacking Volleys")
    skill_dribbling = st.number_input("Player's Skill Dribbling")
    skill_curve = st.number_input("Player's Skill Curve")
    skill_fk_accuracy = st.number_input("Player's Skill Freekick Accuracy")
    skill_long_passing = st.number_input("Player's Skill Long Passing")
    skill_ball_control = st.number_input("Player's Skill Ball Control")
    movement_acceleration = st.number_input("Player's Movement Acceleration")
    movement_sprint_speed = st.number_input("Player's Movement Sprint Speed")
    movement_agility = st.number_input("Player's Movement Agility")
    movement_reactions = st.number_input("Player's Movement Reactions")
    movement_balance = st.number_input("Player's Movement Balance")
    power_shot_power = st.number_input("Player's Power Shot Power")
    power_jumping = st.number_input("Player's Power Jumping")
    power_stamina = st.number_input("Player's Power Stamina")
    power_strength = st.number_input("Player's Power Strength")
    power_long_shots = st.number_input("Player's Power Long Shots")
    mentality_aggression = st.number_input("Player's Mentality Aggression")
    mentality_interceptions = st.number_input("Player's Mentality Interceptions")
    mentality_positioning = st.number_input("Player's Mentality Positioning")
    mentality_vision = st.number_input("Player's Mentality Vision")
    mentality_penalties = st.number_input("Player's Mentality Penalties")
    mentality_composure = st.number_input("Player's Mentality Composure")
    defending_marking_awareness = st.number_input("Player's Defending Marking Awareness")
    defending_standing_tackle = st.number_input("Player's Defending Standing Tackle")
    defending_sliding_tackle = st.number_input("Player's Defending Sliding Tackle")
    goalkeeping_diving = st.number_input("Player's Goalkeeping Diving")
    goalkeeping_handling = st.number_input("Player's Goalkeeping Handling")
    goalkeeping_kicking = st.number_input("Player's Goalkeeping Kicking")
    goalkeeping_positioning = st.number_input("Player's Goalkeeping Positioning")
    goalkeeping_reflexes = st.number_input("Player's Goalkeeping Reflexes")

    data = {
        'potential': potential,
        'value_eur': value_eur,
        'wage_eur': wage_eur,
        'age': age,
        'height_cm': height_cm,
        'weight_kg': weight_kg,
        'league_level': league_level,
        'weak_foot': weak_foot,
        'skill_moves': skill_moves,
        'international_reputation': international_reputation,
        'pace': pace,
        'shooting': shooting,
        'passing': passing,
        'dribbling': dribbling,
        'defending': defending,
        'physic': physic,
        'attacking_crossing': attacking_crossing,
        'attacking_finishing': attacking_finishing,
        'attacking_heading_accuracy': attacking_heading_accuracy,
        'attacking_short_passing': attacking_short_passing,
        'attacking_volleys': attacking_volleys,
        'skill_dribbling': skill_dribbling,
        'skill_curve': skill_curve,
        'skill_fk_accuracy': skill_fk_accuracy,
        'skill_long_passing': skill_long_passing,
        'skill_ball_control': skill_ball_control,
        'movement_acceleration': movement_acceleration,
        'movement_sprint_speed': movement_sprint_speed,
        'movement_agility': movement_agility,
        'movement_reactions': movement_reactions,
        'movement_balance': movement_balance,
        'power_shot_power': power_shot_power,
        'power_jumping': power_jumping,
        'power_stamina': power_stamina,
        'power_strength': power_strength,
        'power_long_shots': power_long_shots,
        'mentality_aggression': mentality_aggression,
        'mentality_interceptions': mentality_interceptions,
        'mentality_positioning': mentality_positioning,
        'mentality_vision': mentality_vision,
        'mentality_penalties': mentality_penalties,
        'mentality_composure': mentality_composure,
        'defending_marking_awareness': defending_marking_awareness,
        'defending_standing_tackle': defending_standing_tackle,
        'defending_sliding_tackle': defending_sliding_tackle,
        'goalkeeping_diving': goalkeeping_diving,
        'goalkeeping_handling': goalkeeping_handling,
        'goalkeeping_kicking': goalkeeping_kicking,
        'goalkeeping_positioning': goalkeeping_positioning,
        'goalkeeping_reflexes': goalkeeping_reflexes
    }

    df = pd.DataFrame(data, index=[0])
    return df

input_data = features()

try:
    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)

    st.subheader('Prediction')
    st.write(f'Predicted Rating: {prediction[0]}')

except Exception as e:
    st.error(f'Error in prediction: {e}')
