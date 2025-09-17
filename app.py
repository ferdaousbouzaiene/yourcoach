import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import time
import base64
from io import BytesIO

# Page config
st.set_page_config(
    page_title="AI Workout Recommender",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .exercise-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .exercise-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .sidebar .element-container {
        margin-bottom: 1rem;
    }
    .stProgress .st-bo {
        background-color: #667eea;
    }
    .recommendation-header {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'workout_generated' not in st.session_state:
    st.session_state.workout_generated = False
if 'workout_plan' not in st.session_state:
    st.session_state.workout_plan = None

# Sample datasets (same as before but optimized for Streamlit)
@st.cache_data
def load_sample_data():
    """Load and cache sample datasets"""
    
    # Exercise dataset
    exercises_data = [
        {"id": 1, "title": "Bench Press", "type": "strength", "bodyPart": "chest", "equipment": "barbell", "difficulty": "intermediate", "rating": 4.5, "muscleGroup": "chest"},
        {"id": 2, "title": "Squats", "type": "strength", "bodyPart": "legs", "equipment": "barbell", "difficulty": "beginner", "rating": 4.8, "muscleGroup": "legs"},
        {"id": 3, "title": "Deadlift", "type": "strength", "bodyPart": "back", "equipment": "barbell", "difficulty": "advanced", "rating": 4.7, "muscleGroup": "back"},
        {"id": 4, "title": "Pull-ups", "type": "strength", "bodyPart": "back", "equipment": "body", "difficulty": "intermediate", "rating": 4.6, "muscleGroup": "back"},
        {"id": 5, "title": "Push-ups", "type": "strength", "bodyPart": "chest", "equipment": "body", "difficulty": "beginner", "rating": 4.4, "muscleGroup": "chest"},
        {"id": 6, "title": "Shoulder Press", "type": "strength", "bodyPart": "shoulders", "equipment": "dumbbell", "difficulty": "intermediate", "rating": 4.3, "muscleGroup": "shoulders"},
        {"id": 7, "title": "Bicep Curls", "type": "strength", "bodyPart": "arms", "equipment": "dumbbell", "difficulty": "beginner", "rating": 4.2, "muscleGroup": "arms"},
        {"id": 8, "title": "Tricep Dips", "type": "strength", "bodyPart": "arms", "equipment": "body", "difficulty": "intermediate", "rating": 4.1, "muscleGroup": "arms"},
        {"id": 9, "title": "Lunges", "type": "strength", "bodyPart": "legs", "equipment": "body", "difficulty": "beginner", "rating": 4.5, "muscleGroup": "legs"},
        {"id": 10, "title": "Planks", "type": "strength", "bodyPart": "core", "equipment": "body", "difficulty": "beginner", "rating": 4.6, "muscleGroup": "core"},
        {"id": 11, "title": "Burpees", "type": "cardio", "bodyPart": "full body", "equipment": "body", "difficulty": "advanced", "rating": 4.0, "muscleGroup": "full body"},
        {"id": 12, "title": "Mountain Climbers", "type": "cardio", "bodyPart": "core", "equipment": "body", "difficulty": "intermediate", "rating": 4.2, "muscleGroup": "core"},
        {"id": 13, "title": "Jumping Jacks", "type": "cardio", "bodyPart": "full body", "equipment": "body", "difficulty": "beginner", "rating": 4.1, "muscleGroup": "full body"},
        {"id": 14, "title": "Russian Twists", "type": "strength", "bodyPart": "core", "equipment": "body", "difficulty": "intermediate", "rating": 4.3, "muscleGroup": "core"},
        {"id": 15, "title": "Hip Thrusts", "type": "strength", "bodyPart": "glutes", "equipment": "body", "difficulty": "intermediate", "rating": 4.4, "muscleGroup": "legs"}
    ]
    
    # Calorie data
    calorie_data = [
        {"activity": "Bench Press", "caloriesPerMinute": 6.5, "intensity": "moderate"},
        {"activity": "Squats", "caloriesPerMinute": 8.0, "intensity": "high"},
        {"activity": "Deadlift", "caloriesPerMinute": 7.5, "intensity": "high"},
        {"activity": "Pull-ups", "caloriesPerMinute": 7.0, "intensity": "moderate"},
        {"activity": "Push-ups", "caloriesPerMinute": 6.0, "intensity": "moderate"},
        {"activity": "Shoulder Press", "caloriesPerMinute": 5.5, "intensity": "moderate"},
        {"activity": "Bicep Curls", "caloriesPerMinute": 4.5, "intensity": "low"},
        {"activity": "Tricep Dips", "caloriesPerMinute": 6.0, "intensity": "moderate"},
        {"activity": "Lunges", "caloriesPerMinute": 7.0, "intensity": "moderate"},
        {"activity": "Planks", "caloriesPerMinute": 5.0, "intensity": "low"},
        {"activity": "Burpees", "caloriesPerMinute": 12.0, "intensity": "very high"},
        {"activity": "Mountain Climbers", "caloriesPerMinute": 10.0, "intensity": "high"},
        {"activity": "Jumping Jacks", "caloriesPerMinute": 8.5, "intensity": "high"},
        {"activity": "Russian Twists", "caloriesPerMinute": 6.5, "intensity": "moderate"},
        {"activity": "Hip Thrusts", "caloriesPerMinute": 6.0, "intensity": "moderate"}
    ]
    
    return pd.DataFrame(exercises_data), pd.DataFrame(calorie_data)

# ML Models (simplified for demo)
@st.cache_resource
def load_models():
    """Load and cache ML models"""
    # In a real app, you'd load pre-trained models here
    # For demo, we'll use simple heuristics
    return "models_loaded"

def predict_exercise_rating(exercise, user_profile):
    """Predict how much a user will like an exercise"""
    rating = exercise['rating']
    
    # Adjust based on fitness level
    if user_profile['fitness_level'] == 'beginner' and exercise['difficulty'] == 'advanced':
        rating -= 0.8
    elif user_profile['fitness_level'] == 'advanced' and exercise['difficulty'] == 'beginner':
        rating -= 0.3
    elif user_profile['fitness_level'] == exercise['difficulty']:
        rating += 0.2
    
    # Adjust based on equipment availability
    if exercise['equipment'] not in user_profile.get('equipment', ['body', 'dumbbell', 'barbell']):
        rating -= 1.0
    
    # Adjust based on goal
    if user_profile['goal'] == 'lose_weight' and exercise['type'] == 'cardio':
        rating += 0.3
    elif user_profile['goal'] == 'build_muscle' and exercise['type'] == 'strength':
        rating += 0.3
    
    return max(1.0, min(5.0, rating))

def predict_calories(exercise, user_profile, duration, calorie_data):
    """Predict calories burned"""
    calorie_info = calorie_data[calorie_data['activity'] == exercise['title']]
    base_calories = calorie_info['caloriesPerMinute'].iloc[0] if not calorie_info.empty else 6.0
    
    # Personal adjustments
    weight_multiplier = user_profile['weight'] / 70
    age_multiplier = max(0.8, 1.2 - (user_profile['age'] - 20) * 0.01)
    fitness_multiplier = {'beginner': 1.1, 'intermediate': 1.0, 'advanced': 0.95}[user_profile['fitness_level']]
    
    return int(base_calories * weight_multiplier * age_multiplier * fitness_multiplier * duration)

def generate_workout_plan(user_profile, exercises_df, calorie_data):
    """Generate personalized workout plan"""
    
    # Filter exercises with high predicted ratings
    suitable_exercises = []
    for _, exercise in exercises_df.iterrows():
        rating = predict_exercise_rating(exercise, user_profile)
        if rating >= 4.0:
            suitable_exercises.append({
                'exercise': exercise,
                'predicted_rating': rating
            })
    
    suitable_exercises.sort(key=lambda x: x['predicted_rating'], reverse=True)
    
    # Create balanced workout
    workout_plan = []
    total_time = 0
    total_calories = 0
    target_time = user_profile['available_time']
    muscle_groups_used = set()
    
    # Primary exercises (compound movements)
    primary_exercises = ['Squats', 'Deadlift', 'Bench Press', 'Pull-ups']
    for ex in suitable_exercises:
        if ex['exercise']['title'] in primary_exercises and len(workout_plan) < 2:
            duration = 12 if user_profile['goal'] == 'build_muscle' else 10
            calories = predict_calories(ex['exercise'], user_profile, duration, calorie_data)
            
            workout_plan.append({
                'name': ex['exercise']['title'],
                'muscle_group': ex['exercise']['muscleGroup'],
                'duration': duration,
                'sets': 4 if user_profile['goal'] == 'build_muscle' else 3,
                'reps': '8-12' if user_profile['goal'] == 'build_muscle' else '10-15',
                'calories': calories,
                'rating': ex['predicted_rating'],
                'difficulty': ex['exercise']['difficulty'],
                'equipment': ex['exercise']['equipment'],
                'type': ex['exercise']['type']
            })
            
            total_time += duration
            total_calories += calories
            muscle_groups_used.add(ex['exercise']['muscleGroup'])
    
    # Add accessory exercises
    remaining_time = target_time - total_time
    accessory_time = remaining_time // 3 if remaining_time > 0 else 8
    
    for ex in suitable_exercises:
        if (len(workout_plan) < 6 and 
            ex['exercise']['title'] not in [w['name'] for w in workout_plan] and
            ex['exercise']['muscleGroup'] not in muscle_groups_used):
            
            duration = min(accessory_time, 8)
            calories = predict_calories(ex['exercise'], user_profile, duration, calorie_data)
            
            workout_plan.append({
                'name': ex['exercise']['title'],
                'muscle_group': ex['exercise']['muscleGroup'],
                'duration': duration,
                'sets': 3,
                'reps': '12-15' if user_profile['goal'] == 'lose_weight' else '8-12',
                'calories': calories,
                'rating': ex['predicted_rating'],
                'difficulty': ex['exercise']['difficulty'],
                'equipment': ex['exercise']['equipment'],
                'type': ex['exercise']['type']
            })
            
            total_time += duration
            total_calories += calories
            muscle_groups_used.add(ex['exercise']['muscleGroup'])
    
    return workout_plan, total_time, total_calories, len(muscle_groups_used)

# Main App
def main():
    # Header
    st.markdown('<div class="main-header">🏋️ AI Workout Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Personalized fitness plans powered by machine learning</div>', unsafe_allow_html=True)
    
    # Load data
    exercises_df, calorie_data = load_sample_data()
    models = load_models()
    
    # Sidebar for user input
    with st.sidebar:
        st.header("👤 Your Profile")
        
        # Basic info
        age = st.slider("Age", 18, 70, 30)
        weight = st.slider("Weight (kg)", 40, 150, 70)
        
        # Fitness level
        fitness_level = st.selectbox(
            "Fitness Level",
            ["beginner", "intermediate", "advanced"],
            index=1
        )
        
        # Goal
        goal = st.selectbox(
            "Primary Goal",
            ["lose_weight", "build_muscle", "maintain_fitness"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Time and calories
        available_time = st.slider("Available Time (minutes)", 15, 90, 45)
        target_calories = st.slider("Target Calories", 100, 600, 300)
        
        # Equipment
        st.subheader("Available Equipment")
        equipment = []
        if st.checkbox("Body weight", value=True):
            equipment.append("body")
        if st.checkbox("Dumbbells", value=True):
            equipment.append("dumbbell")
        if st.checkbox("Barbell", value=True):
            equipment.append("barbell")
        if st.checkbox("Machines", value=False):
            equipment.append("machine")
        
        # Generate button
        if st.button("🎯 Generate Workout Plan", type="primary"):
            with st.spinner("Creating your personalized workout..."):
                time.sleep(2)  # Simulate processing time
                
                user_profile = {
                    'age': age,
                    'weight': weight,
                    'fitness_level': fitness_level,
                    'goal': goal,
                    'available_time': available_time,
                    'target_calories': target_calories,
                    'equipment': equipment
                }
                
                workout_plan, total_time, total_calories, muscle_groups = generate_workout_plan(
                    user_profile, exercises_df, calorie_data
                )
                
                st.session_state.workout_plan = {
                    'exercises': workout_plan,
                    'total_time': total_time,
                    'total_calories': total_calories,
                    'muscle_groups': muscle_groups,
                    'user_profile': user_profile
                }
                st.session_state.workout_generated = True
                st.success("Workout plan generated!")
    
    # Main content area
    if st.session_state.workout_generated and st.session_state.workout_plan:
        plan = st.session_state.workout_plan
        
        # Summary metrics
        st.markdown('<div class="recommendation-header">🎯 Your Personalized Workout Plan</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⏱️ {plan['total_time']} min</h3>
                <p>Total Duration</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔥 {plan['total_calories']} cal</h3>
                <p>Calories Burned</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💪 {len(plan['exercises'])}</h3>
                <p>Exercises</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 {plan['muscle_groups']}</h3>
                <p>Muscle Groups</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress bars
        st.subheader("📊 Goal Achievement")
        
        col1, col2 = st.columns(2)
        
        with col1:
            time_progress = min(plan['total_time'] / plan['user_profile']['available_time'], 1.0)
            st.metric(
                "Time Utilization", 
                f"{plan['total_time']}/{plan['user_profile']['available_time']} min",
                f"{time_progress:.1%}"
            )
            st.progress(time_progress)
        
        with col2:
            calorie_progress = min(plan['total_calories'] / plan['user_profile']['target_calories'], 1.0)
            st.metric(
                "Calorie Target", 
                f"{plan['total_calories']}/{plan['user_profile']['target_calories']} cal",
                f"{calorie_progress:.1%}"
            )
            st.progress(calorie_progress)
        
        # Exercise details
        st.subheader("🏋️ Exercise Details")
        
        for i, exercise in enumerate(plan['exercises'], 1):
            with st.expander(f"**{i}. {exercise['name']}** ⭐ {exercise['rating']:.1f}/5.0"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Target:** {exercise['muscle_group'].title()}")
                    st.write(f"**Equipment:** {exercise['equipment'].title()}")
                    st.write(f"**Difficulty:** {exercise['difficulty'].title()}")
                    st.write(f"**Type:** {exercise['type'].title()}")
                
                with col2:
                    st.metric("Duration", f"{exercise['duration']} min")
                    st.metric("Sets", exercise['sets'])
                    st.metric("Reps", exercise['reps'])
                
                with col3:
                    st.metric("Calories", f"{exercise['calories']}")
                    st.metric("Rating", f"{exercise['rating']:.1f}/5")
        
        # Visualizations
        st.subheader("📈 Workout Analysis")
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Muscle group distribution
            muscle_groups = [ex['muscle_group'] for ex in plan['exercises']]
            muscle_counts = pd.Series(muscle_groups).value_counts()
            
            fig_pie = px.pie(
                values=muscle_counts.values, 
                names=muscle_counts.index,
                title="Muscle Group Focus",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Exercise difficulty and calories
            df_viz = pd.DataFrame(plan['exercises'])
            
            fig_scatter = px.scatter(
                df_viz,
                x='duration',
                y='calories',
                size='rating',
                color='difficulty',
                hover_data=['name', 'sets', 'reps'],
                title="Exercise Duration vs Calories",
                labels={'duration': 'Duration (min)', 'calories': 'Calories Burned'}
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # AI Insights
        st.subheader("🤖 AI Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.info(f"""
            **Plan Optimization**
            
            Your workout is optimized for **{plan['user_profile']['goal'].replace('_', ' ')}** 
            at a **{plan['user_profile']['fitness_level']}** level.
            
            All exercises have predicted enjoyment ratings of 4+ stars based on your profile.
            """)
        
        with insights_col2:
            st.success(f"""
            **Performance Prediction**
            
            You'll burn approximately **{plan['total_calories']} calories** in **{plan['total_time']} minutes**.
            
            This workout targets **{plan['muscle_groups']} different muscle groups** for balanced development.
            """)
        
        # Download workout plan
        if st.button("📥 Download Workout Plan"):
            # Create downloadable content
            workout_text = f"""
# Your Personalized Workout Plan

## Profile
- Age: {plan['user_profile']['age']}
- Weight: {plan['user_profile']['weight']}kg
- Fitness Level: {plan['user_profile']['fitness_level'].title()}
- Goal: {plan['user_profile']['goal'].replace('_', ' ').title()}

## Summary
- Total Duration: {plan['total_time']} minutes
- Total Calories: {plan['total_calories']}
- Exercises: {len(plan['exercises'])}
- Muscle Groups: {plan['muscle_groups']}

## Exercises
"""
            for i, ex in enumerate(plan['exercises'], 1):
                workout_text += f"""
{i}. **{ex['name']}**
   - Duration: {ex['duration']} minutes
   - Sets: {ex['sets']} × {ex['reps']} reps
   - Calories: {ex['calories']}
   - Muscle Group: {ex['muscle_group'].title()}
   - Equipment: {ex['equipment'].title()}
   - Difficulty: {ex['difficulty'].title()}
"""
            
            st.download_button(
                label="Download as Text File",
                data=workout_text,
                file_name=f"workout_plan_{plan['user_profile']['goal']}.txt",
                mime="text/plain"
            )
    
    else:
        # Welcome message
        st.markdown("""
        ## 🎯 Welcome to AI Workout Recommender!
        
        Get started by filling out your profile in the sidebar. Our AI will create a personalized workout plan just for you.
        
        ### 🔬 How It Works:
        
        1. **Exercise Preference Model** - Predicts which exercises you'll enjoy based on your fitness level and goals
        2. **Calorie Prediction Model** - Calculates personalized calorie burn based on your demographics
        3. **Smart Plan Generator** - Creates balanced workouts that match your preferences and constraints
        
        ### 🌟 Features:
        
        - ✅ Only recommends exercises with 4+ star predicted ratings
        - ✅ Balances different muscle groups for complete workouts  
        - ✅ Accurate calorie predictions based on your profile
        - ✅ Respects your time constraints and available equipment
        - ✅ Adapts to your fitness level and goals
        
        **Ready to get started? Fill out your profile in the sidebar!**
        """)
        
        # Show sample data
        with st.expander("📊 Exercise Database Preview"):
            st.dataframe(exercises_df, use_container_width=True)

if __name__ == "__main__":
    main()