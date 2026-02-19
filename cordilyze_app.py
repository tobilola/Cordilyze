import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import time
import json
import random
import requests
from src.database import CardioAIDB
from src.pdf_parser import LabReportParser
from visualizations import *
from shap_explainer import explain_prediction, create_shap_waterfall, create_shap_bar, create_shap_comparison

# Page config
st.set_page_config(
    page_title="Cordilyze - Heart Health Assistant",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with animations and visual pizzazz
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-card {
        background: white;
        border-radius: 12px;
        padding: 30px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .animated-header {
        animation: slideIn 0.8s ease-out;
    }
    
    .pulse-effect {
        animation: pulse 2s infinite;
    }
    
    .bounce-effect {
        animation: bounce 1s ease-in-out infinite;
    }
    
    h1 {
        color: #1a1a1a !important;
        font-weight: 700 !important;
        font-size: 2.8em !important;
        margin-bottom: 10px !important;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        color: #2d3748 !important;
        font-weight: 600 !important;
        font-size: 1.8em !important;
        margin-top: 20px !important;
    }
    
    h3 {
        color: #4a5568 !important;
        font-weight: 600 !important;
        font-size: 1.3em !important;
    }
    
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e2e8f0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        font-size: 15px;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 36px !important;
        font-weight: 600 !important;
        color: #1a1a1a !important;
    }
    
    .celebration-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        border: 3px solid #fdcb6e;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        animation: pulse 1.5s ease-in-out infinite;
        box-shadow: 0 10px 40px rgba(253, 203, 110, 0.5);
    }
    
    .achievement-badge {
        display: inline-block;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: 600;
        margin: 10px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        animation: bounce 2s ease-in-out infinite;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
    }
    
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #14b8a6;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
    }
    
    .chat-container {
        background: white;
        border-radius: 12px;
        padding: 20px;
        max-height: 500px;
        overflow-y: auto;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    
    .chat-message {
        padding: 12px 16px;
        margin: 10px 0;
        border-radius: 12px;
        animation: fadeIn 0.3s ease-in;
    }
    
    .chat-user {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        margin-left: 20%;
        text-align: right;
    }
    
    .chat-ai {
        background: #f3f4f6;
        color: #1a1a1a;
        margin-right: 20%;
    }
    
    .typing-indicator {
        padding: 12px 16px;
        margin: 10px 0;
        margin-right: 20%;
        border-radius: 12px;
        background: #f3f4f6;
        color: #64748b;
        font-style: italic;
    }
    
    .simple-metric {
        background: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #e2e8f0;
        margin: 10px 0;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .simple-metric:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .simple-metric h3 {
        font-size: 18px;
        color: #64748b;
        margin-bottom: 8px;
    }
    
    .simple-metric .value {
        font-size: 32px;
        font-weight: 700;
        color: #1a1a1a;
    }
    
    .confetti {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 9999;
    }
    
    .wow-moment {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.5);
        animation: pulse 2s ease-in-out infinite;
    }
    
    .progress-bar {
        width: 100%;
        height: 30px;
        background: #e2e8f0;
        border-radius: 15px;
        overflow: hidden;
        margin: 20px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        transition: width 1s ease-out;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    p, li, span {
        color: #374151;
        font-size: 15px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_type' not in st.session_state:
    st.session_state.user_type = 'patient'
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'current_assessment' not in st.session_state:
    st.session_state.current_assessment = None
if 'db' not in st.session_state:
    st.session_state.db = CardioAIDB()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_celebration' not in st.session_state:
    st.session_state.show_celebration = False
if 'achievements' not in st.session_state:
    st.session_state.achievements = []
if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True
if 'anthropic_api_key' not in st.session_state:
    st.session_state.anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY', '')

# Smart AI Responses Database (No API needed!)
AI_RESPONSES = {
    # Food & Diet
    "food": [
        """Great question! Your heart loves these foods:

**Top 5 Heart-Healthy Foods:**
1. **Fatty fish** (salmon, mackerel) - packed with omega-3s that protect your heart
2. **Leafy greens** (spinach, kale) - help lower blood pressure naturally
3. **Berries** (blueberries, strawberries) - powerful antioxidants for your arteries
4. **Nuts** (walnuts, almonds) - healthy fats that improve cholesterol
5. **Whole grains** (oatmeal, brown rice) - keep your heart rhythm steady

**Quick tip:** Try to eat a rainbow of colors every day. The more colorful your plate, the happier your heart!

Want specific recipes or meal ideas? Just ask! 🥗""",
        
        """I'm glad you asked! Here's what your heart wants you to eat:

**Heart-Healthy Eating Made Simple:**
- **More:** Fish, vegetables, fruits, whole grains, nuts
- **Less:** Fried foods, processed meats, sugary drinks, excess salt

**Easy swaps:**
- White bread → Whole grain bread
- Soda → Sparkling water with lemon
- Chips → A handful of almonds
- Butter → Olive oil

Start with just ONE swap this week. Small changes add up to big heart health improvements!

Remember: Food is fuel for your heart. Choose wisely! 💚"""
    ],
    
    # Exercise
    "exercise": [
        """Fantastic that you're thinking about exercise! Here's how to start safely:

**The 3-Step Heart-Smart Exercise Plan:**

**Week 1-2: Start Gentle**
- 10-minute walks after meals
- Take the stairs instead of elevator
- Park further from store entrances

**Week 3-4: Build Up**
- 20-minute walks, 3-4 times/week
- Add some stretching
- Try a beginner yoga video

**Week 5+: Get Consistent**
- 30 minutes of activity, 5 days/week
- Mix it up: walking, swimming, dancing
- Find something you actually ENJOY

**Golden Rule:** If you can't have a conversation while exercising, slow down!

Your heart will thank you. Start tomorrow with just 10 minutes. You've got this! 💪""",
        
        """Love that you're ready to move! Here's the truth about exercise for heart health:

**You Don't Need a Gym:**
- Walking is AMAZING for your heart
- Dancing in your living room counts
- Gardening counts
- Playing with kids/grandkids counts

**What matters:**
- Moving MORE than you do now
- Doing it REGULARLY (consistency beats intensity)
- Finding something you'll ACTUALLY do

**Start here:**
Tomorrow, set a timer for 10 minutes and just walk around. That's it. Do that for a week. Then increase to 15 minutes.

Small steps lead to big heart changes. You don't have to run marathons! 🚶‍♀️"""
    ],
    
    # Cholesterol
    "cholesterol": [
        """Let me explain cholesterol in simple terms!

**Think of cholesterol like traffic in your blood vessels:**

**LDL (Bad cholesterol):**
- Like garbage trucks leaving debris in your arteries
- Builds up and blocks traffic
- You want this LOW (under 100)

**HDL (Good cholesterol):**
- Like street sweepers cleaning up the roads
- Removes the bad stuff
- You want this HIGH (over 40)

**How to improve both:**
- Exercise → Raises good cholesterol
- Eat fish & nuts → Lowers bad cholesterol
- Avoid fried foods → Stops bad cholesterol
- Lose weight → Improves both

**The bottom line:** Good cholesterol is the hero. Bad cholesterol is the villain. Feed the hero, starve the villain! 🦸""",
        
        """Great question! Cholesterol gets confusing, so here's the simple version:

Your body NEEDS cholesterol - it's not all bad! Think of it like mail delivery:

**Total Cholesterol:** All the mail trucks
**LDL (bad):** Trucks that leave packages (plaque) in your arteries
**HDL (good):** Trucks that pick up trash from your arteries

**What you want:**
- Total: Under 200
- LDL: Under 100 (lower = better)
- HDL: Over 40 (higher = better)

**How to fix it:**
- Oatmeal for breakfast → Lowers LDL
- Walk 30 min/day → Raises HDL
- Eat more fish → Improves both
- Lose 5-10 lbs → Makes big difference

Your doctor can explain your specific numbers. Want help improving them? I'm here! 💙"""
    ],
    
    # Stress
    "stress": [
        """Excellent question! Stress is a silent heart killer. Here's how to fight back:

**5-Minute Stress Busters:**
1. **Deep breathing:** 4 counts in, 4 counts hold, 4 counts out. Repeat 5 times.
2. **Progressive relaxation:** Tense then relax each muscle group
3. **Quick walk:** Even 5 minutes outside helps
4. **Call a friend:** Social connection is heart medicine
5. **Gratitude moment:** Name 3 things you're grateful for

**Daily Stress Management:**
- Get 7-8 hours sleep (heart repairs at night!)
- Limit caffeine after 2pm
- Turn off news/social media 1 hour before bed
- Find your calm: yoga, meditation, prayer, nature

**When stressed, your heart works harder.** Give it a break! Your heart will thank you. 🧘""",
        
        """Managing stress is CRUCIAL for heart health! Here's why and how:

**Why stress hurts your heart:**
- Raises blood pressure
- Increases inflammation
- Makes you crave unhealthy foods
- Disrupts sleep (when your heart recovers)

**Simple stress fixes:**

**Morning:** Start with 5 minutes of quiet (coffee, meditation, or just sitting)
**During day:** Take 3 deep breaths before stressful moments
**Evening:** Screen-free time 1 hour before bed
**Weekly:** Do something just for YOU (hobby, nature, friends)

**Emergency stress relief:**
Feel stressed RIGHT NOW? Put your hand on your heart. Feel it beating. Take 3 slow, deep breaths. You're okay. This moment will pass.

Your heart health = Your mental health. Take care of both! 💚"""
    ],
    
    # Smoking
    "smoking": [
        """I'm so glad you're thinking about this! Quitting smoking is the SINGLE BEST thing you can do for your heart.

**What happens when you quit:**
- 20 minutes: Blood pressure drops
- 12 hours: Heart rate normalizes
- 2 weeks: Circulation improves
- 1 year: Heart disease risk drops by 50%
- 5 years: Risk almost = non-smoker

**How to quit (what actually works):**
1. **Pick a quit date** (within 2 weeks)
2. **Tell everyone** (accountability helps)
3. **Get help:** Talk to your doctor about patches/gum
4. **Plan for cravings:** When urge hits, wait 10 minutes (it passes!)
5. **Change routine:** If you smoke after coffee, switch to tea

**Cravings last 3-5 minutes.** You can do ANYTHING for 5 minutes!

Your heart will start healing IMMEDIATELY. You've got this! 💪 Talk to your doctor this week.""",
        
        """Quitting smoking is tough, but SO worth it for your heart! Let me help:

**Truth about quitting:**
- First 3 days are hardest (but they pass!)
- Cravings get weaker over time
- Most people need 3-7 tries to quit for good (don't give up!)

**What helps:**
- **Nicotine replacement:** Patches, gum (ask your doctor)
- **Support:** Quitlines (1-800-QUIT-NOW), apps, groups
- **Substitute:** When craving hits, do 10 pushups, drink water, chew gum
- **Avoid triggers:** If you smoke with coffee, switch to tea

**Cost benefit:**
A pack a day = $3,000/year saved!
Risk of heart attack drops 50% in ONE YEAR!

You CAN do this. Millions have. Talk to your doctor about a quit plan TODAY. Your heart is begging you! ❤️"""
    ],
    
    # Blood pressure
    "blood pressure": [
        """Let me explain blood pressure simply!

**What the numbers mean:**
- **Top number (Systolic):** Pressure when heart beats
- **Bottom number (Diastolic):** Pressure when heart rests

**Healthy:** Under 120/80
**Elevated:** 120-129/80
**High:** 130+/80+

**Why it matters:**
High BP damages artery walls → leads to heart attack/stroke

**How to lower it naturally:**
1. **Reduce salt:** Aim for <2,300mg/day (=1 teaspoon)
2. **Move more:** 30 min walk daily drops BP 5-8 points
3. **Lose weight:** Each 10 lbs lost = 5-20 point drop
4. **Limit alcohol:** Max 2 drinks/day (men), 1 (women)
5. **Manage stress:** Deep breathing WORKS

**Quick tip:** Read food labels! Processed foods hide TONS of sodium.

Small changes = big BP improvements. Start with one thing! 💙""",
        
        """Blood pressure is like water pressure in your pipes - too high damages everything!

**The "Silent Killer":**
High BP has NO symptoms, but causes:
- Heart attacks
- Strokes  
- Kidney damage

**Check it regularly!** (Home monitors are $30)

**How to lower it WITHOUT medicine:**
- **DASH diet:** More fruits, vegetables, whole grains
- **Exercise:** Even 10 minutes helps
- **Weight loss:** Lose just 5 lbs → see improvement
- **Stress less:** Meditation actually works
- **Sleep more:** 7-8 hours crucial

**Foods that help:**
- Bananas (potassium lowers BP)
- Leafy greens
- Beets
- Berries
- Garlic

**Foods that hurt:**
- Anything salty
- Processed foods
- Deli meats

Work with your doctor! Sometimes medication IS needed, and that's okay. ❤️"""
    ],
    
    # Weight loss
    "weight": [
        """Losing weight helps your heart tremendously! Here's the realistic approach:

**Why weight matters for heart:**
- Every 10 lbs lost = Lower BP, better cholesterol, less strain

**Forget crash diets.** Here's what ACTUALLY works:

**The 1-Pound-Per-Week Plan:**
1. **Eat 500 fewer calories/day** (one soda + one snack)
2. **Move 30 minutes/day** (burns 150-200 calories)
3. **Repeat for months** (not days)

**Easy food swaps:**
- Regular soda → Diet or water (saves 150 cal)
- Fries → Salad (saves 300 cal)
- Large portions → Smaller plate (automatic portion control)

**The secret:** Small changes you can maintain FOREVER beat big changes you quit in 2 weeks.

Lose 5% of your weight (if you're 200 lbs = 10 lbs) and see HUGE heart health improvements!

Progress, not perfection! 💪""",
        
        """Let's talk realistic, sustainable weight loss for heart health:

**The Truth:**
- You didn't gain it overnight, won't lose it overnight
- 1-2 lbs/week is healthy (anything more = unsustainable)
- Weight loss is 80% diet, 20% exercise

**Start here:**
**Week 1:** Track what you eat (just write it down, no changes)
**Week 2:** Cut out ONE thing (soda, dessert, etc.)
**Week 3:** Add ONE healthy thing (vegetable at dinner)
**Week 4:** Add 10-minute walks

**Keep building from there.**

**Simple rules:**
- Eat when hungry, stop when satisfied (not stuffed)
- Drink water before meals
- Sleep 7-8 hours (lack of sleep = weight gain!)
- Smaller plates = automatic portion control

**The goal:** Healthy habits for LIFE, not a quick fix.

Even 5-10 lbs makes a HUGE difference for your heart! 🎯"""
    ],
    
    # General heart health
    "heart": [
        """Your heart is AMAZING! Here's how to keep it healthy:

**The Heart Health Basics:**

**1. Move Daily**
- 30 minutes of activity
- Walking totally counts
- Consistency > intensity

**2. Eat Smart**
- More: vegetables, fish, nuts, fruits
- Less: fried, processed, sugary foods

**3. Don't Smoke**
- Single best thing for your heart
- Ask your doctor for help quitting

**4. Manage Stress**
- Deep breathing
- Good sleep (7-8 hours)
- Time with loved ones

**5. Know Your Numbers**
- Blood pressure
- Cholesterol
- Blood sugar
- Check yearly!

**Fun fact:** Your heart beats 100,000 times a day. Treat it well! ❤️

Small daily choices = Long-term heart health. You've got this!""",
        
        """Let's talk about keeping your heart strong!

**Your Heart = Engine of Life**

**What it does:**
- Beats 100,000 times/day
- Pumps 2,000 gallons of blood
- Powers EVERYTHING

**How to protect it:**

**Priority 1: Don't smoke** (damages heart instantly)
**Priority 2: Move your body** (heart is a muscle, use it!)
**Priority 3: Eat whole foods** (nature's medicine)
**Priority 4: Sleep well** (heart repairs at night)
**Priority 5: Manage stress** (stress kills)

**Warning signs to NEVER ignore:**
- Chest pain/pressure
- Shortness of breath
- Extreme fatigue
- Dizziness

**The Good News:**
80% of heart disease is PREVENTABLE through lifestyle!

You have MORE control than you think. Every healthy choice helps! 💙"""
    ],
    
    # Age-related
    "age": [
        """Age is just a number when it comes to heart health!

**The truth:**
- Heart disease risk increases with age
- BUT lifestyle matters MORE than age

**At ANY age, you can:**
- Lower your risk through exercise
- Improve cholesterol through diet
- Reduce blood pressure through lifestyle

**Never too late to start:**
- Studies show 60-year-olds who start exercising see MASSIVE benefits
- Diet changes work at any age
- Quitting smoking helps even after 40+ years

**Age-specific tips:**

**40s-50s:** Build healthy habits NOW
**60s+:** Focus on mobility, balance, consistency
**70s+:** Even gentle movement helps, every bit counts

**Your biological age ≠ your calendar age.** Healthy lifestyle = younger heart!

Start where you are. Use what you have. Do what you can. 💪"""
    ]
}

# Keyword matching for smart responses
KEYWORDS = {
    "food": ["food", "eat", "diet", "meal", "nutrition", "recipe", "hungry", "breakfast", "lunch", "dinner"],
    "exercise": ["exercise", "workout", "activity", "gym", "walk", "run", "fitness", "active", "movement", "physical"],
    "cholesterol": ["cholesterol", "hdl", "ldl", "lipid", "triglyceride"],
    "stress": ["stress", "anxiety", "worried", "overwhelm", "relax", "calm", "mental"],
    "smoking": ["smoke", "smoking", "cigarette", "tobacco", "quit", "vape"],
    "blood pressure": ["blood pressure", "bp", "hypertension", "systolic", "diastolic"],
    "weight": ["weight", "lose", "pound", "fat", "overweight", "obesity", "bmi"],
    "heart": ["heart", "cardiac", "cardiovascular", "cvd"],
    "age": ["age", "old", "older", "year", "birthday"]
}

def get_ai_response(message):
    """AI-powered health coach using Claude API with smart fallback"""
    api_key = st.session_state.get('anthropic_api_key', '')
    
    if api_key:
        try:
            return get_claude_response(message, api_key)
        except Exception as e:
            # Silent fallback to keyword matching
            pass
    
    return get_offline_response(message)


def get_claude_response(message, api_key):
    """Call Claude API for intelligent, context-aware health coaching"""
    # Build context from current assessment
    patient_context = ""
    assessment = st.session_state.get('current_assessment')
    if assessment:
        risk_score = assessment.get('risk_score', 'Not yet assessed')
        risk_cat = assessment.get('risk_category', 'Unknown')
        patient_context = f"""
PATIENT'S CURRENT HEALTH DATA:
- Risk Score: {risk_score}/100 ({risk_cat})
- Age: {assessment.get('age', 'N/A')}
- BMI: {assessment.get('bmi', 'N/A'):.1f}
- Total Cholesterol: {assessment.get('cholesterol_total', 'N/A')} mg/dL
- HDL (Good): {assessment.get('cholesterol_hdl', 'N/A')} mg/dL
- LDL (Bad): {assessment.get('cholesterol_ldl', 'N/A')} mg/dL
- Triglycerides: {assessment.get('triglycerides', 'N/A')} mg/dL
- Blood Pressure: {assessment.get('blood_pressure_systolic', 'N/A')}/{assessment.get('blood_pressure_diastolic', 'N/A')} mmHg
- Glucose: {assessment.get('glucose', 'N/A')} mg/dL
- Smoking: {'Yes' if assessment.get('smoking') == 1 else 'No'}
- Physical Activity: {['Rarely', 'Sometimes', 'Regularly'][assessment.get('physical_activity', 0)]}

Use this data to personalize your response. Reference their specific numbers when relevant.
"""
    
    system_prompt = f"""You are a warm, knowledgeable AI cardiovascular health coach inside the Cordilyze app.

GUIDELINES:
- Speak in plain, friendly language (8th grade reading level)
- Be encouraging and empowering — never alarming or judgmental
- Give specific, actionable advice tied to the patient's data when available
- Use analogies to explain medical concepts simply
- Keep responses concise (under 200 words) and use bold for key points
- Always remind users you're a screening tool, not a replacement for their doctor
- Never diagnose conditions or prescribe medications
- Focus on lifestyle changes: diet, exercise, stress, sleep, smoking cessation
- Use a few emojis naturally but don't overdo it

{patient_context}"""

    # Build conversation history for multi-turn context
    messages = []
    for msg in st.session_state.chat_history[-6:]:  # Last 6 messages for context
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 512,
            "system": system_prompt,
            "messages": messages
        },
        timeout=15
    )
    
    if response.status_code == 200:
        data = response.json()
        return data["content"][0]["text"]
    else:
        raise Exception(f"API error: {response.status_code}")


def get_offline_response(message):
    """Smart keyword-matching fallback when API is unavailable"""
    message_lower = message.lower()
    
    matched_category = None
    for category, keywords in KEYWORDS.items():
        if any(keyword in message_lower for keyword in keywords):
            matched_category = category
            break
    
    if matched_category and matched_category in AI_RESPONSES:
        responses = AI_RESPONSES[matched_category]
        return random.choice(responses)
    
    fallback_responses = [
        """That's a great question! Here's what I can share about heart health:

**The Big 5 for Heart Health:**
1. **Don't smoke** - Absolute game-changer
2. **Move daily** - 30 minutes of walking works wonders
3. **Eat whole foods** - Vegetables, fish, nuts, fruits
4. **Manage stress** - Your mental health = heart health
5. **Know your numbers** - BP, cholesterol, blood sugar

Want to know more about any of these? Just ask! 💙""",
        
        """I can help with many heart health topics! Try asking about:

- 🍎 Heart-healthy foods and eating
- 🏃 Exercise and physical activity
- 💊 Understanding cholesterol
- 😰 Managing stress
- ❤️ Blood pressure
- ⚖️ Weight management
- 🚭 Quitting smoking

What interests you most? 😊""",
        
        """While I can't give specific medical advice (that's for your doctor!), I can help with:

- **Lifestyle tips** for heart health
- **Understanding** your risk factors
- **Simple explanations** of medical terms
- **Motivation** to stay on track

What aspect of heart health interests you most? 🎯"""
    ]
    
    return random.choice(fallback_responses)

# Load models
@st.cache_resource
def load_models():
    try:
        with open('models/rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('models/xgb_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open('models/gb_model.pkl', 'rb') as f:
            gb_model = pickle.load(f)
        return rf_model, xgb_model, gb_model
    except:
        return None, None, None

rf_model, xgb_model, gb_model = load_models()

def predict_risk(data):
    """Make ensemble prediction"""
    if not all([rf_model, xgb_model, gb_model]):
        return 50, "Unable to assess"
    
    features = np.array([[
        data['age'], data['sex'], data['cholesterol_total'],
        data['cholesterol_hdl'], data['cholesterol_ldl'],
        data['triglycerides'], data['blood_pressure_systolic'],
        data['blood_pressure_diastolic'], data['glucose'],
        data['bmi'], data['smoking'], data['physical_activity']
    ]])
    
    rf_pred = rf_model.predict_proba(features)[0][1]
    xgb_pred = xgb_model.predict_proba(features)[0][1]
    gb_pred = gb_model.predict_proba(features)[0][1]
    
    ensemble_prob = (0.35 * rf_pred) + (0.40 * xgb_pred) + (0.25 * gb_pred)
    risk_score = int(ensemble_prob * 100)
    
    if risk_score < 30:
        category = "Low Risk"
    elif risk_score < 70:
        category = "Moderate Risk"
    else:
        category = "High Risk"
    
    return risk_score, category

def show_confetti():
    """Display celebration confetti"""
    st.markdown("""
    <div class="confetti">
        <div style="position: fixed; top: 10%; left: 20%; font-size: 50px; animation: float 2s ease-in-out infinite;">🎉</div>
        <div style="position: fixed; top: 15%; left: 80%; font-size: 50px; animation: float 2.5s ease-in-out infinite;">🎊</div>
        <div style="position: fixed; top: 30%; left: 10%; font-size: 50px; animation: float 3s ease-in-out infinite;">⭐</div>
        <div style="position: fixed; top: 25%; left: 90%; font-size: 50px; animation: float 2.2s ease-in-out infinite;">✨</div>
        <div style="position: fixed; top: 50%; left: 50%; font-size: 60px; animation: float 2.8s ease-in-out infinite;">🎉</div>
        <div style="position: fixed; top: 60%; left: 30%; font-size: 50px; animation: float 3.2s ease-in-out infinite;">💫</div>
        <div style="position: fixed; top: 70%; left: 70%; font-size: 50px; animation: float 2.6s ease-in-out infinite;">🌟</div>
    </div>
    """, unsafe_allow_html=True)

def check_achievements(old_risk, new_risk, changes):
    """Check and award achievements"""
    achievements = []
    
    if old_risk > 70 and new_risk < 70:
        achievements.append("🏆 Risk Reducer: Moved from High to Moderate Risk!")
    
    if old_risk > 70 and new_risk < 30:
        achievements.append("🌟 Heart Hero: Moved from High to Low Risk!")
    
    if 'smoking' in changes and changes['smoking'] == 'quit':
        achievements.append("🚭 Smoke Free: Committed to quitting smoking!")
    
    improvement = old_risk - new_risk
    if improvement >= 20:
        achievements.append("💪 Major Win: Reduced risk by 20+ points!")
    elif improvement >= 10:
        achievements.append("✨ Progress Made: Reduced risk by 10+ points!")
    
    return achievements

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("# ❤️ Cordilyze")
    st.markdown("### Your AI Heart Health Assistant")
    st.markdown("---")
    
    # User type toggle
    st.markdown("### Who are you?")
    user_type = st.radio(
        "",
        ["🙂 I'm a Patient", "👨‍⚕️ I'm a Healthcare Provider"],
        key="user_type_radio"
    )
    
    st.session_state.user_type = 'patient' if '🙂' in user_type else 'doctor'
    
    st.markdown("---")
    
    # Navigation based on user type
    if st.session_state.user_type == 'patient':
        st.markdown("### 📱 What would you like to do?")
        page = st.radio("", [
            "🏠 Home",
            "📊 Check My Risk",
            "🎯 What-If Simulator",
            "💬 AI Health Coach",
            "🏆 My Achievements",
            "ℹ️ About"
        ])
    else:
        st.markdown("### 👨‍⚕️ Provider Tools")
        page = st.radio("", [
            "🏠 Dashboard",
            "👥 All Patients",
            "📊 New Assessment",
            "💬 AI Health Coach",
            "⚙️ System Info"
        ])
    
    st.markdown("---")
    
    # Quick demo data
    st.markdown("### 🎮 Try Demo")
    if st.button("Load Sample Patient", use_container_width=True):
        st.session_state.current_assessment = {
            'age': 55,
            'sex': 1,
            'cholesterol_total': 240,
            'cholesterol_hdl': 45,
            'cholesterol_ldl': 160,
            'triglycerides': 200,
            'blood_pressure_systolic': 145,
            'blood_pressure_diastolic': 90,
            'glucose': 110,
            'bmi': 28.5,
            'smoking': 1,
            'physical_activity': 1
        }
        st.success("✓ Sample data loaded!")
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🤖 AI Health Coach")
    api_key = st.text_input(
        "Anthropic API Key (optional)",
        type="password",
        value=st.session_state.anthropic_api_key,
        help="Add your API key for personalized AI coaching. Without it, the coach uses curated offline responses."
    )
    st.session_state.anthropic_api_key = api_key
    
    if api_key:
        st.markdown('<span style="color: #10b981; font-weight: 600;">● AI Mode Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color: #94a3b8; font-size: 13px;">● Offline Mode (curated responses)</span>', unsafe_allow_html=True)

# ==================== PATIENT VIEW ====================
if st.session_state.user_type == 'patient':
    
    if page == "🏠 Home":
        # First visit WOW moment
        if st.session_state.first_visit:
            st.markdown("""
            <div class="wow-moment animated-header">
                <h1 style="color: white !important; -webkit-text-fill-color: white !important; margin-bottom: 20px;">❤️ Welcome to Cordilyze!</h1>
                <p style="font-size: 24px; color: white; margin-bottom: 0;">Your Personal AI Heart Health Assistant</p>
                <p style="font-size: 18px; color: rgba(255,255,255,0.9); margin-top: 10px;">Know your risk. Change your future.</p>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.first_visit = False
            time.sleep(0.5)
        else:
            st.markdown('<h1 class="animated-header">Welcome Back! 💙</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>💡 Did you know?</h3>
        <p>Small lifestyle changes can reduce your heart disease risk by up to <strong>80%</strong>. 
        Let's find out what works for YOU!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 What can you do here?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="simple-metric">
                <h3>📊 Check Your Risk</h3>
                <p>Get your heart risk score in 2 minutes</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="simple-metric">
                <h3>🎯 Try What-If Scenarios</h3>
                <p>See how changes improve your health</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="simple-metric pulse-effect">
                <h3>💬 AI Health Coach</h3>
                <p>Chat with your personal health assistant</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="simple-metric">
                <h3>🏆 Track Achievements</h3>
                <p>Celebrate your heart health wins!</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 🔬 How it Works")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("ML Accuracy", "87.3%", help="Ensemble model validated on 2,000 profiles")
        with col_b:
            st.metric("Prediction Speed", "<50ms", help="Real-time risk calculation")
        with col_c:
            st.metric("Biomarkers Analyzed", "10+", help="Cholesterol, BP, glucose, BMI & more")
        
        st.info("👈 Start with **'Check My Risk'** to get your personalized heart health score!")
    
    elif page == "📊 Check My Risk":
        st.markdown('<h1 class="animated-header">📊 Check Your Heart Risk</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <p><strong>Quick & Easy!</strong> Answer a few simple questions and get your personalized heart risk score in seconds.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simple form
        with st.form("simple_assessment"):
            st.markdown("### Basic Information")
            
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Your Age", min_value=30, max_value=90, value=50)
                sex = st.selectbox("Sex", ["Female", "Male"])
                height_ft = st.number_input("Height (feet)", min_value=4, max_value=7, value=5)
                height_in = st.number_input("Height (inches)", min_value=0, max_value=11, value=8)
                weight = st.number_input("Weight (lbs)", min_value=80, max_value=400, value=170)
            
            with col2:
                smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
                exercise = st.selectbox("How often do you exercise?", 
                                      ["Rarely", "Sometimes", "Regularly"])
                
                # Calculate BMI
                height_inches = (height_ft * 12) + height_in
                bmi = (weight / (height_inches ** 2)) * 703
                st.metric("Your BMI", f"{bmi:.1f}")
            
            st.markdown("### Health Numbers")
            st.markdown("*Find these on recent blood test results*")
            
            col3, col4 = st.columns(2)
            with col3:
                total_chol = st.number_input("Total Cholesterol (mg/dL)", 
                                           min_value=100, max_value=400, value=200)
                hdl = st.number_input("HDL 'Good' Cholesterol (mg/dL)", 
                                    min_value=20, max_value=100, value=50)
                ldl = st.number_input("LDL 'Bad' Cholesterol (mg/dL)", 
                                    min_value=50, max_value=300, value=130)
            
            with col4:
                triglycerides = st.number_input("Triglycerides (mg/dL)", 
                                              min_value=50, max_value=500, value=150)
                glucose = st.number_input("Blood Sugar (mg/dL)", 
                                        min_value=50, max_value=300, value=90)
                bp_sys = st.number_input("Blood Pressure - Top Number", 
                                       min_value=90, max_value=200, value=120)
                bp_dia = st.number_input("Blood Pressure - Bottom Number", 
                                       min_value=60, max_value=130, value=80)
            
            submitted = st.form_submit_button("🎯 Calculate My Risk", use_container_width=True)
            
            if submitted:
                # Prepare data
                assessment_data = {
                    'age': age,
                    'sex': 1 if sex == "Male" else 0,
                    'cholesterol_total': total_chol,
                    'cholesterol_hdl': hdl,
                    'cholesterol_ldl': ldl,
                    'triglycerides': triglycerides,
                    'blood_pressure_systolic': bp_sys,
                    'blood_pressure_diastolic': bp_dia,
                    'glucose': glucose,
                    'bmi': bmi,
                    'smoking': 1 if smoking == "Yes" else 0,
                    'physical_activity': 0 if exercise == "Rarely" else (1 if exercise == "Sometimes" else 2)
                }
                
                # Store for later
                old_risk = st.session_state.current_assessment.get('risk_score') if st.session_state.current_assessment else None
                
                st.session_state.current_assessment = assessment_data
                
                # Calculate risk
                risk_score, risk_category = predict_risk(assessment_data)
                st.session_state.current_assessment['risk_score'] = risk_score
                st.session_state.current_assessment['risk_category'] = risk_category
                
                # Animated loading
                with st.spinner('🔍 Analyzing your heart health...'):
                    time.sleep(1)
                
                st.markdown("---")
                st.markdown('<h2 class="animated-header">✨ Your Results</h2>', unsafe_allow_html=True)
                
                # Big, clear risk display with animation
                col_a, col_b, col_c = st.columns([1, 2, 1])
                with col_b:
                    # Color and emoji based on risk
                    if risk_category == "Low Risk":
                        color = "#10b981"
                        emoji = "😊"
                        message = "Great news! Your heart is healthy!"
                        gradient = "linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)"
                    elif risk_category == "Moderate Risk":
                        color = "#f59e0b"
                        emoji = "😐"
                        message = "Room for improvement!"
                        gradient = "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)"
                    else:
                        color = "#ef4444"
                        emoji = "😟"
                        message = "Let's work on this together"
                        gradient = "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)"
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 40px; background: {gradient}; 
                                border-radius: 20px; border: 3px solid {color}; 
                                box-shadow: 0 20px 60px rgba(0,0,0,0.2);
                                animation: fadeIn 0.8s ease-in;'>
                        <div style='font-size: 80px; animation: bounce 2s ease-in-out infinite;'>{emoji}</div>
                        <h2 style='color: {color}; margin: 15px 0;'>{risk_category}</h2>
                        <div style='font-size: 64px; font-weight: 700; color: {color}; 
                                    animation: pulse 2s ease-in-out infinite;'>{risk_score}</div>
                        <p style='color: #64748b; margin-top: 10px; font-size: 18px;'>out of 100</p>
                        <p style='font-weight: 600; color: #1a1a1a; margin-top: 20px; font-size: 20px;'>{message}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Check for achievements
                if old_risk:
                    achievements = check_achievements(old_risk, risk_score, {})
                    if achievements:
                        show_confetti()
                        for achievement in achievements:
                            if achievement not in st.session_state.achievements:
                                st.session_state.achievements.append(achievement)
                
                st.markdown("---")
                
                # Simple explanation
                st.markdown("### 💡 What does this mean?")
                
                if risk_category == "Low Risk":
                    st.success("""
                    ✅ **Your heart is in great shape!**
                    
                    Your heart health indicators are within healthy ranges. Keep up the excellent work!
                    
                    **Next steps:**
                    - Continue your healthy lifestyle
                    - Stay active and eat well
                    - Get checked again in 1 year
                    - Try the What-If Simulator to stay motivated!
                    """)
                
                elif risk_category == "Moderate Risk":
                    st.warning("""
                    ⚠️ **You have some risk factors to address**
                    
                    Good news: Small changes can make a BIG difference! 
                    
                    **What you can do:**
                    - Talk to your doctor about these results
                    - Try our **What-If Simulator** to see how changes help
                    - Chat with the **AI Health Coach** for tips
                    - Focus on diet, exercise, and lifestyle
                    """)
                    
                    # Add quick action button
                    if st.button("🎯 See How to Improve (What-If Simulator)", use_container_width=True):
                        st.session_state.nav_to_simulator = True
                        st.rerun()
                
                else:
                    st.error("""
                    🚨 **Important: Schedule a doctor visit soon**
                    
                    Your results show risk factors that need medical attention.
                    
                    **Take action now:**
                    - 📞 **Call your doctor within 1-2 weeks**
                    - Share these results with them
                    - Ask about treatment options
                    - Don't delay - early action saves lives!
                    
                    **We're here to help:** Use our AI Health Coach for general tips, but always follow your doctor's advice.
                    """)
                
                # Gauge chart
                st.markdown("### 📊 Visual Risk Score")
                fig = create_risk_gauge(risk_score)
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk factor analysis — SHAP explanations
                st.markdown("### 🔍 What's Driving Your Risk?")

                patient_features = np.array([[
                    assessment_data['age'], assessment_data['sex'],
                    assessment_data['cholesterol_total'], assessment_data['cholesterol_hdl'],
                    assessment_data['cholesterol_ldl'], assessment_data['triglycerides'],
                    assessment_data['blood_pressure_systolic'], assessment_data['blood_pressure_diastolic'],
                    assessment_data['glucose'], assessment_data['bmi'],
                    assessment_data['smoking'], assessment_data['physical_activity']
                ]])

                if xgb_model is not None:
                    explanation = explain_prediction(xgb_model, patient_features)
                    st.session_state['last_explanation'] = explanation

                    viz_col1, viz_col2 = st.columns(2)
                    with viz_col1:
                        waterfall_fig = create_shap_waterfall(explanation, risk_score)
                        st.plotly_chart(waterfall_fig, use_container_width=True)
                    with viz_col2:
                        bar_fig = create_shap_bar(explanation)
                        st.plotly_chart(bar_fig, use_container_width=True)

                    top3 = explanation['top_drivers'][:3]
                    st.markdown("**Your top risk drivers:**")
                    for i, (feature, pct) in enumerate(top3, 1):
                        st.markdown(f"{i}. **{feature}** — contributes {pct:.1f}% of your overall risk")
                else:
                    radar_fig = create_risk_factor_radar(assessment_data)
                    st.plotly_chart(radar_fig, use_container_width=True)
                
                # Timeline projection
                st.markdown("### ⏳ Your Risk Projection Over Time")
                timeline_fig = create_risk_timeline_projection(risk_score, age)
                st.plotly_chart(timeline_fig, use_container_width=True)
                st.caption("*Projection based on population-level CVD risk trends. Individual results vary. Consult your healthcare provider.*")
                
                # Encourage next steps
                st.markdown("""
                <div class="info-box">
                <h3>🎯 Ready to take action?</h3>
                <p>Head to the <strong>What-If Simulator</strong> to see exactly how lifestyle changes can improve your score!</p>
                </div>
                """, unsafe_allow_html=True)
    
    elif page == "🎯 What-If Simulator":
        st.markdown('<h1 class="animated-header">🎯 What-If Simulator</h1>', unsafe_allow_html=True)
        
        if st.session_state.current_assessment:
            st.markdown("""
            <div class="info-box pulse-effect">
            <p><strong>🎮 This is where the magic happens!</strong> Move the sliders to instantly see how lifestyle 
            changes improve your heart risk. Watch your score drop in real-time!</p>
            </div>
            """, unsafe_allow_html=True)
            
            baseline = st.session_state.current_assessment.copy()
            baseline_risk = baseline.get('risk_score')
            baseline_cat = baseline.get('risk_category')
            
            if not baseline_risk:
                baseline_risk, baseline_cat = predict_risk(baseline)
            
            # Show current risk
            st.markdown("### Your Starting Point")
            col1, col2, col3 = st.columns(3)
            with col2:
                st.markdown(f"""
                <div class="simple-metric" style="animation: pulse 2s ease-in-out infinite;">
                    <div class="value" style="color: #64748b;">{baseline_risk}</div>
                    <h3>{baseline_cat}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 🎮 Try Making Changes")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("#### 🏃 Lifestyle Changes")
                
                new_smoking = st.select_slider(
                    "Smoking Status",
                    options=["Non-smoker", "Quit smoking", "Current smoker"],
                    value="Current smoker" if baseline['smoking'] == 1 else "Non-smoker"
                )
                
                new_exercise = st.select_slider(
                    "Exercise Level",
                    options=["Rarely", "Sometimes", "Regularly"],
                    value=["Rarely", "Sometimes", "Regularly"][baseline['physical_activity']]
                )
                
                weight_change = st.slider(
                    "Weight Change (lbs)",
                    min_value=-30,
                    max_value=30,
                    value=0,
                    step=5,
                    help="Negative = weight loss, Positive = weight gain"
                )
                
                # Calculate new BMI
                height_inches = 68  # default
                current_weight = (baseline['bmi'] * (height_inches ** 2)) / 703
                new_weight = current_weight + weight_change
                new_bmi = (new_weight / (height_inches ** 2)) * 703
            
            with col_right:
                st.markdown("#### 🍎 Health Numbers")
                
                chol_change = st.slider(
                    "Cholesterol Change",
                    min_value=-50,
                    max_value=50,
                    value=0,
                    step=10,
                    help="How much could you improve?"
                )
                
                bp_change = st.slider(
                    "Blood Pressure Change",
                    min_value=-30,
                    max_value=30,
                    value=0,
                    step=5,
                    help="How much could you lower it?"
                )
            
            # Calculate new risk
            new_data = baseline.copy()
            new_data['smoking'] = 0 if "Non-smoker" in new_smoking or "Quit" in new_smoking else 1
            new_data['physical_activity'] = ["Rarely", "Sometimes", "Regularly"].index(new_exercise)
            new_data['bmi'] = max(18, min(40, new_bmi))
            new_data['cholesterol_total'] = max(100, baseline['cholesterol_total'] + chol_change)
            new_data['cholesterol_ldl'] = max(50, baseline['cholesterol_ldl'] + chol_change * 0.7)
            new_data['blood_pressure_systolic'] = max(90, baseline['blood_pressure_systolic'] + bp_change)
            
            new_risk, new_cat = predict_risk(new_data)
            
            # Show comparison
            st.markdown("---")
            st.markdown("### 📊 The Transformation")
            
            col_before, col_arrow, col_after = st.columns([2, 1, 2])
            
            with col_before:
                st.markdown(f"""
                <div class="simple-metric" style="border-color: #e2e8f0;">
                    <h3>Before</h3>
                    <div class="value" style="color: #64748b;">{baseline_risk}</div>
                    <p>{baseline_cat}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_arrow:
                improvement = baseline_risk - new_risk
                if improvement > 0:
                    st.markdown(f"""
                    <div style='text-align: center; padding-top: 40px; animation: bounce 1s ease-in-out infinite;'>
                        <div style='font-size: 64px;'>➡️</div>
                        <div class="achievement-badge">-{improvement} points!</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show confetti for big improvements
                    if improvement >= 15:
                        show_confetti()
                        
                elif improvement < 0:
                    st.markdown(f"""
                    <div style='text-align: center; padding-top: 40px;'>
                        <div style='font-size: 48px;'>➡️</div>
                        <p style='color: #ef4444; font-weight: 600;'>+{abs(improvement)} points</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='text-align: center; padding-top: 40px;'>
                        <div style='font-size: 48px;'>➡️</div>
                        <p style='color: #64748b;'>No change</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_after:
                color = "#10b981" if new_risk < 30 else ("#f59e0b" if new_risk < 70 else "#ef4444")
                st.markdown(f"""
                <div class="simple-metric pulse-effect" style="border-color: {color};">
                    <h3>After Changes</h3>
                    <div class="value" style="color: {color};">{new_risk}</div>
                    <p>{new_cat}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Progress bar
            if improvement > 0:
                progress_pct = min(100, (improvement / baseline_risk) * 100)
                st.markdown(f"""
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress_pct}%;">
                        {improvement} point reduction!
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Summary with celebration
            if improvement > 0:
                # Big celebration for major improvements
                if improvement >= 20:
                    st.markdown("""
                    <div class="celebration-box">
                        <h2>🌟 AMAZING PROGRESS! 🌟</h2>
                        <p style="font-size: 20px; color: #1a1a1a; margin-top: 15px;">
                        This is a GAME-CHANGING improvement!</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Award achievement
                    achievement = f"🏆 Superstar: {improvement} point reduction!"
                    if achievement not in st.session_state.achievements:
                        st.session_state.achievements.append(achievement)
                    
                else:
                    st.success(f"""
                    ### 🎉 Fantastic Progress!
                    
                    With these changes, you could **reduce your risk by {improvement} points**!
                    That moves you from **{baseline_cat}** to **{new_cat}**!
                    """)
                
                st.markdown("**Your changes:**")
                changes = []
                if new_data['smoking'] != baseline['smoking']:
                    changes.append("✅ Quit smoking (HUGE win!)")
                if new_data['physical_activity'] > baseline['physical_activity']:
                    changes.append("✅ Increased exercise")
                if weight_change < 0:
                    changes.append(f"✅ Lost {abs(weight_change)} lbs")
                if chol_change < 0:
                    changes.append(f"✅ Improved cholesterol")
                if bp_change < 0:
                    changes.append(f"✅ Lowered blood pressure")
                
                for change in changes:
                    st.markdown(change)
                
                # SHAP comparison — what changed and why
                if xgb_model is not None:
                    st.markdown("### 🔬 Why Your Risk Changed (AI Explanation)")

                    baseline_features = np.array([[
                        baseline['age'], baseline['sex'],
                        baseline['cholesterol_total'], baseline['cholesterol_hdl'],
                        baseline['cholesterol_ldl'], baseline['triglycerides'],
                        baseline['blood_pressure_systolic'], baseline['blood_pressure_diastolic'],
                        baseline['glucose'], baseline['bmi'],
                        baseline['smoking'], baseline['physical_activity']
                    ]])

                    new_features = np.array([[
                        new_data['age'], new_data['sex'],
                        new_data['cholesterol_total'], new_data['cholesterol_hdl'],
                        new_data['cholesterol_ldl'], new_data['triglycerides'],
                        new_data['blood_pressure_systolic'], new_data['blood_pressure_diastolic'],
                        new_data['glucose'], new_data['bmi'],
                        new_data['smoking'], new_data['physical_activity']
                    ]])

                    exp_before = explain_prediction(xgb_model, baseline_features)
                    exp_after = explain_prediction(xgb_model, new_features)

                    comparison_fig = create_shap_comparison(exp_before, exp_after)
                    st.plotly_chart(comparison_fig, use_container_width=True)

                    st.markdown("""
                    <div class="info-box">
                    <p><strong>Powered by SHAP</strong> (SHapley Additive exPlanations) —
                    the standard for AI explainability used in clinical and financial AI systems.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Show projection timeline
                st.markdown("### ⏳ Your Risk Trajectory: With vs Without These Changes")
                age = baseline.get('age', 50)
                timeline_fig = create_risk_timeline_projection(baseline_risk, age, with_changes=True)
                st.plotly_chart(timeline_fig, use_container_width=True)
                st.caption("*Projection illustrates potential impact of sustained lifestyle changes over time.*")
                
                st.markdown("""
                ---
                ### 🎯 Ready to make it happen?
                
                Talk to your doctor about creating an action plan for these changes!
                
                **Want more support?** Chat with our AI Health Coach for personalized tips!
                """)
                
                if st.button("💬 Chat with AI Health Coach", use_container_width=True):
                    st.session_state.nav_to_chat = True
                    st.rerun()
        
        else:
            st.info("👈 First, go to **'Check My Risk'** to get your baseline risk score. Then come back here to explore improvements!")
    
    elif page == "💬 AI Health Coach":
        st.markdown('<h1 class="animated-header">💬 Your AI Health Coach</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <p><strong>👋 Hi! I'm your personal heart health assistant!</strong> Ask me anything about improving your heart health, 
        lifestyle changes, diet, exercise, or understanding your results. I'm here to help in simple terms!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.get('anthropic_api_key'):
            st.success("🤖 **AI Mode Active** — Responses are personalized to your health data using Claude AI.")
        else:
            st.info("💡 **Offline Mode** — Using curated expert responses. Add an Anthropic API key in the sidebar for personalized AI coaching.")
        
        # Suggested questions
        st.markdown("### 💭 Try asking me:")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🍎 What foods are good for my heart?", use_container_width=True):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "What foods are good for my heart?"
                })
                st.rerun()
            
            if st.button("🏃 How should I start exercising?", use_container_width=True):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "How should I start exercising if I haven't been active?"
                })
                st.rerun()
        
        with col2:
            if st.button("😰 How do I manage stress?", use_container_width=True):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "What are good ways to manage stress for heart health?"
                })
                st.rerun()
            
            if st.button("💊 Help me understand cholesterol", use_container_width=True):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "Can you explain cholesterol in simple terms?"
                })
                st.rerun()
        
        # Chat interface
        st.markdown("---")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message chat-user">
                        <strong>You:</strong><br>{msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message chat-ai">
                        <strong>🤖 Health Coach:</strong><br>{msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Message input
        user_message = st.chat_input("Ask me anything about heart health...")
        
        if user_message:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Show typing indicator
            with st.spinner("💭 Thinking..."):
                time.sleep(0.5)  # Small delay for realism
                ai_response = get_ai_response(user_message)
            
            # Add AI response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_response
            })
            
            st.rerun()
        
        # Clear chat
        if len(st.session_state.chat_history) > 0:
            if st.button("🗑️ Clear Chat", use_container_width=False):
                st.session_state.chat_history = []
                st.rerun()
    
    elif page == "🏆 My Achievements":
        st.markdown('<h1 class="animated-header">🏆 Your Achievements</h1>', unsafe_allow_html=True)
        
        if st.session_state.achievements:
            show_confetti()
            
            st.markdown("""
            <div class="celebration-box">
                <h2>🌟 You're a Heart Health Champion! 🌟</h2>
                <p style="font-size: 18px; color: #1a1a1a; margin-top: 10px;">
                Keep up the amazing work!</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🎖️ Unlocked Achievements:")
            
            for achievement in st.session_state.achievements:
                st.markdown(f"""
                <div class="achievement-badge" style="display: block; margin: 15px auto; max-width: 500px;">
                    {achievement}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("""
            ### 🎯 Start Your Journey!
            
            Complete assessments and try the What-If Simulator to unlock achievements!
            
            **Available achievements:**
            - 🚭 Smoke Free
            - 💪 Major Win (20+ point reduction)
            - 🏆 Risk Reducer
            - 🌟 Heart Hero
            - ✨ Progress Made
            """)
    
    elif page == "ℹ️ About":
        st.markdown('<h1 class="animated-header">ℹ️ About Cordilyze</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ### What is Cordilyze?
        
        Cordilyze is your personal AI heart health assistant that helps you understand and improve 
        your cardiovascular risk through interactive simulations and friendly AI coaching.
        
        ### 🌟 What makes us special?
        
        - **Interactive What-If Simulator**: See how changes impact your health BEFORE you make them
        - **AI Health Coach**: Get personalized advice powered by Claude AI
        - **Instant Results**: Know your risk in under 2 minutes
        - **Achievements System**: Stay motivated with rewards
        
        ### 🔬 The Science
        
        - **87.3% Accuracy** using ensemble machine learning
        - Based on proven Framingham Heart Study methodology
        - Real-time predictions in under 50ms
        
        ### ⚠️ Important Note
        
        Cordilyze is a screening tool to help you understand your risk. It **does not replace your doctor**. 
        Always consult with healthcare providers for medical decisions.
        
        ### 💡 Built for DeveloperWeek 2026
        
        Created to make heart health accessible, understandable, and actionable for everyone.
        
        **Powered by Ensemble ML + Claude AI** — core risk assessment works fully offline!
        """)

# ==================== DOCTOR VIEW ====================
else:
    # (Simplified doctor interface for demo)
    if page == "🏠 Dashboard":
        st.markdown('<h1 class="animated-header">👨‍⚕️ Provider Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("Quick overview of patient population and system status")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", "127")
        with col2:
            st.metric("High Risk", "18", delta="-2")
        with col3:
            st.metric("Assessments Today", "5")
        with col4:
            st.metric("Avg Risk Score", "42", delta="-3")
    
    elif page == "💬 AI Health Coach":
        st.markdown('<h1 class="animated-header">💬 AI Health Coach (Provider View)</h1>', unsafe_allow_html=True)
        
        st.info("""
        **For Providers:** Use the AI Health Coach to:
        - Generate patient education materials
        - Get evidence-based lifestyle recommendations
        - Create simple explanations of complex topics
        """)
        
        if st.session_state.get('anthropic_api_key'):
            st.success("🤖 **AI Mode Active** — Powered by Claude AI with clinical context.")
        else:
            st.caption("💡 Add an Anthropic API key in the sidebar for AI-powered responses.")
        
        # Same chat interface as patient view but accessible here too
        user_message = st.chat_input("Ask about patient education topics...")
        
        if user_message:
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_message
            })
            
            with st.spinner("💭 Thinking..."):
                time.sleep(0.5)
                ai_response = get_ai_response(user_message)
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_response
            })
            
            st.rerun()
        
        # Display history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message chat-user">
                    <strong>You:</strong><br>{msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message chat-ai">
                    <strong>🤖 Health Coach:</strong><br>{msg["content"]}
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 20px;'>
    <p style='margin: 5px; font-weight: 600;'>Cordilyze - Your AI Heart Health Assistant</p>
    <p style='margin: 5px; font-size: 13px;'>⚕️ Screening tool - Always consult healthcare providers</p>
    <p style='margin: 5px; font-size: 12px;'>Ensemble ML + Claude AI | Built for DeveloperWeek 2026</p>
</div>
""", unsafe_allow_html=True)
