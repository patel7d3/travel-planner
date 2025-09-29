"""
Enhanced Travel Itinerary Concierge - Detailed & Complete
Balanced speed with comprehensive details, proper origin tracking, visual calendar

Features:
- Full origin/destination tracking
- Detailed itineraries with context
- Visual calendar timeline
- Rich activity descriptions
- Better balance of speed vs detail
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Initialize OpenAI
try:
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error("⚠️ OPENAI_API_KEY not found in .env file")
        st.stop()
    
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"Error initializing OpenAI: {str(e)}")
    st.stop()


class EnhancedTravelPlannerAI:
    """Enhanced AI travel planner with better detail"""
    
    _cache = {}
    
    @staticmethod
    def _get_cache_key(func_name: str, *args) -> str:
        """Generate cache key"""
        key = f"{func_name}_{'_'.join(map(str, args))}"
        return hashlib.md5(key.encode()).hexdigest()
    
    @staticmethod
    def get_destination_insights(destination: str) -> Dict[str, Any]:
        """Get comprehensive destination insights"""
        cache_key = EnhancedTravelPlannerAI._get_cache_key("insights", destination)
        
        if cache_key in EnhancedTravelPlannerAI._cache:
            return EnhancedTravelPlannerAI._cache[cache_key]
        
        try:
            prompt = f"""Provide detailed travel insights for {destination} as JSON:

{{
  "description": "2-3 sentence overview of what makes this destination special",
  "best_time_to_visit": "detailed explanation with months and reasons",
  "average_daily_budget": {{
    "budget": 60,
    "mid_range": 150,
    "luxury": 400
  }},
  "top_attractions": [
    {{"name": "attraction", "description": "why visit", "time_needed": "2-3 hours", "cost": 15}},
    {{"name": "attraction2", "description": "why visit", "time_needed": "half day", "cost": 0}}
  ],
  "local_cuisine": [
    {{"dish": "name", "description": "brief", "where": "type of place"}},
    {{"dish": "name2", "description": "brief", "where": "type of place"}}
  ],
  "cultural_tips": [
    "important cultural insight 1",
    "important cultural insight 2",
    "important cultural insight 3"
  ],
  "safety_info": {{"rating": 8, "notes": "specific safety tips"}},
  "weather_by_season": {{"spring": "...", "summer": "...", "fall": "...", "winter": "..."}},
  "transportation": {{"getting_around": "detailed transit info", "from_airport": "how to get from airport to city"}},
  "language_tips": ["useful phrase 1", "useful phrase 2"],
  "currency": "currency name and exchange tips",
  "neighborhoods": [
    {{"name": "neighborhood", "vibe": "description", "best_for": "what to do here"}}
  ]
}}

Be thorough and practical."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert travel guide providing detailed, accurate information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            EnhancedTravelPlannerAI._cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Insights error: {e}")
            return {}
    
    @staticmethod
    def create_daily_itinerary(destination: str, origin: str, days: int, 
                              preferences: List[str], budget_level: str,
                              start_date: str) -> List[Dict]:
        """Generate detailed daily itinerary"""
        try:
            prefs_str = ', '.join(preferences) if preferences else 'general sightseeing'
            
            prompt = f"""Create a detailed {days}-day itinerary for {destination}.

Trip details:
- Starting from: {origin}
- Preferences: {prefs_str}
- Budget: {budget_level}
- Start date: {start_date}

For EACH day provide detailed JSON:
[{{
  "day": 1,
  "date": "{start_date}",
  "title": "Descriptive day theme (e.g., Historic Heart & Local Flavors)",
  "morning": [
    {{
      "time": "8:00 AM",
      "activity": "Breakfast at...",
      "description": "Why this is great, what to expect (2-3 sentences)",
      "duration": "1 hour",
      "cost": 15,
      "location": "neighborhood/address",
      "tips": "insider tip"
    }},
    {{
      "time": "9:30 AM",
      "activity": "Visit Main Attraction",
      "description": "Detailed explanation of what you'll see and experience",
      "duration": "2-3 hours",
      "cost": 25,
      "location": "specific area",
      "tips": "best time to visit, what to bring"
    }}
  ],
  "afternoon": [
    {{
      "time": "1:00 PM",
      "activity": "Lunch suggestion",
      "description": "What to try, atmosphere",
      "duration": "1.5 hours",
      "cost": 20,
      "location": "area",
      "tips": "reservation tips"
    }},
    {{
      "time": "3:00 PM",
      "activity": "Afternoon activity",
      "description": "Full experience description",
      "duration": "2 hours",
      "cost": 30,
      "location": "where",
      "tips": "practical advice"
    }}
  ],
  "evening": [
    {{
      "time": "7:00 PM",
      "activity": "Dinner & evening plans",
      "description": "Evening experience details",
      "duration": "2-3 hours",
      "cost": 50,
      "location": "area",
      "tips": "what to wear, reservations"
    }}
  ],
  "transportation": "How to get around this day (metro lines, walking routes, etc)",
  "total_cost": 140,
  "energy_level": "moderate",
  "weather_considerations": "what to prepare for",
  "flexibility_note": "optional activities if time permits"
}}]

Make it realistic with proper timing, real locations, and practical advice. 
Day 1 should include arrival from {origin}.
Last day should account for departure logistics.
Each activity should have meaningful descriptions, not generic statements."""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional travel planner creating detailed, realistic itineraries with specific recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=500 * days,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            itinerary = result.get("itinerary", result.get("days", []))
            
            # Ensure dates are properly set
            for i, day in enumerate(itinerary):
                day_date = datetime.fromisoformat(start_date) + timedelta(days=i)
                day['date'] = day_date.strftime('%Y-%m-%d')
                day['day_of_week'] = day_date.strftime('%A')
            
            return itinerary
        except Exception as e:
            logger.error(f"Itinerary error: {e}")
            return []
    
    @staticmethod
    def get_packing_list(destination: str, days: int, season: str,
                        activities: List[str]) -> Dict[str, List[str]]:
        """Detailed packing list"""
        cache_key = EnhancedTravelPlannerAI._get_cache_key("packing", destination, season, str(activities))
        
        if cache_key in EnhancedTravelPlannerAI._cache:
            return EnhancedTravelPlannerAI._cache[cache_key]
        
        try:
            prompt = f"""Create comprehensive packing list for {destination} in {season}, {days} days.
Activities: {', '.join(activities)}

JSON format:
{{
  "documents": ["item with reason"],
  "clothing": ["specific items for weather/activities"],
  "footwear": ["what shoes and why"],
  "toiletries": ["essentials"],
  "electronics": ["device + accessories"],
  "medications": ["health items"],
  "accessories": ["bags, sunglasses, etc"],
  "activity_specific": ["gear for activities"],
  "optional": ["nice to have items"]
}}

Be specific about quantities and reasons (e.g., "Light rain jacket - afternoon showers common")"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=600,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            EnhancedTravelPlannerAI._cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Packing error: {e}")
            return {}
    
    @staticmethod
    def get_budget_breakdown(destination: str, days: int, travelers: int,
                           budget_level: str) -> Dict[str, Any]:
        """Detailed budget estimates"""
        try:
            prompt = f"""Create detailed budget breakdown for {travelers} traveler(s) in {destination} for {days} days ({budget_level} level).

JSON format:
{{
  "accommodation": {{
    "per_night": 0,
    "total_nights": {days},
    "total": 0,
    "notes": "type of accommodation"
  }},
  "food": {{
    "breakfast_avg": 0,
    "lunch_avg": 0,
    "dinner_avg": 0,
    "daily_total": 0,
    "trip_total": 0
  }},
  "transportation": {{
    "airport_transfer": 0,
    "daily_local": 0,
    "total": 0,
    "notes": "what's included"
  }},
  "activities": {{
    "daily_avg": 0,
    "total": 0,
    "notes": "typical costs"
  }},
  "shopping": {{
    "budget": 0,
    "notes": "souvenirs and extras"
  }},
  "emergency_fund": 0,
  "total_per_person": 0,
  "total_all_travelers": 0,
  "daily_average": 0,
  "savings_tips": ["tip 1", "tip 2"]
}}

Provide realistic estimates with context."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Budget error: {e}")
            return {}
    
    @staticmethod
    def generate_all_parallel(origin: str, destination: str, days: int, 
                            travelers: int, preferences: List[str], 
                            budget_level: str, season: str,
                            start_date: str) -> Dict[str, Any]:
        """Run all AI calls in parallel"""
        results = {
            'insights': None,
            'itinerary': None,
            'budget': None,
            'packing': None,
            'errors': []
        }
        
        def run_insights():
            try:
                return ('insights', EnhancedTravelPlannerAI.get_destination_insights(destination))
            except Exception as e:
                return ('insights', {'error': str(e)})
        
        def run_itinerary():
            try:
                return ('itinerary', EnhancedTravelPlannerAI.create_daily_itinerary(
                    destination, origin, days, preferences, budget_level, start_date
                ))
            except Exception as e:
                return ('itinerary', {'error': str(e)})
        
        def run_budget():
            try:
                return ('budget', EnhancedTravelPlannerAI.get_budget_breakdown(
                    destination, days, travelers, budget_level
                ))
            except Exception as e:
                return ('budget', {'error': str(e)})
        
        def run_packing():
            try:
                return ('packing', EnhancedTravelPlannerAI.get_packing_list(
                    destination, days, season, preferences
                ))
            except Exception as e:
                return ('packing', {'error': str(e)})
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(run_insights),
                executor.submit(run_itinerary),
                executor.submit(run_budget),
                executor.submit(run_packing)
            ]
            
            for future in as_completed(futures):
                try:
                    key, value = future.result()
                    results[key] = value
                except Exception as e:
                    results['errors'].append(str(e))
        
        return results


# ==================== Streamlit App ====================

def init_session_state():
    if 'trip_plan' not in st.session_state:
        st.session_state.trip_plan = None


def render_calendar_view(itinerary: List[Dict], start_date: str):
    """Render visual calendar timeline"""
    st.markdown("### 📅 Trip Timeline")
    
    for day in itinerary:
        day_num = day.get('day', 0)
        date_str = day.get('date', start_date)
        day_of_week = day.get('day_of_week', '')
        title = day.get('title', f'Day {day_num}')
        
        # Calendar card for each day
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 10px; margin: 1rem 0; color: white;'>
            <h3 style='margin: 0; color: white;'>Day {day_num} - {day_of_week}</h3>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>{date_str} • {title}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Timeline for the day
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("**🌅 Morning**")
            for act in day.get('morning', []):
                with st.container():
                    st.markdown(f"**{act.get('time', '')}** - {act.get('activity', '')}")
                    if act.get('description'):
                        st.caption(act['description'])
                    st.caption(f"📍 {act.get('location', '')} • ⏱️ {act.get('duration', '')} • 💵 ${act.get('cost', 0)}")
                    if act.get('tips'):
                        st.info(f"💡 {act['tips']}")
                    st.markdown("---")
        
        with col2:
            st.markdown("**☀️ Afternoon**")
            for act in day.get('afternoon', []):
                with st.container():
                    st.markdown(f"**{act.get('time', '')}** - {act.get('activity', '')}")
                    if act.get('description'):
                        st.caption(act['description'])
                    st.caption(f"📍 {act.get('location', '')} • ⏱️ {act.get('duration', '')} • 💵 ${act.get('cost', 0)}")
                    if act.get('tips'):
                        st.info(f"💡 {act['tips']}")
                    st.markdown("---")
        
        with col3:
            st.markdown("**🌙 Evening**")
            for act in day.get('evening', []):
                with st.container():
                    st.markdown(f"**{act.get('time', '')}** - {act.get('activity', '')}")
                    if act.get('description'):
                        st.caption(act['description'])
                    st.caption(f"📍 {act.get('location', '')} • ⏱️ {act.get('duration', '')} • 💵 ${act.get('cost', 0)}")
                    if act.get('tips'):
                        st.info(f"💡 {act['tips']}")
                    st.markdown("---")
        
        # Day summary
        st.markdown(f"""
        **🚇 Getting Around:** {day.get('transportation', 'Walk/Metro')}  
        **💰 Daily Budget:** ${day.get('total_cost', 0)}  
        **⚡ Energy Level:** {day.get('energy_level', 'Moderate')}
        """)
        
        if day.get('weather_considerations'):
            st.caption(f"☁️ Weather: {day['weather_considerations']}")
        
        if day.get('flexibility_note'):
            st.caption(f"✨ Optional: {day['flexibility_note']}")
        
        st.markdown("<br>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="AI Travel Planner",
        page_icon="✈️",
        layout="wide"
    )
    
    init_session_state()
    
    # Enhanced CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .route-display {
            text-align: center;
            font-size: 1.5rem;
            margin: 1rem 0;
            color: #667eea;
            font-weight: bold;
        }
        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
            border: none;
            padding: 0.75rem;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">✈️ AI Travel Planner</h1>', unsafe_allow_html=True)
    
    # Sidebar form
    with st.sidebar:
        st.header("Plan Your Trip")
        
        with st.form("trip_form"):
            origin = st.text_input("From*", placeholder="New York, London, Tokyo...")
            destination = st.text_input("To*", placeholder="Paris, Bali, Rome...")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start", min_value=datetime.now())
            with col2:
                end_date = st.date_input("End", min_value=datetime.now() + timedelta(days=1))
            
            col3, col4 = st.columns(2)
            with col3:
                travelers = st.number_input("Travelers", 1, 10, 2)
            with col4:
                budget = st.selectbox("Budget", ["Budget", "Mid-Range", "Luxury"], index=1)
            
            prefs = st.multiselect(
                "Interests",
                ["Culture", "Adventure", "Food", "Relaxation", "Shopping", "Nature", "Photography"],
                default=["Culture"]
            )
            
            submit = st.form_submit_button("🚀 Generate Itinerary", use_container_width=True)
        
        if submit:
            if not origin or not destination:
                st.error("Please enter both origin and destination")
            elif end_date <= start_date:
                st.error("End date must be after start date")
            else:
                days = (end_date - start_date).days
                
                if days > 14:
                    st.warning(f"⚠️ {days} days is quite long - generation may take 45-60 seconds")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()
                
                try:
                    status_text.text("🤖 Analyzing destination...")
                    progress_bar.progress(20)
                    
                    season = ["winter", "spring", "summer", "fall"][(start_date.month % 12) // 3]
                    start_date_str = start_date.strftime('%Y-%m-%d')
                    
                    status_text.text("⚡ Generating detailed itinerary...")
                    progress_bar.progress(40)
                    
                    results = EnhancedTravelPlannerAI.generate_all_parallel(
                        origin, destination, days, travelers, 
                        prefs or ["Culture"], budget.lower(), season, start_date_str
                    )
                    
                    progress_bar.progress(90)
                    status_text.text("✨ Finalizing your plan...")
                    
                    st.session_state.trip_plan = {
                        "origin": origin,
                        "destination": destination,
                        "dates": f"{start_date.strftime('%B %d, %Y')} - {end_date.strftime('%B %d, %Y')}",
                        "start_date": start_date_str,
                        "days": days,
                        "travelers": travelers,
                        "budget_level": budget,
                        "preferences": prefs,
                        "insights": results.get('insights', {}),
                        "itinerary": results.get('itinerary', []),
                        "budget": results.get('budget', {}),
                        "packing": results.get('packing', {})
                    }
                    
                    elapsed = time.time() - start_time
                    progress_bar.progress(100)
                    status_text.text(f"✅ Complete in {elapsed:.1f}s!")
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
    
    # Main content
    if st.session_state.trip_plan:
        plan = st.session_state.trip_plan
        
        # Route display with origin
        st.markdown(
            f'<div class="route-display">✈️ {plan["origin"]} → {plan["destination"]}</div>',
            unsafe_allow_html=True
        )
        
        # Trip stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", f"{plan['days']} Days")
        with col2:
            st.metric("Travelers", plan['travelers'])
        with col3:
            st.metric("Budget", plan['budget_level'])
        with col4:
            total = plan.get('budget', {}).get('total_all_travelers', 0)
            if total:
                st.metric("Total Cost", f"${total:,.0f}")
        
        st.caption(f"📅 {plan['dates']}")
        
        # Enhanced tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗺️ Destination Guide", "📅 Daily Itinerary", 
            "💰 Budget Details", "🧳 Packing List", "📝 Summary"
        ])
        
        with tab1:
            insights = plan.get('insights', {})
            
            if insights.get('description'):
                st.info(insights['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                if insights.get('best_time_to_visit'):
                    st.subheader("🌤️ Best Time to Visit")
                    st.write(insights['best_time_to_visit'])
                
                if insights.get('top_attractions'):
                    st.subheader("🏛️ Top Attractions")
                    for attr in insights['top_attractions']:
                        with st.expander(f"📍 {attr.get('name', '')}"):
                            st.write(attr.get('description', ''))
                            st.caption(f"⏱️ {attr.get('time_needed', '')} • 💵 ${attr.get('cost', 0)}")
                
                if insights.get('neighborhoods'):
                    st.subheader("🏘️ Neighborhoods")
                    for n in insights['neighborhoods']:
                        st.markdown(f"**{n.get('name')}** - {n.get('vibe')}")
                        st.caption(f"Best for: {n.get('best_for')}")
            
            with col2:
                if insights.get('local_cuisine'):
                    st.subheader("🍽️ Must-Try Food")
                    for food in insights['local_cuisine']:
                        st.markdown(f"**{food.get('dish')}**")
                        st.write(food.get('description', ''))
                        st.caption(f"Where: {food.get('where', '')}")
                
                if insights.get('transportation'):
                    st.subheader("🚇 Transportation")
                    trans = insights['transportation']
                    st.write(f"**Getting Around:** {trans.get('getting_around', '')}")
                    st.write(f"**From Airport:** {trans.get('from_airport', '')}")
                
                if insights.get('cultural_tips'):
                    st.subheader("💡 Cultural Tips")
                    for tip in insights['cultural_tips']:
                        st.info(tip)
                
                if insights.get('safety_info'):
                    safety = insights['safety_info']
                    st.subheader("🛡️ Safety")
                    st.write(f"Rating: {safety.get('rating', 'N/A')}/10")
                    st.caption(safety.get('notes', ''))
        
        with tab2:
            render_calendar_view(plan.get('itinerary', []), plan.get('start_date', ''))
        
        with tab3:
            st.subheader("💵 Detailed Budget Breakdown")
            budget = plan.get('budget', {})
            
            if budget:
                # Accommodation
                if 'accommodation' in budget:
                    acc = budget['accommodation']
                    st.markdown("### 🏨 Accommodation")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Per Night", f"${acc.get('per_night', 0)}")
                    col2.metric("Nights", acc.get('total_nights', 0))
                    col3.metric("Total", f"${acc.get('total', 0):,.0f}")
                    if acc.get('notes'):
                        st.caption(acc['notes'])
                
                # Food
                if 'food' in budget:
                    food = budget['food']
                    st.markdown("### 🍽️ Food & Dining")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Breakfast", f"${food.get('breakfast_avg', 0)}")
                    col2.metric("Lunch", f"${food.get('lunch_avg', 0)}")
                    col3.metric("Dinner", f"${food.get('dinner_avg', 0)}")
                    col4.metric("Trip Total", f"${food.get('trip_total', 0):,.0f}")
                
                # Transportation
                if 'transportation' in budget:
                    trans = budget['transportation']
                    st.markdown("### 🚇 Transportation")
                    col1, col2 = st.columns(2)
                    col1.metric("Daily Local", f"${trans.get('daily_local', 0)}")
                    col2.metric("Total", f"${trans.get('total', 0):,.0f}")
                    if trans.get('notes'):
                        st.caption(trans['notes'])
                
                # Activities
                if 'activities' in budget:
                    act = budget['activities']
                    st.markdown("### 🎭 Activities")
                    col1, col2 = st.columns(2)
                    col1.metric("Daily Average", f"${act.get('daily_avg', 0)}")
                    col2.metric("Total", f"${act.get('total', 0):,.0f}")
                
                # Summary
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("Per Person", f"${budget.get('total_per_person', 0):,.0f}")
                col2.metric("All Travelers", f"${budget.get('total_all_travelers', 0):,.0f}")
                col3.metric("Daily Average", f"${budget.get('daily_average', 0):,.0f}")
                
                if budget.get('savings_tips'):
                    st.markdown("### 💡 Money-Saving Tips")
                    for tip in budget['savings_tips']:
                        st.success(tip)
        
        with tab4:
            st.subheader("🧳 Complete Packing List")
            packing = plan.get('packing', {})
            
            if packing:
                cols = st.columns(2)
                for idx, (category, items) in enumerate(packing.items()):
                    with cols[idx % 2]:
                        st.markdown(f"### {category.replace('_', ' ').title()}")
                        for item in items:
                            st.checkbox(item, key=f"pack_{idx}_{item}")
                        st.markdown("---")
        
        with tab5:
            st.subheader("📋 Trip Summary")
            
            # Quick overview
            st.markdown(f"""
            **Route:** {plan['origin']} → {plan['destination']}  
            **Dates:** {plan['dates']}  
            **Duration:** {plan['days']} days  
            **Travelers:** {plan['travelers']} people  
            **Budget Level:** {plan['budget_level']}  
            **Interests:** {', '.join(plan['preferences'])}
            """)
            
            # Highlights
            st.markdown("### ✨ Trip Highlights")
            itinerary = plan.get('itinerary', [])
            if itinerary:
                for day in itinerary:
                    st.markdown(f"**Day {day.get('day')}:** {day.get('title', 'Explore')}")
            
            # Download
            st.markdown("---")
            st.download_button(
                "📥 Download Complete Itinerary (JSON)",
                data=json.dumps(plan, indent=2),
                file_name=f"trip_{plan['origin']}_{plan['destination']}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
            
            # Share
            st.markdown("### 📤 Share Your Itinerary")
            st.info("Copy this summary to share with travel companions!")
            
            share_text = f"""
🌍 Trip to {plan['destination']}
📅 {plan['dates']}
✈️ From {plan['origin']}
👥 {plan['travelers']} travelers
💰 {plan['budget_level']} budget

Daily Highlights:
"""
            for day in itinerary[:5]:  # First 5 days
                share_text += f"\nDay {day.get('day')}: {day.get('title', 'Explore')}"
            
            st.code(share_text, language=None)
    
    else:
        # Welcome screen
        st.markdown("### 👋 Welcome to AI Travel Planner")
        st.info("👈 Fill out the form in the sidebar to create your personalized travel itinerary!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### ⚡ Fast & Detailed")
            st.write("Get comprehensive itineraries in 15-30 seconds with hour-by-hour planning")
        with col2:
            st.markdown("#### 🎯 AI-Powered")
            st.write("Smart recommendations based on your preferences and budget")
        with col3:
            st.markdown("#### 📍 Real Places")
            st.write("Actual attractions, restaurants, and local tips from AI knowledge")
        
        st.markdown("---")
        st.markdown("### 🌟 What You'll Get:")
        st.markdown("""
        - **Detailed Daily Itineraries** - Hour-by-hour plans with descriptions, costs, and insider tips
        - **Visual Calendar View** - See your entire trip at a glance
        - **Destination Guide** - Learn about attractions, food, culture, and safety
        - **Complete Budget Breakdown** - Know exactly what to expect to spend
        - **Smart Packing List** - Don't forget anything important
        - **Downloadable Plans** - Take your itinerary offline
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Pro Tips:")
        st.markdown("""
        - **Origin matters** - We'll plan your arrival and departure logistics
        - **Pick 2-3 interests** - More focused = better recommendations  
        - **Budget level** - This affects accommodation, dining, and activity choices
        - **Trip length** - 3-7 days works best, longer trips take more time to generate
        """)


if __name__ == "__main__":
    main()