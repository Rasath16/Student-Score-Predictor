import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, scaler, feature_names

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/Cleaned_student_data.csv')
    return df

# Initialize
try:
    model, scaler, feature_names = load_model()
    df = load_data()
    model_loaded = True
except:
    model_loaded = False
    st.error("⚠️ Model files not found! Please run the training notebook first.")

# Header
st.title("🎓 Student Performance Prediction System")
st.markdown("### Predict student exam scores based on various factors")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/graduation-cap.png", width=100)
    st.title("Navigation")
    page = st.radio("Go to", ["🏠 Home", "📊 Data Explorer", "📈 Model Performance", "🎯 Make Prediction"])
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This app predicts student exam scores using machine learning based on study habits, attendance, and other factors.")
    
    if model_loaded:
        st.success("✅ Model loaded successfully!")
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# Main content
if not model_loaded:
    st.warning("Please ensure the model files are in the 'models/' directory.")
    st.stop()

# Page 1: Home
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Students", f"{len(df):,}")
    with col2:
        st.metric("Average Score", f"{df['Exam_Score'].mean():.2f}")
    with col3:
        st.metric("Features Used", len(feature_names))
    
    st.markdown("---")
    
    # Dataset overview
    st.subheader("📋 Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dataset Information:**")
        st.write(f"- Total Records: {len(df):,}")
        st.write(f"- Number of Features: {len(df.columns)}")
        st.write(f"- Exam Score Range: {df['Exam_Score'].min()} - {df['Exam_Score'].max()}")
        st.write(f"- Mean Score: {df['Exam_Score'].mean():.2f}")
        st.write(f"- Standard Deviation: {df['Exam_Score'].std():.2f}")
    
    with col2:
        st.markdown("**Key Features:**")
        st.write("- 📚 Hours Studied")
        st.write("- 🎯 Attendance")
        st.write("- 😴 Sleep Hours")
        st.write("- 📝 Previous Scores")
        st.write("- 👨‍👩‍👧‍👦 Parental Involvement")
        st.write("- 💻 Internet Access")
        st.write("- And more...")
    
    st.markdown("---")
    
    # Score distribution
    st.subheader("📊 Exam Score Distribution")
    fig = px.histogram(df, x='Exam_Score', nbins=30, 
                      title='Distribution of Exam Scores',
                      labels={'Exam_Score': 'Exam Score', 'count': 'Frequency'},
                      color_discrete_sequence=['#1f77b4'])
    fig.add_vline(x=df['Exam_Score'].mean(), line_dash="dash", line_color="red",
                 annotation_text=f"Mean: {df['Exam_Score'].mean():.2f}")
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Page 2: Data Explorer
elif page == "📊 Data Explorer":
    st.header("📊 Data Explorer")
    
    tab1, tab2, tab3 = st.tabs(["📈 Numerical Features", "📋 Categorical Features", "🔗 Correlations"])
    
    # Tab 1: Numerical Features
    with tab1:
        numerical_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 
                         'Tutoring_Sessions', 'Physical_Activity']
        
        selected_feature = st.selectbox("Select a feature to analyze:", numerical_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(df, x=selected_feature, nbins=30,
                             title=f'Distribution of {selected_feature}',
                             color_discrete_sequence=['#2ecc71'])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(df, y=selected_feature,
                        title=f'Box Plot of {selected_feature}',
                        color_discrete_sequence=['#e74c3c'])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot with Exam Score
        st.subheader(f"{selected_feature} vs Exam Score")
        fig = px.scatter(df, x=selected_feature, y='Exam_Score',
                        trendline="ols", trendline_color_override="red",
                        title=f'{selected_feature} vs Exam Score',
                        labels={selected_feature: selected_feature, 'Exam_Score': 'Exam Score'})
        fig.update_traces(marker=dict(size=5, opacity=0.6))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{df[selected_feature].mean():.2f}")
        with col2:
            st.metric("Median", f"{df[selected_feature].median():.2f}")
        with col3:
            st.metric("Min", f"{df[selected_feature].min():.2f}")
        with col4:
            st.metric("Max", f"{df[selected_feature].max():.2f}")
    
    # Tab 2: Categorical Features
    with tab2:
        categorical_cols = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level',
                           'Internet_Access', 'School_Type', 'Gender', 'Learning_Disabilities']
        
        selected_cat = st.selectbox("Select a categorical feature:", categorical_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Count plot
            value_counts = df[selected_cat].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        title=f'Distribution of {selected_cat}',
                        labels={'x': selected_cat, 'y': 'Count'},
                        color_discrete_sequence=['#9b59b6'])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie chart
            fig = px.pie(values=value_counts.values, names=value_counts.index,
                        title=f'Proportion of {selected_cat}')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Box plot by category
        st.subheader(f"Exam Score by {selected_cat}")
        fig = px.box(df, x=selected_cat, y='Exam_Score',
                    title=f'Exam Score Distribution by {selected_cat}',
                    color=selected_cat)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Correlations
    with tab3:
        st.subheader("🔗 Feature Correlations with Exam Score")
        
        numerical_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 
                         'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score']
        
        corr_matrix = df[numerical_cols].corr()
        
        # Heatmap
        fig = px.imshow(corr_matrix, 
                       text_auto='.2f',
                       aspect="auto",
                       color_continuous_scale='RdBu_r',
                       title='Correlation Heatmap')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation with Exam Score
        st.subheader("Correlation with Exam Score")
        target_corr = corr_matrix['Exam_Score'].drop('Exam_Score').sort_values(ascending=True)
        
        fig = px.bar(x=target_corr.values, y=target_corr.index, orientation='h',
                    title='Feature Correlation with Exam Score',
                    labels={'x': 'Correlation Coefficient', 'y': 'Feature'},
                    color=target_corr.values,
                    color_continuous_scale='RdYlGn')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Page 3: Model Performance
elif page == "📈 Model Performance":
    st.header("📈 Model Performance")
    
    # Load model metrics (you can update these from your notebook results)
    st.info("📝 Note: Update these metrics from your notebook results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", "0.XXXX", help="Coefficient of determination")
    with col2:
        st.metric("RMSE", "X.XXXX", help="Root Mean Squared Error")
    with col3:
        st.metric("MAE", "X.XXXX", help="Mean Absolute Error")
    
    st.markdown("---")
    
    # Feature importance (from your notebook)
    st.subheader("🔑 Feature Importance")
    st.info("Feature importance shows which factors have the most impact on exam scores.")
    
    # Placeholder - you can load this from your model
    st.write("Run the training notebook to see feature importance visualization here.")
    
    st.markdown("---")
    
    # Model explanation
    st.subheader("📚 Model Explanation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**How the model works:**")
        st.write("""
        1. Takes student information as input
        2. Processes and scales the features
        3. Uses trained regression model
        4. Predicts exam score (0-100)
        """)
    
    with col2:
        st.markdown("**Key factors affecting scores:**")
        st.write("""
        - 📚 Study hours and consistency
        - 🎯 Attendance rate
        - 📝 Previous academic performance
        - 😴 Sleep and health habits
        - 👨‍👩‍👧‍👦 Family support and involvement
        """)

# Page 4: Make Prediction
elif page == "🎯 Make Prediction":
    st.header("🎯 Make Prediction")
    st.markdown("Enter student information to predict their exam score")
    
    # Create input form
    with st.form("prediction_form"):
        st.subheader("📝 Student Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hours_studied = st.slider("Hours Studied per Week", 0, 40, 20)
            attendance = st.slider("Attendance %", 0, 100, 80)
            sleep_hours = st.slider("Sleep Hours per Day", 4, 10, 7)
            previous_scores = st.slider("Previous Scores", 0, 100, 70)
        
        with col2:
            tutoring_sessions = st.slider("Tutoring Sessions per Month", 0, 8, 2)
            physical_activity = st.slider("Physical Activity Hours/Week", 0, 6, 3)
            parental_involvement = st.selectbox("Parental Involvement", ['Low', 'Medium', 'High'])
            access_resources = st.selectbox("Access to Resources", ['Low', 'Medium', 'High'])
        
        with col3:
            motivation = st.selectbox("Motivation Level", ['Low', 'Medium', 'High'])
            internet_access = st.selectbox("Internet Access", ['Yes', 'No'])
            school_type = st.selectbox("School Type", ['Public', 'Private'])
            gender = st.selectbox("Gender", ['Male', 'Female'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            extracurricular = st.selectbox("Extracurricular Activities", ['Yes', 'No'])
        with col2:
            learning_disabilities = st.selectbox("Learning Disabilities", ['Yes', 'No'])
        with col3:
            family_income = st.selectbox("Family Income", ['Low', 'Medium', 'High'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            teacher_quality = st.selectbox("Teacher Quality", ['Low', 'Medium', 'High'])
        with col2:
            peer_influence = st.selectbox("Peer Influence", ['Negative', 'Neutral', 'Positive'])
        with col3:
            parental_education = st.selectbox("Parental Education Level", 
                                             ['High School', 'College', 'Postgraduate'])
        
        distance_home = st.selectbox("Distance from Home", ['Near', 'Moderate', 'Far'])
        
        submitted = st.form_submit_button("🎯 Predict Exam Score", use_container_width=True)
        
        if submitted:
            # Create input dataframe
            input_data = pd.DataFrame({
                'Hours_Studied': [hours_studied],
                'Attendance': [attendance],
                'Parental_Involvement': [parental_involvement],
                'Access_to_Resources': [access_resources],
                'Extracurricular_Activities': [extracurricular],
                'Sleep_Hours': [sleep_hours],
                'Previous_Scores': [previous_scores],
                'Motivation_Level': [motivation],
                'Internet_Access': [internet_access],
                'Tutoring_Sessions': [tutoring_sessions],
                'Family_Income': [family_income],
                'Teacher_Quality': [teacher_quality],
                'School_Type': [school_type],
                'Peer_Influence': [peer_influence],
                'Physical_Activity': [physical_activity],
                'Learning_Disabilities': [learning_disabilities],
                'Parental_Education_Level': [parental_education],
                'Distance_from_Home': [distance_home],
                'Gender': [gender]
            })
            
            # Encode categorical variables
            input_encoded = pd.get_dummies(input_data)
            
            # Align with training features
            for col in feature_names:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            input_encoded = input_encoded[feature_names]
            
            # Scale features
            input_scaled = scaler.transform(input_encoded)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display result
            st.markdown("---")
            st.subheader("📊 Prediction Result")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Predicted Exam Score", 'font': {'size': 24}},
                    delta={'reference': df['Exam_Score'].mean(), 'increasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': '#ffcccc'},
                            {'range': [40, 60], 'color': '#ffffcc'},
                            {'range': [60, 80], 'color': '#ccffcc'},
                            {'range': [80, 100], 'color': '#ccffff'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance category
            if prediction >= 80:
                category = "🌟 Excellent"
                color = "green"
            elif prediction >= 60:
                category = "✅ Good"
                color = "blue"
            elif prediction >= 40:
                category = "⚠️ Average"
                color = "orange"
            else:
                category = "❌ Needs Improvement"
                color = "red"
            
            st.markdown(f"### Performance Category: :{color}[{category}]")
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            recommendations = []
            
            if hours_studied < 15:
                recommendations.append("📚 Increase study hours to at least 15-20 hours per week")
            if attendance < 75:
                recommendations.append("🎯 Improve attendance - aim for at least 80%")
            if sleep_hours < 6:
                recommendations.append("😴 Get more sleep - 7-8 hours recommended")
            if tutoring_sessions < 2:
                recommendations.append("👨‍🏫 Consider additional tutoring sessions")
            if motivation == 'Low':
                recommendations.append("🔥 Work on motivation and goal-setting")
            
            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success("✅ Great job! Keep up the good work!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Student Performance Prediction System | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)