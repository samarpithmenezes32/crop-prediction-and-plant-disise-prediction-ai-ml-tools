"""
Theme styles for the Agriculture AI app
"""

def get_theme_css(theme_mode='light'):
    """
    Returns CSS based on theme mode
    """
    if theme_mode == 'dark':
        colors = {
            'bg_gradient': 'linear-gradient(-45deg, #0a0e27, #16213e, #0f3460, #1a1a2e)',
            'card_bg': '#1e2139',
            'text_primary': '#ffffff',
            'text_secondary': '#b0b3b8',
            'border_color': '#3d4043',
            'shadow': '0 8px 30px rgba(0,0,0,0.6)',
            'sidebar_bg': 'linear-gradient(180deg, #1a1d24 0%, #0e1117 100%)',
        }
    else:
        colors = {
            'bg_gradient': 'linear-gradient(-45deg, #e3f2fd, #f3e5f5, #e8f5e9, #fff3e0)',
            'card_bg': '#ffffff',
            'text_primary': '#333333',
            'text_secondary': '#666666',
            'border_color': '#e0e0e0',
            'shadow': '0 8px 30px rgba(0,0,0,0.12)',
            'sidebar_bg': 'linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%)',
        }
    
    return f"""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        /* Global Styles */
        * {{
            font-family: 'Poppins', sans-serif;
        }}
        
        /* Theme Background */
        .stApp {{
            background: {colors['bg_gradient']};
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            color: {colors['text_primary']};
        }}
        
        @keyframes gradientShift {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        
        /* Main Header */
        .main-header {{
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(120deg, #2E7D32, #66BB6A, #81C784);
            background-size: 200% auto;
            color: transparent;
            -webkit-background-clip: text;
            background-clip: text;
            text-align: center;
            margin-bottom: 1rem;
            animation: shine 3s linear infinite, fadeIn 1s ease-in;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        
        @keyframes shine {{
            to {{ background-position: 200% center; }}
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        /* Sub Header */
        .sub-header {{
            font-size: 1.5rem;
            color: {colors['text_secondary']};
            text-align: center;
            margin-bottom: 2rem;
            animation: slideIn 1s ease-out;
        }}
        
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateX(-30px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}
        
        /* Card Container */
        .card-container {{
            background: {colors['card_bg']};
            padding: 2rem;
            border-radius: 20px;
            box-shadow: {colors['shadow']};
            margin: 1rem 0;
            transition: all 0.3s ease;
            animation: slideUp 0.6s ease-out;
            color: {colors['text_primary']};
        }}
        
        .card-container:hover {{
            box-shadow: 0 12px 40px rgba(102,126,234,0.3);
            transform: translateY(-3px);
        }}
        
        @keyframes slideUp {{
            from {{ opacity: 0; transform: translateY(30px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        /* Recommendation Box */
        .recommendation-box {{
            background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
            padding: 1.5rem;
            border-radius: 15px;
            border-left: 5px solid #4CAF50;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            animation: popIn 0.5s ease-out;
            color: #1B5E20;  /* Dark green text for readability */
            font-weight: 500;
        }}
        
        .recommendation-box:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(76,175,80,0.3);
        }}
        
        @keyframes popIn {{
            from {{ opacity: 0; transform: scale(0.9); }}
            to {{ opacity: 1; transform: scale(1); }}
        }}
        
        /* Metric Card */
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 20px;
            color: white;
            text-align: center;
            box-shadow: 0 10px 30px rgba(102,126,234,0.3);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            animation: float 3s ease-in-out infinite;
        }}
        
        .metric-card:hover {{
            transform: scale(1.05) rotate(2deg);
            box-shadow: 0 15px 40px rgba(102,126,234,0.5);
        }}
        
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-10px); }}
        }}
        
        /* Badges */
        .success-badge {{
            background: linear-gradient(135deg, #4CAF50, #8BC34A);
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 25px;
            display: inline-block;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(76,175,80,0.4);
            animation: pulse 2s ease-in-out infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        
        .confidence-badge {{
            background: linear-gradient(90deg, #FF6B6B, #FFE66D, #4ECDC4, #FF6B6B);
            background-size: 300% 100%;
            color: white;
            padding: 0.4rem 1.2rem;
            border-radius: 20px;
            display: inline-block;
            font-weight: 600;
            box-shadow: 0 3px 10px rgba(255,107,107,0.3);
            animation: shimmer 3s linear infinite;
        }}
        
        @keyframes shimmer {{
            0% {{ background-position: 0% 50%; }}
            100% {{ background-position: 300% 50%; }}
        }}
        
        /* Buttons */
        .stButton > button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102,126,234,0.4);
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102,126,234,0.6);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: {colors['card_bg']};
            border-radius: 15px;
            padding: 0.5rem;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            color: {colors['text_primary']};
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: rgba(102,126,234,0.1);
            transform: scale(1.02);
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
        }}
        
        /* Sidebar */
        .css-1d391kg, [data-testid="stSidebar"] {{
            background: {colors['sidebar_bg']};
        }}
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
            color: {colors['text_primary']};
        }}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {colors['border_color']};
            border-radius: 10px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(180deg, #667eea, #764ba2);
            border-radius: 10px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: linear-gradient(180deg, #764ba2, #667eea);
        }}
        
        /* Input Fields */
        .stTextInput > div > div {{
            border-radius: 10px;
            border: 2px solid {colors['border_color']};
            background-color: {colors['card_bg']};
            color: {colors['text_primary']};
            transition: all 0.3s ease;
        }}
        
        .stTextInput > div > div:focus-within {{
            border-color: #667eea;
            box-shadow: 0 0 10px rgba(102,126,234,0.3);
        }}
        
        /* Alerts */
        .stAlert {{
            border-radius: 15px;
            animation: slideIn 0.5s ease-out;
        }}
        
        /* File Uploader */
        [data-testid="stFileUploader"] {{
            border: 2px dashed #667eea;
            border-radius: 15px;
            padding: 2rem;
            background: rgba(102,126,234,0.05);
            transition: all 0.3s ease;
        }}
        
        [data-testid="stFileUploader"]:hover {{
            border-color: #764ba2;
            background: rgba(102,126,234,0.1);
        }}
        
        /* Progress Bar */
        .stProgress > div > div {{
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 10px;
        }}
    </style>
    """
