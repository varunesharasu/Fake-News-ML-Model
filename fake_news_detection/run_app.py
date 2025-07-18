import os
import sys
import subprocess

def setup_and_run():
    """Setup and run the Flask application"""
    
    print("🚀 Starting Fake News Detector Web App...")
    
    # Check if models exist
    model_path = 'models/saved_models/lightgbm_model.pkl'
    if not os.path.exists(model_path):
        print("❌ Model files not found!")
        print("Please run the main training script first:")
        print("python main_optimized.py")
        return
    
    print("✅ Model files found!")
    
    # Install web requirements if needed
    try:
        import flask
        print("✅ Flask is installed")
    except ImportError:
        print("📦 Installing web requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_web.txt"])
    
    # Run the Flask app
    print("🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    
    from app import app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    setup_and_run()
