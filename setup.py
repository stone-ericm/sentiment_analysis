#!/usr/bin/env python3
"""
Setup script for the Sentiment Analysis Project
This script automates the setup process for new users.
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_virtual_environment():
    """Create and activate virtual environment"""
    if os.path.exists("venv"):
        print("✅ Virtual environment already exists")
        return True
    
    return run_command("python -m venv venv", "Creating virtual environment")

def install_dependencies():
    """Install required dependencies"""
    # Determine the correct pip path
    if platform.system() == "Windows":
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"
    
    commands = [
        (f"{pip_path} install --upgrade pip", "Upgrading pip"),
        (f"{pip_path} install -r requirements.txt", "Installing dependencies")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True

def download_nltk_data():
    """Download required NLTK data"""
    nltk_script = """
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
print("NLTK data downloaded successfully")
"""
    
    # Determine the correct python path
    if platform.system() == "Windows":
        python_path = "venv\\Scripts\\python"
    else:
        python_path = "venv/bin/python"
    
    return run_command(f'{python_path} -c "{nltk_script}"', "Downloading NLTK data")

def create_sample_data():
    """Create sample data directory and file if it doesn't exist"""
    if not os.path.exists("data"):
        os.makedirs("data")
        print("✅ Created data directory")
    
    sample_data_path = "data/sample_data.csv"
    if not os.path.exists(sample_data_path):
        sample_data = """datetime,product,quote
2024-01-15 10:30:00,Product A,"This product is amazing and works perfectly!"
2024-01-15 11:45:00,Product B,"I'm not satisfied with the quality."
2024-01-16 09:15:00,Product A,"Great customer service and fast delivery."
2024-01-16 14:20:00,Product C,"The interface is confusing and hard to use."
2024-01-17 16:30:00,Product B,"Excellent value for money, highly recommended!"
"""
        with open(sample_data_path, "w") as f:
            f.write(sample_data)
        print("✅ Created sample data file")

def main():
    """Main setup function"""
    print("🚀 Setting up Sentiment Analysis Project...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Failed to create virtual environment")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Download NLTK data
    if not download_nltk_data():
        print("❌ Failed to download NLTK data")
        sys.exit(1)
    
    # Create sample data
    create_sample_data()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Run the analysis:")
    print("   python src/main.py")
    print("3. Or explore with Jupyter:")
    print("   jupyter lab")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main() 