"""
Project Agent Handler
Shared logic for project code generation
"""

from crewai import Agent, Task, Crew
from a2a.schemas import A2ATask, A2AResponse


from config import LLM_MODEL


def handle_project(task: A2ATask) -> A2AResponse:
    """Handle project generation task"""
    try:
        engineer = Agent(
            role="Software Engineer",
            goal="Generate production-ready Python analysis code with proper preprocessing",
            backstory="You are an expert Python developer who writes clean, robust data analysis scripts with proper error handling and preprocessing.",
            llm=LLM_MODEL
        )
        
        t = Task(
            description=f"""
Generate a complete, production-ready analysis.py file based on the analysis context.

Context:
{task.input}

CRITICAL REQUIREMENTS:
1. Use 'data.csv' as the filename (it will be copied to the output directory)
2. ALWAYS preprocess categorical columns using LabelEncoder BEFORE training models
3. **Create directories DIRECTLY at root level**: os.makedirs('stats', exist_ok=True), os.makedirs('plots', exist_ok=True), os.makedirs('reports', exist_ok=True)
   - DO NOT create a parent 'output/' directory
   - Create stats/, plots/, reports/ directly in the same directory as analysis.py
4. Handle errors gracefully with try-except blocks
5. Keep print output minimal (use head() not full data display)
6. Include all necessary imports
7. **IMPORTANT**: Save model performance to 'stats/model_performance.txt' (direct path, NOT output/stats/)
8. Suppress warnings: Use max_iter=1000 for LogisticRegression, zero_division=0 for metrics

FOLDER STRUCTURE (all at root level):
- stats/ → Model accuracy and performance metrics (.txt files)
- plots/ → Visualizations (histograms, bar charts, confusion matrices)
- reports/ → Detailed classification reports and analysis summaries

Example directory creation:
```python
os.makedirs('stats', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('reports', exist_ok=True)
```

Example file saving:
```python
# CORRECT:
with open('stats/model_performance.txt', 'w') as f:
# WRONG:
with open('output/stats/model_performance.txt', 'w') as f:  # NO!
```

The analysis.py file must:
- Load data.csv
- Create required directories (stats, plots, reports)
- Perform EDA with visualizations saved to plots/
- **Encode ALL categorical columns using LabelEncoder**
- Train the best models (Logistic Regression with max_iter=1000, Random Forest, Decision Tree, XGBoost)
- Save model accuracy to stats/model_performance.txt
- Save detailed reports to reports/
- Run without errors or warnings

Generate ONLY the Python code for analysis.py. Make it complete and executable.
Include proper preprocessing steps to convert categorical data before model training.
""",
            expected_output="Complete Python code for analysis.py that handles preprocessing correctly and saves outputs to correct folders",
            agent=engineer
        )
        
        crew = Crew(agents=[engineer], tasks=[t])
        result = crew.kickoff()
        result_str = str(result)
        
        # Extract only Python code (remove markdown and explanatory text)
        code = result_str
        
        # Remove markdown code fences
        if "```python" in code:
            # Extract code between ```python and ```
            start = code.find("```python") + len("```python")
            end = code.find("```", start)
            if end != -1:
                code = code[start:end].strip()
        elif "```" in code:
            # Extract code between ``` and ```
            start = code.find("```") + 3
            end = code.find("```", start)
            if end != -1:
                code = code[start:end].strip()
        
        # Remove common explanatory prefixes
        lines = code.split('\n')
        clean_lines = []
        started = False
        
        for line in lines:
            # Skip explanatory text before code starts
            if not started:
                # Check if this is actual Python code
                if (line.strip().startswith('import ') or 
                    line.strip().startswith('from ') or
                    line.strip().startswith('#')):
                    started = True
                    clean_lines.append(line)
            else:
                # Only include the line if it's not trailing explanation
                if line.strip() and not line.strip().startswith('This code'):
                    clean_lines.append(line)
        
        code = '\n'.join(clean_lines)
        
        # Also generate a simple README
        readme_content = """# AutoEDA Analysis Report

## Overview
Automated exploratory data analysis and machine learning model training.

## Files Generated
- `stats/` - Statistical summaries and model performance
- `plots/` - Visualizations and correlation matrices
- `reports/` - Detailed analysis reports

## How to Run
```bash
python analysis.py
```

## Requirements
Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Models Trained
- Logistic Regression
- Random Forest
- Decision Tree
- XGBoost

Results are saved in the stats/ directory.
"""
        
        return A2AResponse(
            task_id=task.task_id,
            sender="project-agent",
            status="COMPLETED",
            output={
                "analysis_code": code,  # Use cleaned code, not raw result
                "readme": readme_content
            }
        )
    except Exception as e:
        print(f"Error: {e}")
        raise
