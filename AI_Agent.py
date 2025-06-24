import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv
import openai
from pydantic import BaseModel
from langgraph.graph import StateGraph
import numpy as np
import json
import uuid
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
client = openai.OpenAI(api_key=OPENAI_API_KEY)

class AnalysisState(BaseModel):
    user_query: str = ""
    conversation_history: List[Dict] = []
    waiting_for_human: bool = False
    human_feedback: Optional[str] = None
    human_decision: Optional[str] = None
    feedback_timestamp: Optional[str] = None
    current_analysis: Dict = {}
    data_subset: Dict = {}
    enhancement_requests: List[str] = []
    chart_requests: List[str] = []
    generated_charts: List[Dict] = []
    current_response: str = ""
    loop_iteration: int = 0
    max_iterations: int = 10
    session_active: bool = True
    session_id: str = ""

class DataAnalyzer:
    def __init__(self, csv_path: str):
        self.data = self.load_csv_data(csv_path)
        
    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """Load CSV data with proper date parsing, limiting to first 200 columns"""
        try:
            df_header = pd.read_csv(csv_path, nrows=0)
            
            columns_to_use = df_header.columns[:200]
            
            df = pd.read_csv(csv_path, usecols=columns_to_use)
            
            print(f"Loaded {len(df)} rows and {len(df.columns)} columns (limited to first 200)")
            
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass  
            
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return pd.DataFrame()
    
    def get_data_summary(self) -> Dict:
        """Generate basic statistics for the data"""
        if self.data.empty:
            return {"error": "No data available"}
        
        return {
            "row_count": len(self.data),
            "column_count": len(self.data.columns),
            "columns": list(self.data.columns),
            "numeric_columns": list(self.data.select_dtypes(include=['number']).columns),
            "date_columns": [col for col in self.data.columns if 'date' in col.lower() or 'time' in col.lower()]
        }
    def analyze_top_products(self, limit: int = 5) -> Dict:
        """Find highest selling products"""
        if 'product' not in self.data.columns or 'sales' not in self.data.columns:
            return {"error": "Required columns not found"}
        
        top_products = self.data.groupby('product')['sales'].sum().sort_values(ascending=False).head(limit)
        return {
            "top_products": top_products.to_dict(),
            "analysis_type": "top_products"
        }
    
    def analyze_top_products_by_column(self, product_col: str, value_col: str, limit: int = 10) -> Dict:
        """Find top products using specified columns"""
        if product_col not in self.data.columns or value_col not in self.data.columns:
            return {"error": f"Required columns '{product_col}' or '{value_col}' not found"}
        
        try:
            # Group by product column and sum the value column
            top_products = self.data.groupby(product_col)[value_col].sum().sort_values(ascending=False).head(limit)
            
            return {
                "top_products": top_products.to_dict(),
                "analysis_type": "top_products_by_column",
                "product_column": product_col,
                "value_column": value_col,
                "total_products_analyzed": len(self.data[product_col].unique()),
                "total_records": len(self.data)
            }
        except Exception as e:
            return {"error": f"Error analyzing top products: {e}"}
    
    def analyze_trends_over_time(self, time_column: str = 'date', value_column: str = 'sales') -> Dict:
        """Analyze trends over time"""
        if time_column not in self.data.columns or value_column not in self.data.columns:
            return {"error": f"Required columns {time_column} or {value_column} not found"}
        
        try:
            # Group by date and calculate sum of values
            time_trend = self.data.groupby(pd.Grouper(key=time_column, freq='M'))[value_column].sum()
            
            return {
                "time_data": {str(date): value for date, value in time_trend.items()},
                "time_column": time_column,
                "value_column": value_column,
                "analysis_type": "time_trends"
            }
        except Exception as e:
            return {"error": f"Error analyzing time trends: {e}"}

# Chart Generator
class ChartGenerator:
    def __init__(self, output_dir: str = "./charts"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def generate_line_chart(self, data: Dict, context: Dict) -> Dict:
        """Generate a line chart for time-based data"""
        try:
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract data
            dates = list(data.keys())
            values = list(data.values())
            
            # Plot line chart
            ax.plot(dates, values, marker='o', linestyle='-')
            
            # Set title and labels
            ax.set_title(f"{context.get('value_column', 'Value')} over {context.get('time_column', 'Time')}")
            ax.set_xlabel(context.get('time_column', 'Time'))
            ax.set_ylabel(context.get('value_column', 'Value'))
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Tight layout
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/line_chart_{timestamp}.png"
            plt.savefig(filename)
            plt.close(fig)
            
            return {
                "type": "line_chart",
                "filename": filename,
                "title": f"{context.get('value_column', 'Value')} over {context.get('time_column', 'Time')}"
            }
        except Exception as e:
            return {"error": f"Error generating line chart: {e}"}

# Vector Embedding and Storage Manager
class VectorMemoryManager:
    def __init__(self):
        self.sessions = {}
        self.embeddings_cache = {}
        self.analysis_cache = {}
        
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI embedding model"""
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return a random embedding as fallback (not ideal but prevents crashes)
            return list(np.random.rand(1536))  # OpenAI embeddings are 1536 dimensions
    
    def save_session(self, session_id: str, state: Dict):
        """Save session state with embeddings"""
        # Convert complex types to strings for storage
        session_data = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in state.items()}
        self.sessions[f"session:{session_id}"] = session_data
    
    def get_session(self, session_id: str) -> Dict:
        """Retrieve session state"""
        return self.sessions.get(f"session:{session_id}", {})
    
    def save_analysis_result(self, query: str, result: Dict):
        """Cache analysis results with embeddings for semantic retrieval"""
        query_id = str(uuid.uuid4())
        query_embedding = self._generate_embedding(query)
        
        # Store both the embedding and the result
        self.embeddings_cache[query_id] = {
            "query": query,
            "embedding": query_embedding,
            "timestamp": datetime.now().isoformat()
        }
        
        self.analysis_cache[query_id] = result
    
    def find_similar_analysis(self, query: str, threshold: float = 0.8) -> Optional[Dict]:
        """Find similar previous analysis using vector similarity"""
        if not self.embeddings_cache:
            return None
            
        query_embedding = self._generate_embedding(query)
        
        # Calculate cosine similarity with stored embeddings
        max_similarity = 0
        most_similar_id = None
        
        for query_id, item in self.embeddings_cache.items():
            stored_embedding = item["embedding"]
            similarity = self._cosine_similarity(query_embedding, stored_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_id = query_id
        
        # Return the most similar analysis if above threshold
        if max_similarity >= threshold and most_similar_id:
            print(f"Found similar query: '{self.embeddings_cache[most_similar_id]['query']}' with similarity: {max_similarity:.2f}")
            return self.analysis_cache.get(most_similar_id)
        
        return None
    
    def save_to_disk(self, directory: str = "./vector_store"):
        """Save vector store to disk for persistence"""
        try:
            # Create directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # Save embeddings cache (without actual embeddings for JSON serialization)
            serializable_embeddings = {}
            for query_id, item in self.embeddings_cache.items():
                serializable_embeddings[query_id] = {
                    "query": item["query"],
                    "embedding": list(map(float, item["embedding"])),  # Ensure all values are JSON serializable
                    "timestamp": item["timestamp"]
                }
            
            with open(f"{directory}/embeddings.json", "w") as f:
                json.dump(serializable_embeddings, f)
            
            # Save analysis cache
            with open(f"{directory}/analyses.json", "w") as f:
                json.dump(self.analysis_cache, f)
            
            # Save sessions
            with open(f"{directory}/sessions.json", "w") as f:
                json.dump(self.sessions, f)
                
            print(f"Vector store saved to {directory}")
        except Exception as e:
            print(f"Error saving vector store to disk: {e}")
    
    def load_from_disk(self, directory: str = "./vector_store"):
        """Load vector store from disk"""
        try:
            # Check if directory and files exist
            if not os.path.exists(directory):
                print(f"Directory {directory} does not exist. Creating a new vector store.")
                return
            
            # Load embeddings cache
            embeddings_path = f"{directory}/embeddings.json"
            if os.path.exists(embeddings_path):
                with open(embeddings_path, "r") as f:
                    self.embeddings_cache = json.load(f)
            
            # Load analysis cache
            analyses_path = f"{directory}/analyses.json"
            if os.path.exists(analyses_path):
                with open(analyses_path, "r") as f:
                    self.analysis_cache = json.load(f)
            
            # Load sessions
            sessions_path = f"{directory}/sessions.json"
            if os.path.exists(sessions_path):
                with open(sessions_path, "r") as f:
                    self.sessions = json.load(f)
                    
            print(f"Vector store loaded from {directory}")
            print(f"Loaded {len(self.embeddings_cache)} embeddings, {len(self.analysis_cache)} analyses, and {len(self.sessions)} sessions")
        except Exception as e:
            print(f"Error loading vector store from disk: {e}")
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not v1 or not v2:
            return 0
            
        v1 = np.array(v1)
        v2 = np.array(v2)
        
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0
            
        return dot_product / (norm_v1 * norm_v2)

# Human Feedback Controller
class HumanFeedbackController:
    def __init__(self):
        self.active_session = None
        self.waiting_for_input = False
    
    def wait_for_human_input(self, state: AnalysisState) -> Dict:
        """Pause execution and wait for human input"""
        print(f"\nAI: {state.current_response}")
        print("\nWhat would you like to do next? (You can ask for more analysis, request a chart, ask a new question, or say you're done)")
        
        # SYSTEM WAITS HERE - NO AUTOMATIC PROGRESSION
        self.waiting_for_input = True
        
        # Get natural language feedback from human
        human_input = input("\nYour feedback: ")
        
        # Let the LLM determine the action from the feedback
        return {
            "action": "feedback",  # This will be interpreted by the LLM router
            "details": human_input,
            "timestamp": datetime.now().isoformat()
        }

# LangGraph Node Implementations
def data_analysis_agent(state: AnalysisState) -> AnalysisState:
    """Analyze data based on user query"""
    query = state.user_query.lower()
    
    # First, get data summary to understand what columns are available
    data_summary = data_analyzer.get_data_summary()
    
    # First, check for similar previous analyses
    similar_analysis = memory_manager.find_similar_analysis(query)
    
    if similar_analysis and "error" not in similar_analysis:
        print("Using cached similar analysis")
        analysis_result = similar_analysis
    else:
        # Show available columns to help with analysis
        print(f"Available columns in dataset: {data_summary.get('columns', [])}")
        print(f"Numeric columns: {data_summary.get('numeric_columns', [])}")
        
        # Determine analysis type based on query and available columns
        if any(word in query for word in ["top", "best", "highest", "profit"]):
            # Try to find profit-related columns
            profit_columns = [col for col in data_summary.get('columns', []) if 'profit' in col.lower()]
            product_columns = [col for col in data_summary.get('columns', []) if any(term in col.lower() for term in ['product', 'item', 'name', 'title', 'asin'])]
            
            if profit_columns and product_columns:
                # Use actual profit and product columns
                analysis_result = data_analyzer.analyze_top_products_by_column(
                    product_col=product_columns[0], 
                    value_col=profit_columns[0],
                    limit=10
                )
            else:
                # Fallback to any numeric column for top products analysis
                numeric_cols = data_summary.get('numeric_columns', [])
                product_cols = product_columns if product_columns else [col for col in data_summary.get('columns', []) if 'name' in col.lower() or 'title' in col.lower()]
                
                if numeric_cols and product_cols:
                    analysis_result = data_analyzer.analyze_top_products_by_column(
                        product_col=product_cols[0],
                        value_col=numeric_cols[0],
                        limit=10
                    )
                else:
                    analysis_result = {
                        "error": f"Cannot perform top products analysis. Available columns: {data_summary.get('columns', [])}",
                        "data_summary": data_summary
                    }
        elif any(word in query for word in ["time", "trend", "over", "period"]):
            analysis_result = data_analyzer.analyze_trends_over_time()
        else:
            # Default analysis with available data
            analysis_result = {
                "data_summary": data_summary,
                "analysis_type": "data_overview"
            }    # Generate AI response using OpenAI GPT-4o-mini
    prompt = f"""
    You are an expert data analyst examining Amazon trading data.
    
    User Query: {query}
    Data Analysis: {analysis_result}
    Available Data Summary: {data_summary}
    
    Based on the ACTUAL data available, provide analysis including:
    1. Key insights with specific metrics from the actual data
    2. Business implications based on real findings
    3. Actionable recommendations
    
    Important: 
    - Use the actual column names and data found in the dataset
    - If there's an error, explain what data is missing and what columns are actually available
    - Be specific with numbers and percentages from the real data
    - Don't make assumptions about data that doesn't exist
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert data analyst specializing in Amazon trading data analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        analysis_text = response.choices[0].message.content
    except Exception as e:
        analysis_text = f"Error generating analysis: {e}"
    
    # Update state
    state.current_analysis = analysis_result
    state.current_response = analysis_text
    
    # Save to memory
    memory_manager.save_analysis_result(query.strip().lower(), analysis_result)
    
    return state

def human_feedback_interrupt(state: AnalysisState) -> AnalysisState:
    """Pause execution and wait for human input"""
    state.waiting_for_human = True
    return state

def process_feedback(state: AnalysisState) -> AnalysisState:
    """Process human feedback and enhance response"""
    feedback = state.human_feedback
    
    if not feedback:
        return state
    
    # Generate enhanced response based on feedback
    prompt = f"""
    The user provided feedback on the previous analysis.
    
    Original Analysis: {state.current_response}
    User Feedback: {feedback}
    Data: {state.current_analysis}
    
    Enhance the analysis by addressing the user's specific request:
    {feedback}
    
    Provide a complete, updated analysis that incorporates this feedback.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert data analyst. Enhance your analysis based on user feedback."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        enhanced_text = response.choices[0].message.content
    except Exception as e:
        enhanced_text = f"Error enhancing analysis: {e}\nOriginal analysis: {state.current_response}"
    
    # Update state
    state.current_response = enhanced_text
    state.enhancement_requests.append(feedback)
    
    return state

def create_visualization(state: AnalysisState) -> AnalysisState:
    """Create visualization based on analysis type"""
    analysis_type = state.current_analysis.get("analysis_type")
    
    if analysis_type == "time_trends":
        # Generate line chart for time trends
        chart_data = state.current_analysis.get("time_data", {})
        context = {
            "time_column": state.current_analysis.get("time_column", "Time"),
            "value_column": state.current_analysis.get("value_column", "Value")
        }
        
        chart_result = chart_generator.generate_line_chart(chart_data, context)
        
        if "error" not in chart_result:
            state.generated_charts.append(chart_result)
            state.current_response += f"\n\nI've created a line chart showing {context['value_column']} over {context['time_column']}. The chart is saved as: {chart_result['filename']}"
        else:
            state.current_response += f"\n\nError creating chart: {chart_result['error']}"
            
    elif analysis_type == "top_products_by_column":
        # Generate bar chart for top products
        top_products = state.current_analysis.get("top_products", {})
        if top_products:
            try:
                # Create figure and axis
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Extract product names and values
                products = list(top_products.keys())
                values = list(top_products.values())
                
                # Create bar chart
                bars = ax.bar(range(len(products)), values)
                
                # Customize chart
                ax.set_xlabel('Products (ASIN)')
                ax.set_ylabel(state.current_analysis.get("value_column", "Value"))
                ax.set_title(f'Top {len(products)} Products by {state.current_analysis.get("value_column", "Value")}')
                ax.set_xticks(range(len(products)))
                ax.set_xticklabels(products, rotation=45, ha='right')
                
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, values)):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                           f'${value:.2f}', ha='center', va='bottom', fontsize=9)
                
                # Tight layout
                plt.tight_layout()
                
                # Save chart
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{chart_generator.output_dir}/top_products_chart_{timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                chart_result = {
                    "type": "bar_chart",
                    "filename": filename,
                    "title": f'Top {len(products)} Products by {state.current_analysis.get("value_column", "Value")}'
                }
                
                state.generated_charts.append(chart_result)
                state.current_response += f"\n\nI've created a bar chart showing the top {len(products)} products by {state.current_analysis.get('value_column', 'value')}. The chart is saved as: {filename}"
                
            except Exception as e:
                state.current_response += f"\n\nError creating bar chart: {e}"
        else:
            state.current_response += "\n\nNo product data available for visualization."
    else:
        state.current_response += f"\n\nVisualization not supported for analysis type: {analysis_type}. Available types: time_trends, top_products_by_column"
    
    return state

def present_to_human(state: AnalysisState) -> AnalysisState:
    """Present current results to human"""
    return state

def route_based_on_human_input(state: AnalysisState) -> str:
    """Intelligently route based on human input using LLM"""
    if not state.human_feedback and not state.human_decision:
        return "end"
      # If human explicitly provided a decision, use it directly
    if state.human_decision:
        if state.human_decision == "enhance":
            return "process_feedback"
        elif state.human_decision == "visualize":
            return "create_charts"
        elif state.human_decision == "new_query":
            return "analyze_data"
        elif state.human_decision == "end":
            return "end"
    
    # Use LLM to analyze feedback and determine next action
    if state.human_feedback:
        prompt = f"""
        Analyze this user feedback and determine the best next action to take.
        
        Previous AI Response: {state.current_response}
        User Feedback: {state.human_feedback}
        
        Based only on the user feedback, determine which ONE of these actions is most appropriate:
        1. "process_feedback" - The user wants additional analysis or enhancement of existing analysis
        2. "create_charts" - The user wants a visualization or chart created
        3. "analyze_data" - The user is asking a new question or wants different data analyzed
        4. "end" - The user is satisfied or wants to end the conversation
        
        Return ONLY ONE of these exact strings: "process_feedback", "create_charts", "analyze_data", or "end".
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a decision-making AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.1
            )
            action = response.choices[0].message.content.strip().lower()
            
            # Extract just the action if there's additional text
            if "process_feedback" in action:
                return "process_feedback"
            elif "create_charts" in action:
                return "create_charts"
            elif "analyze_data" in action:
                return "analyze_data"
            elif "end" in action:
                return "end"
        except Exception as e:
            print(f"Error in LLM routing: {e}")
    
    # Default fallback
    return "wait_for_feedback"

# Create LangGraph workflow
def create_workflow():
    workflow = StateGraph(AnalysisState)
    
    # Add nodes
    workflow.add_node("analyze_data", data_analysis_agent)
    workflow.add_node("wait_for_feedback", human_feedback_interrupt)
    workflow.add_node("process_feedback", process_feedback)
    workflow.add_node("create_charts", create_visualization)
    workflow.add_node("present_result", present_to_human)
    
    # Define human-in-the-loop flow
    workflow.set_entry_point("analyze_data")
    workflow.add_edge("analyze_data", "present_result")
    workflow.add_edge("present_result", "wait_for_feedback")
    
    # Import the proper END constant
    from langgraph.graph.graph import END
      # Conditional routing based on human feedback
    workflow.add_conditional_edges(
        "wait_for_feedback",
        route_based_on_human_input,
        {
            "process_feedback": "process_feedback",
            "create_charts": "create_charts", 
            "analyze_data": "analyze_data",
            "end": END  # Properly use END constant to end the workflow
        }
    )
    
    workflow.add_edge("process_feedback", "present_result")
    workflow.add_edge("create_charts", "present_result")
    
    return workflow.compile()

# Main Agentic AI Class
class HumanInTheLoopAgenticAI:
    def __init__(self, csv_path: str):
        self.workflow = create_workflow()
        self.feedback_controller = HumanFeedbackController()
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def start_interactive_session(self, initial_query: str):
        """Start a new human-in-the-loop session"""
        print(f"🤖 Starting new analysis session: {self.session_id}")
        
        # Initialize state with human query
        state = AnalysisState(
            user_query=initial_query,
            session_active=True,
            loop_iteration=0,
            session_id=self.session_id
        )
        
        # Start the human-in-the-loop process
        self._run_feedback_loop(state)
    
    def _run_feedback_loop(self, state: AnalysisState):
        """Main human-in-the-loop execution with human input at each step"""
        while state.session_active and state.loop_iteration < state.max_iterations:
            # Step 1: Run workflow with current state
            state_dict = state.model_dump()  # Updated from state.dict() to fix deprecation warning
            result = self.workflow.invoke(state_dict)
            
            # Convert result back to AnalysisState
            for key, value in result.items():
                setattr(state, key, value)
              # Step 2: Wait for human decision if workflow is waiting
            if state.waiting_for_human:
                human_feedback = self.feedback_controller.wait_for_human_input(state)
                
                # Process human feedback
                if "I'm done" in human_feedback["details"].lower() or "thank" in human_feedback["details"].lower():
                    print("👋 Thank you! Ending analysis session.")
                    state.session_active = False
                    break
                
                # Store the raw feedback for the LLM to interpret
                state.human_feedback = human_feedback["details"]
                
                # Use LLM to determine what to do with the feedback
                prompt = f"""
                Analyze this user feedback and determine the best next action to take.
                
                Previous AI Response: {state.current_response}
                User Feedback: {state.human_feedback}
                
                Based only on the user feedback, determine which ONE of these actions is most appropriate:
                1. "process_feedback" - The user wants additional analysis or enhancement of existing analysis
                2. "create_charts" - The user wants a visualization or chart created
                3. "analyze_data" - The user is asking a new question or wants different data analyzed
                4. "end" - The user is satisfied or wants to end the conversation
                
                Return ONLY ONE of these exact strings: "process_feedback", "create_charts", "analyze_data", or "end".
                """
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a decision-making AI assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=50,
                        temperature=0.1
                    )
                    action = response.choices[0].message.content.strip().lower()
                    
                    print(f"LLM determined action: {action}")
                      # Extract the action and set decision
                    if "create_charts" in action:
                        state.human_decision = "visualize"
                        state.chart_requests.append("line_chart")
                    elif "analyze_data" in action:
                        state.human_decision = "new_query"
                        state.user_query = state.human_feedback  # Use the feedback as the new query
                        state.loop_iteration = 0
                    elif "end" in action:
                        print("👋 Thank you! Ending analysis session.")
                        state.session_active = False
                        break
                    else:  # Default to process_feedback
                        state.human_decision = "enhance"
                except Exception as e:
                    print(f"Error in LLM decision making: {e}")
                    # Default to process feedback if LLM fails
                    state.human_decision = "enhance"
                
                # Save state to memory
                memory_manager.save_session(state.session_id, state.model_dump())
                
                # Reset waiting flag
                state.waiting_for_human = False
            
            state.loop_iteration += 1
        print("\n" + "="*60)
        print(f"📊 SESSION SUMMARY")
        print("="*60)
        print(f"Total interactions: {state.loop_iteration}")
        print(f"Charts generated: {len(state.generated_charts)}")
        print(f"Enhancement requests: {len(state.enhancement_requests)}")
        print("="*60)
        
        # Save vector store to disk for future sessions
        memory_manager.save_to_disk()

# Initialize components
data_analyzer = None
chart_generator = None
memory_manager = None

def initialize_system(csv_path: str):
    global data_analyzer, chart_generator, memory_manager
    data_analyzer = DataAnalyzer(csv_path)
    chart_generator = ChartGenerator()
    
    # Initialize vector memory manager and load previous data if available
    memory_manager = VectorMemoryManager()
    memory_manager.load_from_disk()  # Load previous embeddings and analyses

def main():
    # Check if API key is set
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set. Please set it in your environment variables.")
        return
    
    # Get CSV path from user
    csv_path = input("Enter path to Amazon trading CSV file: ")
    
    # Initialize system
    initialize_system(csv_path)
    
    # Start AI
    ai = HumanInTheLoopAgenticAI(csv_path)
    
    print("🚀 Welcome to Human-in-the-Loop Agentic AI!")
    print("You will be able to iteratively enhance and modify the analysis.")
    
    initial_query = input("\n🤔 What would you like to analyze? ")
    
    # Start interactive session
    ai.start_interactive_session(initial_query)

if __name__ == "__main__":
    main()