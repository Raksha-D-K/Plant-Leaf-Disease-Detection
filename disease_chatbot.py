import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class DiseaseChatbot:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("api_key"),
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Comprehensive disease database with plant-specific information
        self.disease_knowledge = {
            "Leaf Spot Disease": {
                "symptoms": "Small to large circular brown spots on leaves, often with yellow halos",
                "causes": "Fungal infection (Cercospora, Alternaria, Septoria species)",
                "treatment": "Apply copper-based fungicide, remove infected leaves, improve air circulation",
                "prevention": "Avoid overhead watering, ensure proper spacing, use resistant varieties",
                "severity": "Moderate to severe if untreated"
            },
            "Apple Scab": {
                "symptoms": "Olive-green to black spots on apple leaves and fruit, velvety texture",
                "causes": "Fungal infection (Venturia inaequalis)",
                "treatment": "Apply fungicide containing captan or myclobutanil, remove fallen leaves",
                "prevention": "Resistant varieties, proper pruning, good air circulation, preventive fungicide",
                "severity": "Can cause significant fruit damage if untreated"
            },
            "Fire Blight": {
                "symptoms": "Blossoms turn black and wilt, branches appear burned, shepherd's crook shape",
                "causes": "Bacterial infection (Erwinia amylovora)",
                "treatment": "Prune infected branches, apply streptomycin spray, copper compounds",
                "prevention": "Resistant varieties, avoid nitrogen excess, proper pruning timing",
                "severity": "Can kill entire tree if untreated"
            },
            "Chlorosis": {
                "symptoms": "Yellowing of leaves while veins remain green, stunted growth",
                "causes": "Iron deficiency, poor soil pH, nutrient imbalance",
                "treatment": "Apply iron chelate, adjust soil pH, balanced fertilization",
                "prevention": "Regular soil testing, proper pH management, balanced nutrition",
                "severity": "Moderate but affects photosynthesis"
            },
            "Early Blight": {
                "symptoms": "Target-like brown spots with concentric rings and yellow halos on lower leaves",
                "causes": "Fungal infection (Alternaria solani)",
                "treatment": "Apply fungicide containing chlorothalonil or mancozeb, remove affected leaves",
                "prevention": "Crop rotation, resistant varieties, proper spacing, avoid overhead watering",
                "severity": "Can cause significant yield loss if untreated"
            },
            "Powdery Mildew": {
                "symptoms": "White powdery coating on leaf surface, can spread to stems and flowers",
                "causes": "Fungal infection (various Erysiphales species)",
                "treatment": "Apply sulfur-based fungicide, neem oil, or potassium bicarbonate spray",
                "prevention": "Ensure good air circulation, avoid high humidity, resistant varieties",
                "severity": "Usually mild but can affect plant growth"
            },
            "Northern Leaf Blight": {
                "symptoms": "Long, elliptical grayish-tan lesions on corn leaves, usually starting from lower leaves",
                "causes": "Fungal infection (Exserohilum turcicum)",
                "treatment": "Apply fungicide containing azoxystrobin or pyraclostrobin, resistant hybrids",
                "prevention": "Crop rotation, resistant hybrids, proper spacing, fungicide application",
                "severity": "Can cause significant yield loss in severe cases"
            },
            "Gray Leaf Spot": {
                "symptoms": "Rectangular brown lesions with gray centers on corn leaves, parallel to leaf veins",
                "causes": "Fungal infection (Cercospora zeae-maydis)",
                "treatment": "Apply fungicide containing strobilurin or triazole, remove infected debris",
                "prevention": "Resistant hybrids, crop rotation, proper irrigation management",
                "severity": "Moderate to severe, can reduce photosynthesis"
            },
            "Common Rust": {
                "symptoms": "Small reddish-brown pustules on corn leaves, arranged in rows",
                "causes": "Fungal infection (Puccinia sorghi)",
                "treatment": "Apply fungicide containing propiconazole or tebuconazole",
                "prevention": "Resistant hybrids, early detection, proper fungicide timing",
                "severity": "Usually mild but can be severe in favorable conditions"
            },
            "Stewart's Wilt": {
                "symptoms": "Leaf streaking, wilting, and death of corn plants, especially young plants",
                "causes": "Bacterial infection (Pantoea stewartii)",
                "treatment": "No effective treatment, remove infected plants, use resistant hybrids",
                "prevention": "Resistant hybrids, control corn flea beetles, crop rotation",
                "severity": "Can cause significant yield loss, especially in sweet corn"
            },
            "Late Blight": {
                "symptoms": "Water-soaked lesions on potato leaves and stems, quickly turn brown and black",
                "causes": "Fungal infection (Phytophthora infestans)",
                "treatment": "Apply fungicide containing metalaxyl or mancozeb, destroy infected plants",
                "prevention": "Resistant varieties, proper drainage, avoid overhead watering",
                "severity": "Very severe, can cause complete crop loss"
            },
            "Early Blight": {
                "symptoms": "Dark brown lesions with concentric rings on potato leaves, target-like appearance",
                "causes": "Fungal infection (Alternaria solani)",
                "treatment": "Apply fungicide containing chlorothalonil or copper, remove infected leaves",
                "prevention": "Crop rotation, resistant varieties, proper spacing",
                "severity": "Moderate to severe, reduces yield"
            },
            "Yellow Leaf Disease": {
                "symptoms": "Extensive yellowing of leaves, often starting from older leaves",
                "causes": "Nutrient deficiency (nitrogen, iron, magnesium), viral infections, water stress",
                "treatment": "Apply balanced fertilizer, check soil pH, ensure proper watering",
                "prevention": "Regular fertilization, proper watering schedule, soil testing",
                "severity": "Varies based on underlying cause"
            },
            "Bacterial Blight": {
                "symptoms": "Irregular brown lesions with yellow halos, water-soaked appearance",
                "causes": "Bacterial infection (Xanthomonas, Pseudomonas species)",
                "treatment": "Apply copper-based bactericide, remove infected plant parts, avoid overhead watering",
                "prevention": "Use disease-free seeds, proper sanitation, crop rotation",
                "severity": "Can be severe, especially in warm, humid conditions"
            },
            "Nutrient Deficiency": {
                "symptoms": "Yellowing between leaf veins, stunted growth, leaf discoloration",
                "causes": "Lack of essential nutrients (nitrogen, iron, magnesium, zinc)",
                "treatment": "Apply specific nutrient supplements, balanced fertilizer, soil amendment",
                "prevention": "Regular soil testing, balanced fertilization program",
                "severity": "Moderate but reversible with proper treatment"
            },
            "Healthy Leaf": {
                "symptoms": "No visible disease symptoms, vibrant green color, normal growth",
                "causes": "Proper care, good growing conditions, absence of pathogens",
                "treatment": "Continue current care routine, regular monitoring",
                "prevention": "Maintain good growing conditions, regular inspection",
                "severity": "No issues detected"
            }
        }
    
    def get_disease_info(self, disease_name):
        """
        Get detailed information about a specific disease (handles plant-specific names)
        """
        # Extract base disease name from plant-specific names
        base_disease = disease_name
        plant_type = "Unknown"
        
        # Check if the disease name contains a plant type
        if "Tomato" in disease_name:
            plant_type = "Tomato"
            base_disease = disease_name.replace("Tomato ", "").replace(" Tomato", "")
        elif "Apple" in disease_name:
            plant_type = "Apple"
            base_disease = disease_name.replace("Apple ", "").replace(" Apple", "")
        elif "Corn" in disease_name:
            plant_type = "Corn"
            base_disease = disease_name.replace("Corn ", "").replace(" Corn", "")
        elif "Potato" in disease_name:
            plant_type = "Potato"
            base_disease = disease_name.replace("Potato ", "").replace(" Potato", "")
        elif "Pepper" in disease_name:
            plant_type = "Pepper"
            base_disease = disease_name.replace("Pepper ", "").replace(" Pepper", "")
        elif "General Plant" in disease_name:
            plant_type = "General Plant"
            base_disease = disease_name.replace("General Plant ", "").replace(" General Plant", "")
        
        # Clean up the disease name
        base_disease = base_disease.replace(" - Healthy", "Healthy Leaf").strip()
        
        # Get disease information
        disease_info = self.disease_knowledge.get(base_disease, {
            "symptoms": "Information not available",
            "causes": "Unknown",
            "treatment": "Consult with plant specialist",
            "prevention": "Regular monitoring and proper care",
            "severity": "Unknown"
        })
        
        # Add plant-specific information
        if plant_type != "Unknown":
            disease_info["plant_type"] = plant_type
            disease_info["full_name"] = disease_name
        
        return disease_info
    
    def ask_about_disease(self, disease_name, question):
        """
        Ask a specific question about a disease
        """
        disease_info = self.get_disease_info(disease_name)
        
        system_prompt = f"""
        You are an expert plant pathologist specializing in {disease_name}. 
        Here is the known information about this disease:
        
        Symptoms: {disease_info['symptoms']}
        Causes: {disease_info['causes']}
        Treatment: {disease_info['treatment']}
        Prevention: {disease_info['prevention']}
        Severity: {disease_info['severity']}
        
        Answer the user's question based on this knowledge. Be helpful, specific, and practical.
        If the question is not related to this disease, politely redirect to the specific disease.
        Keep your response concise but informative.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I apologize, but I'm having trouble connecting to my knowledge base. Here's what I know about {disease_name}: {disease_info['treatment']}"
    
    def display_disease_info(self, disease_name):
        """
        Display comprehensive disease information in Streamlit
        """
        info = self.get_disease_info(disease_name)
        
        st.markdown(f"### 🦠 {disease_name}")
        
        # Display information in organized columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔍 Symptoms**")
            st.write(info['symptoms'])
            
            st.markdown("**🦠 Causes**")
            st.write(info['causes'])
        
        with col2:
            st.markdown("**💊 Treatment**")
            st.write(info['treatment'])
            
            st.markdown("**🛡️ Prevention**")
            st.write(info['prevention'])
        
        # Severity indicator
        severity_colors = {
            "Mild": "🟢",
            "Moderate": "🟡", 
            "Severe": "🔴",
            "Unknown": "⚪"
        }
        
        severity = info['severity']
        severity_emoji = "⚪"
        for key, emoji in severity_colors.items():
            if key.lower() in severity.lower():
                severity_emoji = emoji
                break
        
        st.markdown(f"**⚠️ Severity:** {severity_emoji} {severity}")
        
        # Chat interface for specific questions
        st.markdown("---")
        st.markdown("### 💬 Ask About This Disease")
        
        if "disease_chat_history" not in st.session_state:
            st.session_state.disease_chat_history = {}
        
        # Initialize chat history for this disease if not exists
        if disease_name not in st.session_state.disease_chat_history:
            st.session_state.disease_chat_history[disease_name] = []
        
        # Display chat history
        for message in st.session_state.disease_chat_history[disease_name]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if question := st.chat_input(f"Ask about {disease_name}..."):
            # Add user message
            st.session_state.disease_chat_history[disease_name].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            
            # Get AI response
            with st.chat_message("assistant"):
                response = self.ask_about_disease(disease_name, question)
                st.markdown(response)
            
            # Add assistant response
            st.session_state.disease_chat_history[disease_name].append({"role": "assistant", "content": response})

def create_disease_chatbot_interface(disease_name):
    """
    Create a Streamlit interface for the disease chatbot
    """
    chatbot = DiseaseChatbot()
    chatbot.display_disease_info(disease_name)
