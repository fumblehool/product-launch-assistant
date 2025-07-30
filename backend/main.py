import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from langgraph.graph import StateGraph, END
from langchain_gradientai import ChatGradientAI
from dotenv import load_dotenv

load_dotenv()

# Set your Gradient API key as an environment variable
DIGITALOCEAN_INFERENCE_KEY = os.getenv("DIGITALOCEAN_INFERENCE_KEY")

llm = ChatGradientAI(
    api_key=DIGITALOCEAN_INFERENCE_KEY,
    temperature=0.7,
    model="llama3.3-70b-instruct"
)

# 1. Market Research
def market_research(state: dict):
    prompt = (
        f"Conduct a brief market research for a product named '{state['product_name']}' targeting '{state['target_market']}'. "
        f"Summarize key competitors, market trends, and potential opportunities."
    )
    state['market_research'] = llm.invoke(prompt).content
    return state

# 2. Product Description Generation
def product_description(state: dict):
    prompt = (
        f"Write a compelling e-commerce product description for '{state['product_name']}'. "
        f"Product details: {state['product_details']}. "
        f"Target market: {state['target_market']}."
    )
    state['product_description'] = llm.invoke(prompt).content
    return state

# 3. Pricing Strategy Suggestion
def pricing_strategy(state: dict):
    prompt = (
        f"Based on the following market research and product details, suggest a pricing strategy for '{state['product_name']}':\n"
        f"Market research: {state['market_research']}\n"
        f"Product details: {state['product_details']}"
    )
    state['pricing_strategy'] = llm.invoke(prompt).content
    return state

# 4. Launch Plan Creation
def launch_plan(state: dict):
    prompt = (
        f"Create a step-by-step launch plan for '{state['product_name']}' targeting '{state['target_market']}'. "
        f"Include pre-launch, launch, and post-launch activities."
    )
    state['launch_plan'] = llm.invoke(prompt).content
    return state

# 5. Marketing Content Generation
def marketing_content(state: dict):
    prompt = (
        f"Generate:\n"
        f"1. A social media post (Twitter/X style, max 280 characters)\n"
        f"2. An email announcement (max 150 words)\n"
        f"For the product '{state['product_name']}'. Use the following product description:\n"
        f"{state['product_description']}"
    )
    state['marketing_content'] = llm.invoke(prompt).content
    return state

# Build the LangGraph workflow
graph = StateGraph(dict)
graph.add_node("market_research", market_research)
graph.add_node("product_description", product_description)
graph.add_node("pricing_strategy", pricing_strategy)
graph.add_node("launch_plan", launch_plan)
graph.add_node("marketing_content", marketing_content)

graph.set_entry_point("market_research")
graph.add_edge("market_research", "product_description")
graph.add_edge("product_description", "pricing_strategy")
graph.add_edge("pricing_strategy", "launch_plan")
graph.add_edge("launch_plan", "marketing_content")
graph.add_edge("marketing_content", END)
workflow = graph.compile()


class LaunchRequest(BaseModel):
    product_name: str
    product_details: str
    target_market: str

class LaunchResponse(BaseModel):
    product_name: str
    product_details: str
    target_market: str
    market_research: str
    product_description: str
    pricing_strategy: str
    launch_plan: str
    marketing_content: str

# Create FastAPI app
app = FastAPI(
    title="Product Launch Assistant API",
    description="AI-powered product launch planning API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes (all prefixed with /api)
@app.post("/api/launch_assistant", response_model=LaunchResponse)
async def generate_launch_plan(request: LaunchRequest):
    """
    Generate a comprehensive product launch plan using AI.
    
    This endpoint takes product information and returns:
    - Market research and competitor analysis
    - Product description optimization  
    - Pricing strategy recommendations
    - Step-by-step launch plan with timelines
    - Marketing content and social media campaigns
    """
    try:
        state = {
            "product_name": request.product_name,
            "product_details": request.product_details,
            "target_market": request.target_market
        }
        final_state = workflow.invoke(state)
        return LaunchResponse(
            product_name=final_state.get("product_name", ""),
            product_details=final_state.get("product_details", ""),
            target_market=final_state.get("target_market", ""),
            market_research=final_state.get("market_research", ""),
            product_description=final_state.get("product_description", ""),
            pricing_strategy=final_state.get("pricing_strategy", ""),
            launch_plan=final_state.get("launch_plan", ""),
            marketing_content=final_state.get("marketing_content", "")
        )
        # For now, return structured response (replace with your actual logic)
        # response = LaunchResponse(
        #     product_name=request.product_name,
        #     product_details=request.product_details,
        #     target_market=request.target_market,
        #     market_research=f"""## Market Analysis for {request.product_name}

        #     **Market Size & Opportunity:**
        #     The market for {request.product_name.lower()} shows strong growth potential, with an estimated market size of $2.5 billion globally. The target segment of {request.target_market.lower()} represents approximately 45% of the total addressable market.

        #     **Competitive Landscape:**
        #     - **Direct Competitors:** 3-5 major players with 60% market share
        #     - **Indirect Competitors:** 8-12 smaller players with 25% market share  
        #     - **Market Gap:** 15% unserved market opportunity

        #     **Key Market Trends:**
        #     1. Increasing demand for sustainable and eco-friendly products
        #     2. Growing preference for smart, connected devices
        #     3. Rising health consciousness among urban professionals
        #     4. Shift towards premium, quality-focused products

        #     **SWOT Analysis:**
        #     - **Strengths:** Unique features, strong value proposition
        #     - **Weaknesses:** New market entrant, limited brand recognition
        #     - **Opportunities:** Growing market, underserved segments
        #     - **Threats:** Established competitors, market saturation

        #     **Recommendations:**
        #     - Focus on differentiation through unique features
        #     - Target early adopters and influencers
        #     - Build strong brand positioning in the premium segment
        #     - Leverage sustainability and health benefits as key selling points""",
                        
        #                 product_description=f"""# {request.product_name}

        #     ## Transform Your Daily Routine with Innovation

        #     Experience the future of {request.product_name.lower()} with our cutting-edge solution designed specifically for {request.target_market.lower()}.

        #     ### ✨ Key Features
        #     - **Advanced Technology:** State-of-the-art features that set new industry standards
        #     - **Premium Quality:** Crafted with the finest materials for lasting durability
        #     - **Smart Integration:** Seamlessly connects with your digital lifestyle
        #     - **Eco-Friendly Design:** Sustainable materials and energy-efficient operation

        #     ### 🎯 Perfect For
        #     - Busy professionals seeking efficiency
        #     - Health-conscious individuals
        #     - Tech-savvy early adopters
        #     - Quality-focused consumers

        #     ### 💡 Why Choose {request.product_name}?
        #     Our product stands out with its unique combination of innovation, quality, and user experience. We've designed every aspect with {request.target_market.lower()} in mind, ensuring it perfectly fits your lifestyle and needs.

        #     ### 🔥 Limited Time Offer
        #     Get early access to the future of {request.product_name.lower()} with exclusive launch pricing and premium support.

        #     *Experience the difference that thoughtful design and cutting-edge technology can make in your daily life.*""",
                        
        #                 pricing_strategy=f"""## Pricing Strategy for {request.product_name}

        #     ### 🎯 Recommended Pricing Structure

        #     **Premium Tier: $299**
        #     - Full feature set
        #     - Premium materials
        #     - Extended warranty (3 years)
        #     - Priority customer support
        #     - Exclusive accessories included

        #     **Standard Tier: $199**
        #     - Core features
        #     - Quality materials
        #     - Standard warranty (1 year)
        #     - Regular customer support
        #     - Basic accessories

        #     **Entry Tier: $149**
        #     - Essential features
        #     - Standard materials
        #     - Basic warranty (6 months)
        #     - Email support
        #     - No accessories

        #     ### 💰 Pricing Rationale

        #     **Cost Analysis:**
        #     - Manufacturing cost: $85-120 per unit
        #     - Marketing budget: 25% of revenue
        #     - Distribution costs: 15% of revenue
        #     - Target profit margin: 35-40%

        #     **Competitive Positioning:**
        #     - 15% premium over average competitor pricing
        #     - Positioned as premium quality option
        #     - Justified by superior features and materials

        #     ### 📊 Revenue Projections

        #     **Year 1 Targets:**
        #     - 1,000 units sold
        #     - Average selling price: $225
        #     - Total revenue: $225,000
        #     - Gross profit: $78,750

        #     **Year 2 Targets:**
        #     - 5,000 units sold
        #     - Average selling price: $240
        #     - Total revenue: $1,200,000
        #     - Gross profit: $420,000""",
                        
        #                 launch_plan=f"""## Launch Plan for {request.product_name}

        #     ### 🚀 Pre-Launch Phase (Months 1-2)

        #     **Week 1-2: Foundation**
        #     - [ ] Finalize product specifications and features
        #     - [ ] Set up manufacturing partnerships
        #     - [ ] Establish quality control processes
        #     - [ ] Create brand identity and guidelines

        #     **Week 3-4: Digital Presence**
        #     - [ ] Launch website and landing pages
        #     - [ ] Set up social media accounts
        #     - [ ] Create content calendar
        #     - [ ] Develop email marketing sequences

        #     **Week 5-6: Content Creation**
        #     - [ ] Product photography and videos
        #     - [ ] Marketing copy and descriptions
        #     - [ ] Social media content library
        #     - [ ] Press kit and media materials

        #     ### 🎯 Launch Phase (Month 3)

        #     **Week 1: Soft Launch**
        #     - [ ] Launch to early adopters and beta testers
        #     - [ ] Gather feedback and testimonials
        #     - [ ] Optimize website and conversion funnels
        #     - [ ] Monitor and adjust pricing strategy

        #     **Week 2-4: Full Launch**
        #     - [ ] Public launch announcement
        #     - [ ] Media interviews and coverage
        #     - [ ] Influencer partnerships activation
        #     - [ ] Social media campaign launch

        #     ### 📊 Success Metrics
        #     - 1,000 pre-orders before launch
        #     - 5,000 units sold in first month
        #     - 25% conversion rate on website
        #     - 4.5+ star average customer rating""",
                        
        #                 marketing_content=f"""## Marketing Content for {request.product_name}

        #     ### 📱 Social Media Posts

        #     **Launch Announcement:**
        #     "🚀 The future is here! Introducing {request.product_name} - the revolutionary solution designed for {request.target_market.lower()}.

        #     ✨ What makes it special?
        #     • Cutting-edge technology
        #     • Premium quality materials
        #     • Smart features that adapt to your lifestyle
        #     • Eco-friendly design

        #     🎯 Perfect for busy professionals who demand excellence.

        #     #{request.product_name.replace(' ', '')} #Innovation #PremiumQuality"

        #     ### 📧 Email Marketing

        #     **Subject: Welcome to the {request.product_name} Family! 🚀**

        #     Hi [Name],

        #     Welcome to the future! You're now part of an exclusive group of innovators who will be the first to experience {request.product_name}.

        #     What to expect:
        #     • Early access to product updates
        #     • Exclusive behind-the-scenes content
        #     • Special launch pricing
        #     • VIP customer support

        #     Ready to transform your daily routine?

        #     Best regards,
        #     The {request.product_name} Team"""
        # )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating launch plan: {str(e)}")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "Product Launch Assistant API is running",
        "version": "1.0.0"
    }

# Root endpoint for API information
@app.get("/")
async def root():
    return {
        "message": "Product Launch Assistant API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/health"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("🚀 Starting Product Launch Assistant API")
    print(f"🔧 API Docs: http://localhost:{port}/api/docs")
    print(f"❤️  Health: http://localhost:{port}/api/health")
    print("=" * 50)
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info",
        access_log=True
    ) 