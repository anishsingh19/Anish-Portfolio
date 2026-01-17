// --- DATA ---
const projectsData = [
    {
        title: "Enterprise RAG Chatbot",
        media: [
            { type: 'image', url: 'https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/rag1.jpeg' },
            { type: 'image', url: 'https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/rag2.jpeg' }
        ],
        description: "A production-ready RAG chatbot using LangChain, Gemini Pro, and ChromaDB for context-aware, document-grounded question answering.",
        details: [
            "Architected a full-stack RAG application using <strong>LangChain</strong> to orchestrate the retrieval and generation pipeline with <strong>Google's Gemini Pro</strong> model for both embeddings and text generation.",
            "Developed a robust <strong>FastAPI backend</strong> with Pydantic for data validation, featuring endpoints for chat, system health, and a crucial <strong>/reindex_kb</strong> endpoint to reload knowledge base documents in real-time.",
            "Engineered a flexible data ingestion module supporting multiple document formats (including <strong>.pdf, .docx, and .csv</strong>) and implemented <strong>ChromaDB</strong> as the local vector store for efficient semantic search.",
            "Implemented <strong>in-memory conversational memory</strong> to provide session-based context, enabling more natural and coherent follow-up questions from the user, with real-time source attribution for answers.",
            "Designed a clean, mobile-responsive user interface with <strong>TailwindCSS</strong> for a modern chat experience and ensured production-readiness with a fully <strong>containerized Docker environment</strong> for the backend service."
        ],
        skills: ["RAG", "LangChain", "Gemini Pro", "ChromaDB", "FastAPI", "TailwindCSS", "Docker", "Pydantic", "Vector Databases", "NLP", "REST API", "Google Generative AI"]
    },
    {

        title: "AI-Fueled E-commerce Analytics & Sales Forecasting System",
        media: [
            { type: 'image', url: 'https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/ecommerce1.jpeg' },
            { type: 'image', url: 'https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/ecommerce2.jpeg' }
        ],
        description: "An AI-powered platform for e-commerce analytics and sales forecasting, leveraging Facebook Prophet and interactive dashboards to drive revenue strategy and reduce stockouts.",
        details: [
            "Developed an analytics system that connects to an *optimized SQLite3 database* (ecommerce.db) with settings like PRAGMA journal_mode=WAL for better concurrency and a PRAGMA cache_size=-2000 (2MB cache) to store temporary tables in memory (PRAGMA temp_store=MEMORY). The database schema includes a sales table with order_id, customer_id, product_id, purchase_date, amount, product_category, payment_method, and shipping_country, along with a product_inventory table.",
            "Features a data generation module capable of creating *realistic historical data* for 60 days, simulating 50 customers per day, and incorporating seasonal patterns such as a weekday_factor (1.0 for weekdays, 0.7 for weekends) and hour_factors (e.g., 'morning': 0.8, 'afternoon': 1.2, 'evening': 1.5, 'night': 0.5). It generates 10 diverse products including 'Laptop' (priced at $1200), 'Smartphone' ($800), 'Headphones' ($200), 'T-shirt' ($30), 'Jeans' ($80), 'Sneakers' ($120), 'Coffee Maker' ($150), 'Backpack' ($50), 'Watch' ($300), and 'Tablet' ($600).",
            "Generates *comprehensive sales metrics* including total_revenue, average_order_value, total_orders, unique_customers, and identifies the top 5 products by revenue, as well as daily_trends for orders and revenue. For example, 'Laptop' is identified as a top product with significant revenue. This data is cached for 5 minutes (cache_timeout=300) using lru_cache to enhance performance.",
            "Performs *advanced RFM (Recency, Frequency, Monetary) analysis* by calculating additional metrics like avg_order_value, category_diversity (count of distinct product categories), and active_months (count of distinct months with purchases). Customer segmentation is performed using *K-Means clustering* with 4 defined clusters on recency, frequency, and monetary scores after applying percentile scores (1-5).",
            "Creates an interactive *Plotly dashboard* with a make_subplots layout (rows=3, cols=2, height=1200, width=1200). The dashboard includes a 'Daily Revenue Trend' (line chart), 'Customer Segments' (pie chart, e.g., showing 39.9% in one segment), 'Top Products by Revenue' (bar chart), 'Payment Method Distribution' (pie chart, e.g., 25.3% for one method), 'Hourly Sales Pattern' (line chart) based on strftime('%H', purchase_date), and 'Geographic Distribution' (bar chart) based on shipping_country.",
            "Forecasts future sales for 30 periods using the *Facebook Prophet* model, configured with yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, and specific prior scales for changepoint (0.05), seasonality (10), and holidays (10). The forecast includes ds (date), yhat (predicted value), yhat_lower, and yhat_upper for confidence intervals."
        ],
        skills: ["Streamlit", "Prophet", "Pandas", "NumPy", "Plotly Express", "Data Engineering", "Business Intelligence", "Predictive Analytics", "SQLite3", "Scikit-learn (KMeans)"]
    },
    {
        title: "AI-Powered Trading System with Risk Analytics",
        media: [
            { type: 'image', url: 'https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/stocks1.jpeg' },
            { type: 'image', url: 'https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/stocks2.jpeg' }
        ],
        description: "A real-time AI-driven algorithmic trading system deployed on Streamlit, providing live market data, technical indicators, and automated trade execution with robust risk management protocols.",
        details: [
            "The system initializes with a default set of symbols including 'BTC-USD', 'ETH-USD', 'AAPL', and 'GOOGL', and an initial_capital of $100,000.0. The system logs its initialization with the specified symbols.",
            "Fetches historical market data for all specified symbols using yfinance, requesting a period='60d' and interval='1h' to provide granular data for indicator calculations. Data fetching progress is logged for each symbol, including warnings if no data is received.",
            "Calculates a comprehensive set of *technical indicators* including SMA (20-period and 50-period), EMA (12-period and 26-period), MACD (MACD line, Signal line, and Histogram), RSI (14-period), and Bollinger Bands (Middle, Upper with +2*std, and Lower with -2*std). This process includes error handling for robustness.",
            "Generates trading signals based on combined conditions: a strong buy signal (value 1) when RSI < 30 and Close < BB_Lower, and a strong sell signal (value -1) when RSI > 70 and Close > BB_Upper. Additional signals (value +/- 0.5) are derived from MACD crossovers (e.g., MACD > MACD_Signal) with price confirmation against SMA_20.",
            "Implements an EnhancedRiskManager with configurable parameters such as confidence_level (0.95), max_position_size (0.2), and max_drawdown_limit (0.15). It calculates base risk metrics including VaR (np.percentile(returns, (1 - confidence_level)*100)), CVaR, Sharpe Ratio (returns.mean() / returns.std() * np.sqrt(252)), and Max Drawdown. Enhanced metrics like Sortino Ratio, Downside Risk, Calmar Ratio, and Omega Ratio are also computed.",
            "Features an AdvancedVisualizationEngine to create rich graphical reports. This includes a *correlation heatmap* of asset returns (zmin=-1, zmax=1, colorscale='RdBu'), which dynamically includes the generation date. For instance, it correctly identifies an AAPL-GOOGL correlation of 0.354 and a BTC-USD/ETH-USD correlation of 0.775.",
            "Generates a *performance dashboard* using plotly.subplots.make_subplots with a rows=2, cols=2 layout, height=1000, and width=1200. This dashboard displays 'Cumulative Returns', 'Risk Metrics' (bar charts for Sharpe, Sortino, Calmar ratios using px.colors.qualitative.Set3), 'Daily Returns Distribution' (histograms with 50 bins), and 'Rolling Volatility (20-day)' plots.",
            "Provides a detailed text report (_generate_text_report) summarizing risk analysis for each asset. For example, for BTC-USD, it reports a Sharpe Ratio of 0.036, Sortino Ratio of 0.035, Maximum Drawdown of -16.15%, Value at Risk (95%) of -0.88%, Conditional VaR of -1.47%, Calmar Ratio of 0.021, and Omega Ratio of 1.007."
        ],
        skills: ["Streamlit", "NumPy", "Pandas", "PyTorch", "Scikit-learn", "Plotly", "Alpaca API", "Technical Indicators (MACD, RSI, Bollinger Bands, SMAs)", "Financial Analytics", "VaR", "Monte Carlo Simulation", "Real-Time Systems", "Quantitative Finance", "Time Series Forecasting", "Yfinance"]
    },
    {
        title: "AI Services Toolkit Pro (Multi-Modal AI Assistant)",
        media: [
            { type: 'image', url: 'https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/toolkit.jpeg' },
            { type: 'image', url: 'https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/toolkit1.jpeg' }
        ],
        description: "Architected and deployed a comprehensive, integrated Multi-Modal AI Toolkit on Hugging Face Spaces, integrating 9 state-of-the-art Transformer pipelines for diverse AI capabilities.",
        details: [
            "Features a robust application configuration (Config dataclass) with VERSION: \"1.0\", LAST_UPDATED: \"2025-02-10 18:22:33\", and flags for ENABLE_CUDA and ENABLE_TENSORRT. Key processing parameters include MAX_BATCH_SIZE: 32, MIN_BATCH_SIZE: 1, MEMORY_THRESHOLD: 0.85 (for CUDA memory optimization), CHUNK_SIZE: 640 (for image processing), MIN_CONFIDENCE: 0.25, NMS_THRESHOLD: 0.45, WHISPER_CHUNK_SIZE: 1024, and WHISPER_OVERLAP: 256. The default ONNX path is yolov5s.onnx.",
            "Includes a CUDAManager for enhanced CUDA resource management, utilizing cuda.Stream() for concurrent operations and torch.cuda.amp.GradScaler for mixed-precision training (if CUDA is enabled). It actively optimizes GPU memory by clearing the cache and collecting garbage when current_memory exceeds the MEMORY_THRESHOLD, logging the memory optimization process (e.g., 'Memory optimized Usage: 0.20%'). The system logs CUDA availability (e.g., 'CUDA Available: True') and the number/names of detected GPUs (e.g., 'GPU 0: NVIDIA A100-SXM4-40GB').",
            "Integrates *YOLOv5* for object detection, with a function to export_yolo_to_onnx() that loads yolov5s from ultralytics/yolov5, exports it to the specified ONNX path, and verifies the model. The TensorRTYOLO class then builds a TensorRT optimized engine from this ONNX model, configuring an optimization_profile to support dynamic batch sizes (min_shape=1, opt_shape=16, max_shape=32). It uses cuda.mem_alloc for efficient input/output memory allocation on GPU.",
            "Implements MultiGPUWhisper for audio processing, loading a 'tiny' Whisper model. If NUM_GPUS > 1, it utilizes torch.nn.DataParallel for parallel processing across GPUs. It processes audio by splitting it into chunks and asynchronously processing each chunk on a dedicated GPU device (cuda:i) before merging the results.",
            "Developed as a *FastAPI* application (title=\"Advanced ML Processor\", version=config.VERSION, description=\"Optimized ML processing with YOLO and Whisper\") with CORSMiddleware configured to allow all origins, credentials, methods, and headers. It exposes a /process-images endpoint that handles multiple UploadFile inputs for image inference and a /ws/audio WebSocket endpoint utilizing fastapi-websocket-pubsub for real-time audio transcription feedback and results streaming.",
            "An enhanced /health endpoint provides detailed system status, including gpu_info (device ID, name, memory used/total in MB, utilization in percent), memory_stats (recent usage, last cleanup timestamp, current threshold), and confirms if tensorrt_enabled and multi_gpu_enabled are true. It also reports the status and configuration of the 'yolo' and 'whisper' models, e.g., 'yolo' status: 'loaded', 'onnx_path': 'yolov5s.onnx', 'batch_size': 32."
        ],
        skills: ["FastAPI", "Streamlit", "Hugging Face Transformers", "PyTorch", "soundfile", "librosa", "Docker", "Full-Stack Development", "MLOps", "Whisper API", "YOLOv5", "NLP", "Speech-to-Text", "Text-to-Speech", "Real-Time Systems", "Computer Vision", "TensorRT", "ONNX", "CUDA", "WebSockets"]
    },
    {
        title: "Hybrid Predictive Maintenance System",
        media: [
            { type: 'image', url: 'https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/hybrid1.jpeg' },
            { type: 'image', url: 'https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/hybrid2.jpeg' }
        ],
        description: "Developed and deployed an integrated Hybrid Predictive Maintenance system on Streamlit, combining supervised learning (LSTM) and reinforcement learning for optimal maintenance recommendations.",
        details: [
            "The system is configured via SupervisedConfig, AdvancedRLConfig, and HybridSystemConfig dataclasses. Key parameters include: sequence_length=100, feature_dim=18 (though later context implies 10 features, 18 is from config) for supervised learning; state_dim=4, action_dim=4, hidden_units=256, learning_rate=0.0001, gamma=0.99, lambda_gae=0.95, clip_ratio=0.2, critic_loss_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, num_agents=4, batch_size=64, update_epochs=10, reward_scale=1.0 for RL; and update_interval=3600 seconds, prediction_horizon=24 hours, confidence_threshold=0.8, max_concurrent_predictions=10, alert_threshold=0.7, cost_threshold=50000, max_history_size=10000, visualization_enabled=True, report_format=\"detailed\" for the hybrid system.",
            "Generates synthetic data for training with num_samples=1000 and feature_dim=10. The features include vibration, temperature, pressure, current, voltage, rpm, oil_level, humidity, acoustic, and magnetic_field. The data generation simulates health_score, failure_prob, and rul as targets based on sine waves and random walks.",
            "Employs a *Multi-Agent PPO (Proximal Policy Optimization)* system, with PPOActorCritic networks for each of the 4 agents. The actor-critic network uses shared dense layers with 'relu' activation, policy layers ending in 'softmax' for action distribution, and value layers ending in a single dense unit. It selects actions using tf.random.categorical for training and tf.argmax for inference.",
            "Features a RewardModel with configurable cost_weights for maintenance (1.0), failure (10.0), and downtime (5.0). It defines specific maintenance costs (e.g., action 0: $0.0, action 1: $100.0, action 2: $500.0, action 3: $1000.0) and downtime costs (e.g., action 0: $0.0, action 1: $2.0, action 2: $4.0, action 3: $8.0).",
            "Predicts machine health metrics (health_score, failure_prob, rul) from sensor data, with an example output for a monitored machine (Machine 0) showing a health score of 0.612, failure probability of 0.388, and RUL of 61.2. The prediction for a given machine also includes a maintenance action (e.g., Action: 1 for Machine 0) and a value estimate (e.g., 0.057 for Machine 0).",
            "Incorporates an ExplainabilityModule (conceptualized with *SHAP and LIME*) that provides insights into predictions by highlighting top contributing factors based on randomly generated importance scores in the example. For Machine 0, the top factors are magnetic_field: 0.258, temperature: 0.182, and humidity: 0.157.",
            "Visualizes results through Matplotlib/Seaborn plots for 'Health Metrics', 'Maintenance Decisions' (scatterplot), 'Maintenance Costs' (line plot), and 'Feature Importance' (bar plot). These visualizations are saved to the model_checkpoint_dir (e.g., /content/hybrid_checkpoints/visualization_0.png). The system also saves final metrics to system_metrics.json."
        ],
        skills: ["Streamlit", "TensorFlow", "Keras", "NumPy", "Pandas", "SQLite3", "Deep Learning", "Reinforcement Learning (PPO)", "SHAP", "LIME", "Plotly", "Python", "Data Analytics", "Anomaly Detection", "Multi-Agent Systems"]
    },
    {
        title: "Customer Churn Prediction and API Deployment",
        media: [
            { type: 'image', url: 'https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/churn1.jpeg' },
            { type: 'image', url: 'https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/churn2.jpeg' }
        ],
        description: "Architected and deployed an integrated Customer Churn Prediction system on Streamlit with a FastAPI backend for model inference, achieving high accuracy and efficient real-time predictions.",
        details: [
            "Generates a synthetic customer churn dataset containing n_samples=2000 entries, with features such as age, gender, income, contract_type, monthly_charges, total_services, contacts_count, and complaints_count. Churn probability is engineered based on rules like 0.1 * (age > 50), 0.2 * (monthly_charges > 100), 0.3 * (total_services < 2), and 0.2 * (complaints_count > 2).",
            "Performs robust data preprocessing using ColumnTransformer to apply StandardScaler for numerical features (age, income, monthly_charges, total_services, contacts_count, complaints_count) and OneHotEncoder (with handle_unknown=\"ignore\" and sparse_output=False) for categorical features (gender, contract_type).",
            "Addresses class imbalance by applying *SMOTE (Synthetic Minority Over-sampling Technique)* with random_state=42 to the processed data before splitting into training and testing sets (test_size=0.2).",
            "Trains and evaluates two distinct machine learning models: a *Random Forest Classifier* (n_estimators=100, random_state=42) and a *Neural Network* built with *TensorFlow/Keras*. The Neural Network architecture includes Dense(64, activation='relu'), BatchNormalization(), Dropout(0.3), Dense(32, activation='relu'), and a final Dense(1, activation='sigmoid') layer. It's compiled with Adam(0.001) optimizer, binary_crossentropy loss, and trained for 50 epochs with a batch size of 32.",
            "Evaluates model performance comprehensively, calculating rf_accuracy and nn_accuracy on the test set (e.g., Random Forest Accuracy: 0.9416, Neural Network Accuracy: 0.9500 in example output). Visualizations generated and saved to the churn_project/ directory include 'Feature Importances' (e.g., 'monthly_charges' 0.23, 'total_services' 0.21, 'income' 0.17), 'Confusion Matrices' for both models, 'ROC Curves Comparison' with AUC scores (e.g., Random Forest AUC=0.949, Neural Network AUC=0.957), and an 'Impact of Classification Threshold on Different Metrics' (accuracy, precision, recall, f1 for thresholds 0.1 to 0.9).",
            "Deploys a high-performance *RESTful API* using *FastAPI* and *Uvicorn*, served at http://localhost:8000/predict/. The /predict/ endpoint accepts PredictionInput (a Pydantic model with features as a dictionary) and returns timestamp, user (e.g., '03ADY'), random_forest_prediction, neural_network_prediction, and their respective probabilities (rf_probability, nn_probability). Example test cases (high_risk_customer, low_risk_customer, medium_risk_customer) with detailed feature sets are included to demonstrate API functionality and are visualized in a 'Churn Predictions by Model and Customer Profile' bar chart. All test results are saved to api_test_results.json."
        ],
        skills: ["FastAPI", "Streamlit", "Scikit-learn", "Pandas", "NumPy", "imblearn (SMOTE)", "Random Forest", "Neural Networks", "MLOps", "Model Deployment", "REST API", "Python", "Classification", "TensorFlow", "Uvicorn"]
    }
];

const playgroundAppsData = [
    {
        title: "Enterprise RAG Chatbot Demo",
        description: "An interactive demo of the RAG chatbot. Upload your own documents or use the default knowledge base to ask questions and get context-aware answers from the AI.",
        url: "https://rag-chatbot-orcin.vercel.app/",
        image: "https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/rag2.jpeg"
    },
    {
        title: "AI-Powered Customer Churn Prediction",
        description: "An interactive Streamlit application demonstrating a machine learning model that predicts customer churn, allowing users to input customer data and see real-time predictions.",
        url: "https://ai-customer-churn-prediction.streamlit.app/",
        image: "https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/churn2.jpeg"   
    },
    {
        title: "AI-Fueled E-commerce Analytics & Sales Forecasting System",   
        description: "A live Streamlit dashboard providing comprehensive e-commerce analytics, including sales trends, customer segmentation (RFM), and interactive visualizations for business insights.",
        url: "https://ecommerce-sales-analysis-dashboard.streamlit.app/",
        image: "https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/ecommerce2.jpeg"   
    },
    {
        title: "AI Services Toolkit Pro",   
        description: "Explore a suite of multi-modal AI models, including sentiment analysis, summarization, and image captioning, all served through a user-friendly interface.",   
        url: "https://multi-ai-toolkit.streamlit.app/",   
        image: "https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/toolkit1.jpeg"   
    },
    {
        title: "Hybrid Predictive Maintenance Dashboard",   
        description: "An interactive Streamlit dashboard showcasing real-time predictive maintenance insights, including anomaly detection and remaining useful life (RUL) predictions for industrial equipment.",   
        url: "https://smart-predictive-maintenance.streamlit.app/",
        image: "https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/hybrid2.jpeg"   
    },
    {
        title: "AI-Powered Live Trading System",
        description: "An interactive Streamlit application demonstrating a deep learning-based trading system with real-time market data, technical indicators, and robust risk management.",   
        url: "https://trading-system-with-risk-analysis.streamlit.app/",   
        image: "https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/stocks2.jpeg"   
    }
];

const skillsData = [
    { title: 'Programming Languages & Frameworks', skills: ['Python (NumPy, Pandas)', 'SQL', 'FastAPI', 'Flask', 'Data Structures', 'Algorithms'] },   
    { title: 'Machine Learning', skills: ['Scikit-learn', 'XGBoost', 'Random Forests', 'SVMs', 'Regression', 'Classification', 'Clustering', 'Feature Engineering', 'Model Evaluation', 'SMOTE', 'Reinforcement Learning'] },   
    { title: 'Deep Learning', skills: ['TensorFlow', 'PyTorch', 'Keras', 'CNNs', 'RNNs (LSTMs, GRUs)', 'Transformers', 'Transfer Learning', 'GANs'] },   
    { title: 'MLOps & Deployment', skills: ['Docker', 'Kubernetes (basics)', 'Render', 'Heroku', 'Git', 'Uvicorn', 'RESTful API', 'Microservices', 'CI/CD', 'Model Versioning', 'Monitoring', 'Hugging Face Spaces'] },   
    { title: 'Big Data & Databases', skills: ['Hadoop', 'Spark', 'Apache Kafka', 'PostgreSQL', 'MySQL', 'SQLAlchemy', 'MongoDB', 'SQLite3'] },   
    { title: 'Data Visualization & BI', skills: ['Plotly', 'Matplotlib', 'Seaborn', 'Dash', 'Tableau (conceptual)', 'Streamlit'] },   
    { title: 'Specialized Tools', skills: ['SHAP', 'LIME', 'Prophet', 'Whisper API', 'YOLOv5', 'OpenAI API', 'LLMs (ChatGPT)', 'Financial Modeling', 'Monte Carlo Simulation', 'A/B Testing', 'Alpaca API', 'Technical Indicators (MACD, RSI, Bollinger Bands, SMAs)'] }   
];

const blogPostsData = [
    {   
        title: "Crafting My Digital Footprint: A Technical Deep Dive into Portfolio Development",   
        date: "2025-07-08", // Current Date
        image: "https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/Build-Your-Portfolio.png", // Updated with your provided image!
        tags: ["Portfolio", "Web Development", "MLOps", "Journey", "Frontend", "Backend"],   
        content: `
            <p>The journey of building a personal portfolio is far more than just compiling past projects; it's a technical deep dive into system architecture, deployment pipelines, and user experience design. This platform itself stands as a testament to my capabilities, a living showcase of the principles I advocate for in AI and Data Science. My primary motivation was to create a dynamic, engaging space that extends beyond static résumés, providing interactive demonstrations and tangible insights into my technical thought process.</p>
            <p>From a foundational perspective, the entire application is architected around a robust web development stack. HTML provides the semantic structure, ensuring accessibility and search engine optimization. CSS, augmented by the utility-first framework <strong>Tailwind CSS</strong>, enables rapid and consistent styling, allowing for a highly responsive and aesthetically pleasing user interface across various devices. The dynamic and interactive elements that bring this portfolio to life are powered by pure <strong>JavaScript</strong>, demonstrating proficiency in client-side scripting and DOM manipulation.</p>
            <p>A key design consideration was the seamless integration of various data sources and interactive components. Project details, skill sets, and blog content are managed programmatically within JavaScript arrays, facilitating easy updates and a consistent data structure. Features like the dynamic typing animation on the hero section and the interactive modal for project details were meticulously implemented to enhance user engagement and provide a polished experience. The background particle animation, while seemingly simple, involves canvas manipulation to create a visually appealing, low-resource effect.</p>
            <p>Beyond the immediate user interface, the deployment strategy for this portfolio embodies critical <strong>MLOps principles</strong>. The entire codebase is version-controlled using <strong>Git</strong> and hosted on <strong>GitHub</strong>, providing a robust system for tracking changes, collaborating (even if just with myself!), and maintaining a history of development. The selection of <strong>Vercel</strong> for continuous deployment streamlined the integration process significantly. Every commit to the \`main\` branch of the GitHub repository automatically triggers a new build and deployment on Vercel, demonstrating a practical application of CI/CD pipelines. This automation ensures that updates are delivered swiftly and reliably to the live environment, reflecting the agile development methodologies I apply to more complex AI systems.</p>
            <p>This portfolio’s development journey, while focused on showcasing Machine Learning and Data Science, also served as an invaluable learning experience in full-stack web development and robust deployment practices. It reinforced the understanding that a successful technical project encompasses not just sophisticated algorithms or data models, but also the resilient infrastructure, meticulous deployment strategies, and intuitive user interfaces that bring those innovations to life.</p>
        `   
    },
    {   
        title: "Architecting Intelligence: My Journey in Tech",   
        date: "2025-06-26",   
        image: "https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/toolkit.jpeg",   
        tags: ["Career", "Reflection", "MLOps", "Generative AI"],   
        content: `
            <p>Welcome to my digital sanctuary, a space where I articulate my journey as an AI and Data Science enthusiast. My pursuit of a B.Tech at A. C. Patil College of Engineering has been a crucible for mastering the theoretical underpinnings of intelligent systems. However, my true passion transcends mere comprehension; it lies in the architecting and deployment of intelligent, actionable solutions that yield tangible, real-world impact. This platform serves as a chronicle of my exploration, the ambitious projects that ignite my curiosity, and the invaluable lessons gleaned from each technical endeavor.</p>
            <p>My academic trajectory and professional engagements have been rigorously focused on acquiring mastery across the entire machine learning lifecycle. This encompasses the meticulous construction of resilient data pipelines, leveraging distributed computing paradigms such as <strong>Hadoop and Spark</strong>, to the strategic deployment of scalable microservices powered by <strong>Python and FastAPI</strong>. My objective consistently revolves around engineering robust, end-to-end systems that are not only performant but also maintainable and extensible.</p>
            <blockquote class="border-l-4 border-indigo-500 pl-4 text-slate-300 italic">"The goal is to turn data into information, and information into insight." - Carly Fiorina. This profound statement encapsulates the very essence of my philosophical approach to data science and artificial intelligence.</blockquote>
            <p>Beyond the foundational backend infrastructure, my core intellectual curiosity resides in the intricate dynamics of the models themselves. I have immersed myself in a diverse spectrum of machine learning paradigms, ranging from conventional predictive analytics, such as the customer churn models meticulously constructed using <strong>Random Forests</strong>, to the avant-garde frontiers of deep learning, incorporating architectures like <strong>Convolutional Neural Networks (CNNs)</strong>, <strong>Long Short-Term Memory (LSTMs)</strong>, and <strong>Transformers</strong>. A project that I hold in particularly high esteem is the <strong>AI Services Toolkit Pro (Multi-Modal AI Assistant)</strong>, a complex integration where I synergized OpenAI's Whisper for state-of-the-art transcription capabilities with YOLOv5 for real-time object detection. The technical challenge of seamlessly merging these disparate yet powerful technologies into a singular, cohesive, and user-centric application was exhilarating and deeply rewarding.</p>
            <p>My commitment extends beyond mere model development to ensuring that these intelligent systems are not only robustly engineered but also demonstrably scalable and inherently maintainable. This necessitates a stringent adherence to <strong>MLOps</strong> principles, encompassing the meticulous implementation of Continuous Integration/Continuous Deployment (CI/CD) pipelines, comprehensive model versioning strategies, and diligent continuous monitoring protocols. My hands-on experience with <strong>Docker</strong> and platform-as-a-service providers like <strong>Render and Heroku</strong> has been pivotal in facilitating the efficient deployment of these intrinsically complex systems. It is my unwavering conviction that a meticulously engineered deployment strategy is as critically important as the predictive or generative power of the model itself in delivering tangible business value.</p>
            <p>I am particularly enthused by the rapid advancements in <strong>Generative AI and Large Language Models (LLMs)</strong>, and I have proactively engaged in exploring their transformative applications, as evidenced by my specialized certification in this cutting-edge domain. The inherent capacity of these technologies to autonomously create novel content, synthesize vast quantities of information, and even generate executable code unlocks an unprecedented array of possibilities for the next generation of AI solutions, and I am eager to contribute to their evolution.</p>
        `   
    },
    {   
        title: "Deep Dive: Handling Imbalance in Churn Prediction",   
        date: "2025-06-20",   
        image: "https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/churn1.jpeg",   
        tags: ["Deep Dive", "Machine Learning", "Classification", "Data Preprocessing"],   
        content: `
            <p>One of the most pervasive and insidious challenges encountered in real-world classification problems, such as the critical task of customer churn prediction, is the inherent presence of severely imbalanced datasets. In such scenarios, the proportion of the minority class – for instance, customers who actually churn – is disproportionately small compared to the majority class (non-churning customers). Left unaddressed, this severe class imbalance can lead to a highly deceptive model that achieves a superficially high accuracy by merely predicting the dominant class for every instance, rendering it practically useless for identifying the crucial events of interest.</p>
            <p>In my <strong>Customer Churn Prediction and API Deployment</strong> project, I confronted this challenge directly and systematically. The linchpin of my approach was the strategic application of the <strong>SMOTE (Synthetic Minority Over-sampling Technique)</strong> algorithm. Unlike simplistic oversampling methods that merely duplicate existing minority class samples, SMOTE operates by intelligently generating new, synthetic samples. These synthetic data points are created by interpolating between existing minority class instances and their k-nearest neighbors in the feature space. This sophisticated generative approach effectively mitigates overfitting to specific minority samples and provides a richer, more diverse, and crucially, more balanced dataset for the machine learning model to effectively train on.</p>
            <p>The impact of this data re-balancing was profound and quantitatively significant. By meticulously applying SMOTE within the data preprocessing pipeline, I achieved a <strong>demonstrable increase of 25% in the model's recall for the churn class</strong>. This enhancement was not just a statistical improvement; it directly translated into the model becoming substantially more adept at its primary strategic objective: accurately identifying customers who are genuinely at risk of attrition. This heightened ability to detect potential churners proactively within a 30-day window allowed for the timely deployment of targeted retention strategies, moving from reactive mitigation to proactive customer engagement.</p>
            <p>This experience served as a powerful testament to a fundamental principle in machine learning: headline accuracy is rarely the sole arbiter of a model's real-world utility. A nuanced understanding of data characteristics and the strategic application of advanced preprocessing techniques are paramount to developing truly effective and impactful models. The combination of data engineering rigor and algorithmic selection was instrumental in achieving an F1-score of 0.87, signifying a robust balance between precision and recall, and enabling the identification of 75% of high-risk churners, thereby providing actionable intelligence for business stakeholders.</p>
        `   
    },
    {
        title: "The Power of Real-time Data in Trading",
        date: "2025-06-15",
        image: "https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/stocks1.jpeg",   
        tags: ["Finance", "Real-time Systems", "Data Engineering", "Trading"],   
        content: `
            <p>In the high-octane and perpetually evolving domain of financial trading, the acquisition and judicious utilization of real-time data transcend mere advantage; it constitutes an absolute strategic imperative. My <strong>AI-Powered Trading System with Risk Analytics</strong> project was meticulously engineered around this fundamental principle, demanding the construction of highly robust and ultra-low latency data ingestion pipelines. We leveraged distributed streaming technologies, notably <strong>Apache Kafka</strong>, to process an astounding throughput of <strong>over 10,000 data points per minute</strong>, ensuring immediate availability of market intelligence.</p>
            <p>This unparalleled real-time data processing capability was foundational and critical for several interconnected reasons:</p>
            <ul>
                <li><strong>Timely Decision Optimization:</strong> The instantaneous data feeds were directly channeled into our sophisticated LSTM (Long Short-Term Memory) models. This architectural design facilitated up-to-the-minute market analysis and predictive modeling, directly boosting the efficiency of trading decisions by a remarkable <strong>30%</strong>. The ability to react and strategize within milliseconds is paramount in volatile markets.</li>
                <li><strong>Dynamic Risk Mitigation:</strong> The seamless integration of live market data with crucial risk metrics, such as Value-at-Risk (VaR) and Sortino Ratio, empowered our system to react with unparalleled swiftness and precision to market volatility and unforeseen events. This proactive risk management approach was quantitatively demonstrated to lower simulated investment risk by <strong>22%</strong>, safeguarding capital and optimizing portfolio stability.</li>
                <li><strong>Sustained Competitive Superiority:</strong> In algorithmic trading, the capacity to process, analyze, and act upon market data faster than competitors translates directly into a formidable competitive edge. This accelerated data-to-action cycle consistently contributed to significant gains in simulated portfolio returns, capitalizing on ephemeral market inefficiencies.</li>
            </ul>
            <p>The paramount engineering challenge revolved around guaranteeing minimal data latency and maximizing throughput while maintaining data integrity. By architecting a solution around distributed streaming platforms, we successfully established a resilient, highly efficient, and fault-tolerant backbone for our intricate predictive models. This project unequivocally demonstrated that superior data engineering is not merely supportive but absolutely foundational to the successful implementation and sustained performance of advanced AI applications within the demanding landscape of quantitative finance. Our strategic integration with the <strong>Alpaca API</strong> further solidified this capability by providing streamlined and robust access to both historical and real-time stock data, enabling granular bar updates and facilitating seamless, high-frequency order submissions directly to trading venues.</p>
        `   
    },
    {
        title: "Interpretable AI: Beyond the Black Box",
        date: "2025-06-08",
        image: "https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/hybrid1.jpeg",   
        tags: ["AI Ethics", "Explainable AI", "Machine Learning", "Predictive Maintenance"],   
        content: `
            <p>As Artificial Intelligence models escalate in complexity, particularly within mission-critical applications such as predictive maintenance, comprehending the underlying rationale for a model's prediction becomes as strategically vital as the prediction itself. This paradigm defines the burgeoning field of Explainable AI (XAI). In my <strong>Hybrid Predictive Maintenance System</strong>, a core design objective was to demystify the inherently complex interplay of Deep Learning and Reinforcement Learning, rendering the model's decisions transparent and actionable through the strategic application of <strong>SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations)</strong>.</p>
            <p>SHAP offers a unified, game-theoretic framework for interpreting predictions, systematically assigning an importance value (Shapley value) to each input feature for a given prediction. This provides a global understanding of feature importance while maintaining local fidelity to individual predictions. Complementarily, LIME excels at explaining the predictions of any classifier or regressor by constructing a simpler, locally faithful, interpretable model around the specific instance to be explained. The synergistic application of these two techniques provides a multi-faceted view of model behavior, addressing both individual decision-making and overall model characteristics.</p>
            <ul>
                <li><strong>Precision in Anomaly Attribution:</strong> We were able to precisely identify and quantify the top 5 influential factors (e.g., specific sensor readings, operational parameters, environmental conditions) that most significantly contributed to a predicted equipment anomaly or impending failure. This granular insight transforms a mere "failure alert" into actionable diagnostic intelligence.</li>
                <li><strong>Cultivating Stakeholder Trust:</strong> Providing maintenance teams and operational managers with clear, human-understandable justifications behind predicted failures dramatically increased their confidence and trust in the AI system. This transparency is indispensable for fostering adoption and leveraging AI for critical operational decisions.</li>
                <li><strong>Iterative Model Enhancement:</strong> The insights derived from SHAP and LIME analyses were not just for reporting; they served as invaluable feedback loops for refining feature engineering strategies and optimizing the underlying model architecture. This iterative refinement process directly contributed to the <strong>30% improvement in prediction accuracy</strong>, underscoring the reciprocal relationship between interpretability and model performance.</li>
            </ul>
            <p>This unwavering commitment to interpretability ensures that our AI solutions are not merely powerful computational tools but are also actionable, trustworthy, and deeply integrated into operational workflows, effectively bridging the inherent gap between complex algorithms and the pragmatic demands of real-world industrial decisions. Within the Hybrid Predictive Maintenance System, this interpretability was pivotal for generating comprehensive simulation reports and providing granular insights that directly contributed to an estimated <strong>20% reduction in operational downtime costs</strong>, translating into substantial annual savings for industrial operations.</p>
        `   
    },
    {
        title: "Unlocking Multi-Modal AI: The Toolkit Approach",
        date: "2025-06-01",
        image: "https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/toolkit.jpeg",   
        tags: ["AI", "Multi-modal AI", "Transformers", "FastAPI", "Streamlit"],
        content: `
            <p>The contemporary landscape of Artificial Intelligence increasingly necessitates systems capable of processing and synthesizing information across disparate data modalities – text, speech, and images. Building and deploying state-of-the-art AI models that seamlessly handle this multi-modal complexity presents considerable engineering challenges. My <strong>AI Services Toolkit Pro (Multi-Modal AI Assistant)</strong> project was conceived and meticulously designed to address these challenges head-on, by establishing a unified, extensible platform for a diverse array of Transformer-based AI capabilities.</p>
            <p>The core architectural principle was the strategic integration of <strong>9 distinct Transformer pipelines sourced from Hugging Face</strong>, a leading ecosystem for pre-trained models. This integration encompassed a broad spectrum of AI tasks, including nuanced sentiment analysis, concise text summarization, precise image captioning, and advanced question-answering. This modular approach enabled the offering of a wide range of sophisticated AI services from a singular, cohesive, and high-performance application endpoint.</p>
            <p>Key aspects of the system's robust architecture include:</p>
            <ul>
                <li>A high-performance <strong>FastAPI backend</strong> meticulously engineered to handle asynchronous operations, ensuring maximal concurrency and responsiveness. This backend leverages <strong>Pydantic models</strong> for rigorous data validation and serialization, guaranteeing data integrity and type safety. AI services are exposed via clean, RESTful <code>/api</code> endpoints, facilitating easy integration with other systems.</li>
                <li>An intuitive and dynamic <strong>Streamlit frontend</strong> that provides a rich, interactive user experience. This interface allows users to seamlessly interact with various AI services, observe real-time processing feedback, review comprehensive API call histories, and monitor overall system status, enhancing user transparency and control.</li>
                <li>Implementation of advanced multi-modal features, including sophisticated <strong>Text-to-Speech (TTS)</strong> capabilities with dynamic speaker embeddings for personalized voice output, and highly accurate <strong>Speech-to-Text (STT)</strong> with automatic audio resampling to optimize transcription quality across diverse audio inputs. These accessibility enhancements demonstrably improved usability for an estimated <strong>5,000 daily users</strong>, significantly broadening the toolkit’s appeal and utility.</li>
            </ul>
            <p>This project serves as a comprehensive demonstration of a full end-to-end MLOps pipeline, spanning from intricate model integration and robust API development to seamless frontend deployment on scalable platforms such as <strong>Hugging Face Spaces</strong>. It unequivocally highlights the transformative power of strategically combining specialized AI models into a user-friendly, enterprise-ready toolkit, thereby rendering advanced AI capabilities both highly accessible and readily actionable for a diverse user base.</p>
        `
    },
    {
        title: "Mastering MLOps: From Code to Scalable Production AI",
        date: "2025-05-28",   
        image: "https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/mlops.jpeg", // New blog image
        tags: ["MLOps", "Deployment", "Scalability", "DevOps", "Production AI"],
        content: `
            <p>In the dynamic realm of Artificial Intelligence, the development of a powerful and accurate machine learning model represents merely the initial phase of a much broader, complex lifecycle. The quintessential challenge lies in transitioning these experimental models from development environments to reliable, high-performing, and scalable production systems. This critical transition is precisely where <strong>MLOps (Machine Learning Operations)</strong> emerges as an indispensable engineering discipline, acting as the bridge between theoretical models and real-world impact.</p>
            <p>My extensive experience, honed during a Python Backend Developer Internship and through various rigorous AI projects, has profoundly ingrained in me the paramount importance of robust MLOps practices. MLOps is not merely a collection of tools; it is a holistic engineering philosophy that unifies ML system development (Dev) with ML system operations (Ops). It meticulously encompasses a comprehensive spectrum of practices aimed at streamlining the entire machine learning lifecycle, from initial data collection and iterative model training to robust deployment, continuous monitoring, and perpetual improvement in a production setting. Key components that form the bedrock of an effective MLOps framework include:</p>
            <ul>
                <li><strong>Continuous Integration/Continuous Delivery (CI/CD):</strong> This pillar involves automating the entire pipeline for building, rigorously testing, and reliably deploying models. My direct involvement in orchestrating <strong>Docker-containerized deployments to platforms like Render and Heroku</strong> demonstrably reduced critical deployment cycles by a remarkable <strong>40%</strong>. This automation minimizes manual errors, accelerates iteration velocity, and ensures faster time-to-market for AI-driven features.</li>
                <li><strong>Model Versioning and Governance:</strong> Meticulously tracking and managing distinct versions of models, alongside their associated training data and evaluation metrics, is paramount for ensuring reproducibility, facilitating efficient debugging, and enabling seamless rollbacks when necessary. This robust versioning is crucial for regulatory compliance and audit trails in production AI systems.</li>
                <li><strong>Proactive Monitoring and Alerting:</strong> Continuous, real-time observation of model performance, detection of data drift and concept drift, and comprehensive system health monitoring in production are essential. This proactive vigilance allows for the rapid identification and timely resolution of issues, preventing degradation of model efficacy and ensuring consistent business value.</li>
                <li><strong>Scalable Infrastructure Design:</strong> Architecting systems that can seamlessly handle escalating inference loads and data volumes is fundamental for production AI. My direct experience in building scalable microservices using <strong>FastAPI and Flask</strong> resulted in an impressive <strong>25% improvement in API response times</strong> and demonstrated the capability to handle <strong>over 10,000 daily requests with an exceptional 99.9% uptime</strong>, a clear testament to robust architectural design and effective load balancing.</li>
            </ul>
            <p>Furthermore, the advent of platforms like <strong>Hugging Face Spaces</strong> significantly democratizes the application of MLOps principles, providing accessible and efficient avenues for deploying, sharing, and collaborating on machine learning models. By diligently mastering these MLOps tenets, AI solutions transcend the realm of mere innovative prototypes; they evolve into reliable, high-performing, and continuously valuable assets that deliver sustained strategic advantages and profound business impact.</p>
        `   
    },
    {
        title: "Data Engineering for AI: Building Resilient Data Pipelines",
        date: "2025-05-20",   
        image: "https://raw.githubusercontent.com/anishsingh19/Anish-Portfolio/main/images/datascience.jpeg", // New blog image
        tags: ["Data Engineering", "Big Data", "Apache Kafka", "Data Pipelines", "ETL"],
        content: `
            <p>At the foundational core of every highly effective Artificial Intelligence or Machine Learning system lies an undeniably robust, meticulously engineered, and perpetually reliable data pipeline. Without a continuous flow of clean, accessible, and high-fidelity data, even the most exquisitely sophisticated models are ultimately rendered impotent. My academic background, coupled with extensive hands-on project experience, has equipped me with profound expertise in architecting and implementing <strong>resilient data engineering solutions</strong> that serve as the lifeblood of intelligent systems.</p>
            <p>My specialized certifications in Data Engineering, particularly focusing on <strong>Hadoop & Spark</strong>, provided the essential theoretical and practical groundwork for comprehending and managing large-scale data processing paradigms. Key technologies that form the bedrock of modern AI-driven data pipelines are:</p>
            <ul>
                <li><strong>Apache Hadoop:</strong> A foundational framework for distributed storage (HDFS) and batch processing of colossal datasets. Hadoop's architecture enables the handling of petabyte-scale data, making it indispensable for foundational data lakes and historical data analysis for model training.</li>
                <li><strong>Apache Spark:</strong> A powerful and versatile unified analytics engine designed for large-scale data processing. Spark offers unparalleled speed and flexibility, making it ideal for a multitude of tasks including complex ETL (Extract, Transform, Load) operations, advanced analytics, and machine learning feature engineering on massive datasets, significantly outperforming traditional MapReduce in many scenarios.</li>
                <li><strong>Apache Kafka:</strong> A distributed streaming platform architected for building real-time data pipelines and streaming applications. In my <strong>AI-Powered Trading System with Risk Analytics</strong>, I leveraged Apache Kafka to establish ultra-low latency, real-time data ingestion pipelines capable of processing <strong>over 10,000 data points per minute</strong>. This real-time capability was paramount, significantly boosting decision efficiency and enabling immediate responsiveness to market dynamics.</li>
                <li><strong>SQL and NoSQL Databases:</strong> Proficiency extends to proficiently managing and optimizing various database systems, including relational databases like <strong>PostgreSQL and MySQL</strong> for structured data integrity and consistency, and NoSQL databases such as <strong>MongoDB</strong> for flexible, scalable storage of unstructured or semi-structured data. My work specifically utilizing the <strong>SQLAlchemy ORM</strong> in conjunction with PostgreSQL databases resulted in a <strong>quantifiable reduction of data retrieval time by 15%</strong>, optimizing application performance and responsiveness.</li>
            </ul>
            <p>The meticulous construction of these resilient and high-throughput data pipelines is not merely an operational necessity; it is a strategic differentiator. It ensures that AI models consistently receive the high-quality, timely, and relevant data they critically require to perform optimally and deliver accurate predictions or insights. This foundational layer, though often unseen by end-users, is unequivocally paramount to the success, reliability, and ultimately, the tangible business impact of any sophisticated AI initiative.</p>
        `   
    }
].sort((a, b) => new Date(b.date) - new Date(a.date)); // Sort by date descending for recent posts


// --- UI LOGIC ---
document.addEventListener('DOMContentLoaded', () => {
    // Function to create interactive covers (NO LONGER USED FOR PROJECT CARDS DIRECTLY, but kept as a fallback/example if needed elsewhere)
    const createInteractiveCover = (project) => {
        // This function is still defined but the projectsData no longer uses it for the main card display.
        // The project card will now directly use the first image from its 'media' array.
        // This function would only be called if a project explicitly had interactive_cover defined,
        // which we've removed for the main project display.
        const { type } = project.interactive_cover || {}; // Handle case where interactive_cover might not exist
        let svgContent = '';
        switch (type) {
            case 'dashboard':
                svgContent = `
                    <svg viewBox="0 0 300 150" fill="none" xmlns="http://www.w3.org/2000/svg" class="bg-gray-900">
                        <defs><filter id="glow"><feGaussianBlur stdDeviation="2.5" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
                        <rect width="300" height="150" fill="#0a192f"/>
                        <path d="M20,130 C50,20 80,110 140,80 S200,40 280,60" stroke="#a78bfa" stroke-width="1" class="svg-hidden draw-on-hover"/>
                        <path d="M20,130 C50,20 80,110 140,80 S200,40 280,60" stroke="#6366f1" stroke-width="2" style="filter:url(#glow);" class="svg-hidden draw-on-hover"/>
                        <text x="20" y="25" font-family="Inter, sans-serif" font-size="12" fill="#e2e8f0" class="font-bold">Sales Forecasting</text>
                    </svg>`;
                break;
            case 'trading':
                svgContent = `
                    <svg viewBox="0 0 300 150" fill="none" xmlns="http://www.w3.org/2000/svg" class="bg-gray-900">
                        <rect width="300" height="150" fill="#161a1d"/>
                        <path d="M30 110 L 80 40 L 130 80 L 180 60 L 230 90 L 280 30" stroke="#4f46e5" stroke-width="2" class="draw-on-hover"/>
                        <g class="svg-hidden">
                            <circle cx="80" cy="40" r="4" fill="#39ff14"/>
                            <circle cx="280" cy="30" r="4" fill="#dc143c"/>
                        </g>
                        <text x="20" y="25" font-family="Inter, sans-serif" font-size="12" fill="#e2e8f0" class="font-bold">LSTM Analysis</text>
                    </svg>`;
                break;
            case 'toolkit':
                svgContent = `
                    <svg viewBox="0 0 300 150" fill="none" xmlns="http://www.w3.org/2000/svg" class="bg-gray-900">
                        <rect width="300" height="150" fill="#111827"/>
                        <defs><filter id="toolkit-glow"><feGaussianBlur stdDeviation="2" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
                        <g class="toolkit-glow">
                            <path d="M150 50 L 120 67 L 120 101 L 150 118 L 180 101 L 180 67 Z" fill="rgba(139, 92, 246, 0.1)" stroke="#8b5cf6" stroke-width="1.5"/>
                            <text x="138" y="90" font-family="monospace" font-size="14" fill="#c4b5fd">API</text>
                        </g>
                        <g class="svg-hidden">
                            <path d="M110 67 L 80 84" stroke="#a78bfa" stroke-width="1"/>
                            <path d="M190 67 L 220 84" stroke="#a78bfa" stroke-width="1"/>
                            <path d="M110 101 L 80 118" stroke="#a78bfa" stroke-width="1"/>
                            <path d="M190 101 L 220 118" stroke="#a78bfa" stroke-width="1"/>
                            <text x="50" y="88" font-size="10" fill="white">Sentiment</text>
                            <text x="225" y="88" font-size="10" fill="white">Summarize</text>
                            <text x="50" y="122" font-size="10" fill="white">Caption</text>
                            <text x="225" y="122" font-size="10" fill="white">Generate</text>
                        </g>
                        <text x="20" y="25" font-family="Inter, sans-serif" font-size="12" fill="#e2e8f0" class="font-bold">AI Services Toolkit</text>
                    </svg>`;
                break;
            case 'maintenance':
                svgContent = `
                    <svg viewBox="0 0 300 150" fill="none" xmlns="http://www.w3.org/2000/svg" class="bg-gray-900">
                        <rect width="300" height="150" fill="#2d2d2d"/>
                        <g class="pm-gear" style="transform-origin: 80px 75px;">
                            <circle cx="80" cy="75" r="30" fill="none" stroke="#add8e6" stroke-width="4" stroke-dasharray="5 5"/>
                        </g>
                        <path d="M110 75 H 280" stroke="rgba(255,140,0,0.3)" stroke-width="2"/>
                        <path d="M110 75 H 280" stroke="#ff8c00" stroke-width="2" class="draw-on-hover"/>
                        <g class="svg-hidden">
                            <rect x="230" y="65" width="50" height="20" fill="#0a0a0a" stroke="#ff8c00"/>
                            <text x="238" y="79" font-size="10" fill="#ff8c00">RUL: OK</text>
                        </g>
                        <text x="20" y="25" font-family="Inter, sans-serif" font-size="12" fill="#e2e8f0" class="font-bold">Predictive Maintenance</text>
                    </svg>`;
                break;
            case 'churn':
                svgContent = `
                    <svg viewBox="0 0 300 150" fill="none" xmlns="http://www.w3.org/2000/svg" class="bg-gray-900">
                        <rect width="300" height="150" fill="#003638"/>
                        <g class="churn-dot-imbalanced">
                            ${[...Array(6)].map((_, i) => `<circle cx="${40 + i * 40}" cy="50" r="5" fill="#008080" class="churn-dot"/>`).join('')}
                            <circle cx="120" cy="100" r="5" fill="#ffbf00" class="churn-dot"/>
                        </g>
                        <g class="svg-hidden">
                            <circle cx="100" cy="95" r="5" fill="#ffbf00" class="churn-dot churn-dot-synthetic"/>
                            <circle cx="140" cy="105" r="5" fill="#ffbf00" class="churn-dot churn-dot-synthetic"/>
                        </g>
                        <text x="20" y="25" font-family="Inter, sans-serif" font-size="12" fill="#e2e8f0" class="font-bold">SMOTE Data Balancing</text>
                    </svg>`;
                break;
            default:
                // If no interactive_cover type is matched or provided, it would fall back to nothing here.
                // However, the project cards are now designed to just use the first 'media' image.
                svgContent = `<div class="w-full h-full bg-gray-800 flex items-center justify-center"><p class="text-slate-400">Project Image Placeholder</p></div>`;
        }
        return `<div class="interactive-cover-container">${svgContent}</div>`;
    };

    // Populate portfolio sections
    const projectsGrid = document.getElementById('projects-grid');
    projectsData.forEach((project, index) => {
        const card = document.createElement('div');
        card.className = "project-card card-bg rounded-2xl flex flex-col overflow-hidden transform hover:-translate-y-2 transition-transform duration-300 cursor-pointer";
        card.innerHTML = `
            <div class="interactive-cover-container">
                <img src="${project.media[0].url}" alt="${project.title} Cover" class="w-full h-full object-cover object-center" onerror="this.onerror=null;this.src='https://placehold.co/300x150/1e1b4b/c4b5fd?text=Image+Error';">
            </div>
            <div class="p-6 flex flex-col flex-grow">
                <div>
                    <h3 class="text-xl font-bold text-white mb-2">${project.title}</h3>
                    <p class="text-slate-400 mb-4 text-sm">${project.description}</p>
                </div>
                <div class="flex flex-wrap gap-2 mt-auto pt-4">
                    ${project.skills.slice(0, 3).map(skill => `<span class="tag rounded-md px-2 py-1 text-xs">${skill}</span>`).join('')}
                </div>
            </div>`;
        card.addEventListener('click', () => openModal(index));
        projectsGrid.appendChild(card);
    });
    const skillsGrid = document.querySelector('#skills .grid');
    skillsData.forEach(category => {
        const card = document.createElement('div');
        card.className = "card-bg p-6 rounded-2xl";
        card.innerHTML = `<h3 class="text-xl font-bold text-white mb-4">${category.title}</h3><div class="flex flex-wrap gap-2">${category.skills.map(skill => `<span class="tag rounded-md px-3 py-1 text-sm">${skill}</span>`).join('')}</div>`;
        skillsGrid.appendChild(card);
    });

    // Function to render Playground apps
    const renderPlaygroundApps = () => {
        const playgroundAppsGrid = document.getElementById('playground-apps-grid');
        playgroundAppsGrid.innerHTML = ''; // Clear existing content
        playgroundAppsData.forEach(app => {
            const appCard = document.createElement('div');
            appCard.className = "card-bg rounded-2xl flex flex-col overflow-hidden transform hover:-translate-y-2 transition-transform duration-300 cursor-pointer";
            appCard.innerHTML = `
                <img src="${app.image}" alt="${app.title}" class="w-full h-48 object-cover" onerror="this.onerror=null;this.src='https://placehold.co/600x400/1e1b4b/c4b5fd?text=App+Image';">
                <div class="p-6 flex flex-col flex-grow">
                    <div>
                        <h3 class="text-xl font-bold text-white mb-2">${app.title}</h3>
                        <p class="text-slate-400 mb-4 text-sm">${app.description}</p>
                    </div>
                    <div class="mt-auto pt-4">
                        <a href="${app.url}" target="_blank" class="inline-block bg-indigo-500 hover:bg-indigo-600 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-300">
                            Launch App
                            <svg xmlns="http://www.w3.org/2000/svg" class="inline-block ml-1 -mt-0.5" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
                        </a>
                    </div>
                </div>
            `;
            playgroundAppsGrid.appendChild(appCard);
        });
    };


    // Mobile menu
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    mobileMenuButton.addEventListener('click', () => mobileMenu.classList.toggle('hidden'));

    // Typing animation
    const roles = ["Data Scientist", "Machine Learning Engineer", "AI Engineer"];
    let roleIndex = 0, charIndex = 0;
    const roleTextElement = document.getElementById('role-text');
    function typeRole() { if(!roleTextElement) return; if (charIndex < roles[roleIndex].length) { roleTextElement.textContent += roles[roleIndex].charAt(charIndex++); setTimeout(typeRole, 100); } else { setTimeout(eraseRole, 2000); } }
    function eraseRole() { if(!roleTextElement) return; if (charIndex > 0) { roleTextElement.textContent = roles[roleIndex].substring(0, --charIndex); setTimeout(eraseRole, 50); } else { roleIndex = (roleIndex + 1) % roles.length; setTimeout(typeRole, 500); } }
    typeRole();

    // Modal logic
    const modal = document.getElementById('project-modal');
    const modalContent = modal.querySelector('.modal-content');
    window.openModal = (projectIndex) => {
        const project = projectsData[projectIndex];
        let galleryHtml = `<div id="media-viewer" class="mb-4 rounded-lg overflow-hidden bg-black"></div><div id="thumbnail-strip" class="flex gap-2 justify-center flex-wrap"></div>`;
        modalContent.innerHTML = `<button class="absolute top-4 right-6 text-slate-400 hover:text-white text-3xl z-10" onclick="closeModal()">&times;</button>${galleryHtml}<div class="px-1 mt-6"><h2 class="text-3xl font-bold text-white mb-2">${project.title}</h2><p class="text-indigo-300 mb-6">${project.description}</p><h4 class="text-lg font-semibold text-white mb-2">Key Achievements:</h4><ul class="list-none space-y-2 mb-6">${project.details.map(detail => `<li class="flex items-start text-slate-300"><span class="text-indigo-400 mr-3 mt-1">▪</span><div class="flex-1">${detail}</div></li>`).join('')}</ul><h4 class="text-lg font-semibold text-white mb-3">Technologies Used:</h4><div class="flex flex-wrap gap-2">${project.skills.map(skill => `<span class="tag rounded-md px-3 py-1 text-sm">${skill}</span>`).join('')}</div></div>`;
        populateGallery(projectIndex);
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    };
    window.closeModal = () => {
        modal.style.animation = 'fadeOut 0.3s ease-out forwards';
        setTimeout(() => {
            modal.style.display = 'none';
            modal.style.animation = 'fadeIn 0.3s ease-out';
            document.body.style.overflow = 'auto';
            modalContent.innerHTML = '';
        }, 300);
    };
    window.populateGallery = (projectIndex) => {
        const project = projectsData[projectIndex];
        const thumbnailStrip = document.getElementById('thumbnail-strip');
        if (project.media.length > 1) {
            thumbnailStrip.innerHTML = project.media.map((mediaItem, index) => `<div class="relative"><img src="${mediaItem.url}" class="thumbnail rounded-md w-24 h-16 object-cover" data-media-index="${index}" data-project-index="${projectIndex}"></div>`).join('');
            thumbnailStrip.querySelectorAll('.thumbnail').forEach(thumb => {
                thumb.addEventListener('click', (e) => switchMedia(e.target.dataset.mediaIndex, e.target.dataset.projectIndex));
            });
        } else { thumbnailStrip.innerHTML = ''; } // Hide thumbnail strip if only one image
        switchMedia(0, projectIndex);
    };
    window.switchMedia = (mediaIndex, projectIndex) => {
        const mediaItem = projectsData[projectIndex].media[mediaIndex];
        const viewer = document.getElementById('media-viewer');
        viewer.innerHTML = `<img src="${mediaItem.url}" alt="Project media" class="w-full h-auto max-h-[50vh] object-contain" onerror="this.onerror=null;this.src='https://placehold.co/1080x720/1e1b4b/c4b5fd?text=Error+Loading+Image';">`;
        document.querySelectorAll('#thumbnail-strip .thumbnail').forEach((thumb, i) => thumb.classList.toggle('active', i == mediaIndex));
    };
    window.onclick = (event) => { if (event.target == modal) closeModal(); };

    // Blog interactions
    const blogPostsContainer = document.getElementById('blog-posts-container');
    const blogFiltersContainer = document.getElementById('blog-filters');
    const recentPostsContainer = document.getElementById('recent-posts-container');

    const renderBlogPosts = (filter = 'All') => {
        blogPostsContainer.innerHTML = '';
        const filteredPosts = filter === 'All' ? blogPostsData : blogPostsData.filter(p => p.tags.includes(filter));
        filteredPosts.forEach((post, index) => {
            const article = document.createElement('article');
            article.className = 'card-bg p-8 rounded-2xl';
            article.innerHTML = `
                <img src="${post.image}" alt="${post.title}" class="rounded-lg mb-6 w-full h-64 object-cover">
                <div class="flex flex-wrap gap-2 mb-4">
                    ${post.tags.map(tag => `<span class="tag rounded-md px-2 py-1 text-xs">${tag}</span>`).join('')}
                </div>
                <h3 class="text-2xl font-bold text-white mb-2">${post.title}</h3>
                <p class="text-sm text-slate-400 mb-4">Posted on ${post.date}</p>
                <div id="blog-content-${index}" class="prose-custom text-slate-400 blog-content-truncated">
                    ${post.content}
                </div>
                <button class="read-more-btn text-indigo-400 hover:text-indigo-300 mt-4 text-sm font-semibold" data-post-index="${index}">Read More</button>
            `;
            blogPostsContainer.appendChild(article);
        });
        hljs.highlightAll(); // Highlight code snippets (even if no code, still good practice)
        
        // Add event listeners for "Read More" buttons
        document.querySelectorAll('.read-more-btn').forEach(button => {
            button.addEventListener('click', (e) => {
                const postIndex = e.target.dataset.postIndex;
                const contentDiv = document.getElementById(`blog-content-${postIndex}`);
                contentDiv.classList.toggle('blog-content-truncated');
                contentDiv.classList.toggle('blog-content-expanded');
                e.target.textContent = contentDiv.classList.contains('blog-content-truncated') ? 'Read More' : 'Read Less';
            });
        });
    };
    
    const renderBlogFilters = () => {
        const allTags = ['All', ...new Set(blogPostsData.flatMap(p => p.tags))].sort(); // Sort tags alphabetically
        blogFiltersContainer.innerHTML = allTags.map(tag => 
            `<button class="blog-tag-filter tag px-4 py-2 rounded-lg ${tag === 'All' ? 'active' : ''}" data-filter="${tag}">${tag}</button>`
        ).join('');
        
        blogFiltersContainer.querySelectorAll('.blog-tag-filter').forEach(button => {
            button.addEventListener('click', (e) => {
                const filter = e.target.dataset.filter;
                blogFiltersContainer.querySelector('.active').classList.remove('active');
                e.target.classList.add('active');
                renderBlogPosts(filter);
            });
        });
    };

    const renderRecentPosts = () => {
        recentPostsContainer.innerHTML = '';
        // Get the top 3 most recent posts (already sorted by date in data)
        const recentPosts = blogPostsData.slice(0, 3);   
        recentPosts.forEach(post => {
            const postLink = document.createElement('a');
            postLink.href = "#blog"; // Link to the blog section
            postLink.className = "block card-bg p-4 rounded-lg hover:bg-slate-800 transition-colors";
            postLink.innerHTML = `
                <p class="text-sm font-semibold text-white">${post.title}</p>
                <p class="text-xs text-slate-400">${post.date}</p>
            `;
            recentPostsContainer.appendChild(postLink);
        });
    };

    renderBlogFilters();
    renderBlogPosts();
    renderRecentPosts(); // Render recent posts on load
    renderPlaygroundApps(); // Render playground apps on load

    // Q&A and Poll interactions
    const commentForm = document.getElementById('comment-form');
    commentForm.addEventListener('submit', (e) => { e.preventDefault(); e.target.reset(); /* Replaced alert with console.log for Canvas environment */ console.log("Thank you! Your question has been submitted."); });
    const pollOptions = document.querySelectorAll('.poll-option');
    pollOptions.forEach(option => { option.addEventListener('click', () => { pollOptions.forEach(opt => { opt.disabled = true; opt.classList.add('opacity-50'); }); option.classList.add('bg-indigo-500'); document.getElementById('poll-feedback').classList.remove('hidden'); }); });

    // Background canvas animation
    const bgCanvas = document.getElementById('bg-canvas');
    const ctx = bgCanvas.getContext('2d');
    let particlesArray;
    const mouse = { x: null, y: null, radius: 0 };
    const resizeHandler = () => { bgCanvas.width = window.innerWidth; bgCanvas.height = window.innerHeight; mouse.radius = (bgCanvas.height / 120) * (bgCanvas.width / 120); initParticles(); };
    window.addEventListener('resize', resizeHandler);
    window.addEventListener('mousemove', (e) => { mouse.x = e.x; mouse.y = e.y; }); 
    class Particle { constructor(x, y, dx, dy) { this.x = x; this.y = y; this.directionX = dx; this.directionY = dy; this.size = (Math.random() * 2) + 1; } draw() { ctx.beginPath(); ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2); ctx.fillStyle = 'rgba(139, 92, 246, 0.2)'; ctx.fill(); } update() { if (this.x > bgCanvas.width || this.x < 0) this.directionX = -this.directionX; if (this.y > bgCanvas.height || this.y < 0) this.directionY = -this.directionY; let dx = mouse.x - this.x; let dy = mouse.y - this.y; if (Math.hypot(dx, dy) < mouse.radius + this.size) { if (mouse.x < this.x && this.x < bgCanvas.width - this.size * 10) this.x += 5; if (mouse.x > this.x && this.x > this.size * 10) this.x -= 5; if (mouse.y < this.y && this.y < bgCanvas.height - this.size * 10) this.y += 5; if (mouse.y > this.y && this.y > this.size * 10) this.y -= 5; } this.x += this.directionX; this.y += this.directionY; this.draw(); } } 
    function initParticles() { particlesArray = []; let num = (bgCanvas.height * bgCanvas.width) / 9000; for (let i = 0; i < num; i++) { let x = Math.random() * innerWidth; let y = Math.random() * innerHeight; let dx = (Math.random() * .4) - 0.2; let dy = (Math.random() * .4) - 0.2; particlesArray.push(new Particle(x, y, dx, dy)); } }
    function animateParticles() { requestAnimationFrame(animateParticles); ctx.clearRect(0, 0, innerWidth, innerHeight); particlesArray.forEach(p => p.update()); connectParticles(); }
    function connectParticles() { let opacityValue = 1; for (let a = 0; a < particlesArray.length; a++) { for (let b = a; b < particlesArray.length; b++) { let distance = Math.hypot(particlesArray[a].x - particlesArray[b].x, particlesArray[a].y - particlesArray[b].y); if (distance < 120) { opacityValue = 1 - (distance / 120); ctx.strokeStyle = `rgba(167, 139, 250, ${opacityValue * 0.3})`; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(particlesArray[a].x, particlesArray[a].y); ctx.lineTo(particlesArray[b].x, particlesArray[b].y); ctx.stroke(); } } } }
    resizeHandler();
    animateParticles();
});
