FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY genomics_tests_fixed.xlsx .
COPY progenics_chatbot.py .
CMD ["streamlit", "run", "progenics_chatbot.py", "--server.port=8501"]
