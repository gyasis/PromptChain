from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json
import os

# Create the base class for declarative models
Base = declarative_base()

# Define the Question model to match existing table
class Question(Base):
    __tablename__ = 'questions'
    
    id = Column(Integer, primary_key=True)
    question_text = Column(Text, nullable=False)
    options = Column(Text, nullable=False)
    correct_option = Column(String(255), nullable=False)
    subject = Column(String(100), nullable=False)
    reasoning = Column(Text, nullable=True)

# Database configuration
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    "postgresql+psycopg2://postgresUser:postgresPW@localhost:5455/postgresDB"
)

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def get_random_question():
    """Fetch a random question from the database."""
    try:
        # Get database session
        session = SessionLocal()
        
        try:
            # Get random question
            random_question = session.query(Question).order_by(func.random()).first()
            
            if not random_question:
                return "No questions found in the database."
            
            # Parse options from JSON string
            try:
                options = json.loads(random_question.options)
            except json.JSONDecodeError:
                options = random_question.options.strip('"\'').split(',')
            
            # Format the output
            output = f"""
Question: {random_question.question_text}

Options:
{chr(10).join(f"{i+1}. {opt.strip()}" for i, opt in enumerate(options))}

Correct Answer: {random_question.correct_option}

Subject: {random_question.subject}
"""
            if random_question.reasoning:
                output += f"\nExplanation:\n{random_question.reasoning}"
            
            return output
            
        finally:
            session.close()
            
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Random Question Fetcher")
    print("="*50 + "\n")
    
    result = get_random_question()
    print(result)