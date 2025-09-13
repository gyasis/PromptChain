"""
Database utilities for test data management
"""
import os
import psycopg2
import pymongo
from typing import Dict, Any, Optional, List
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DatabaseManager:
    """Manages database operations for test data setup and cleanup."""

    def __init__(self):
        self.db_type = os.getenv('DB_TYPE', 'postgresql').lower()
        self.connection = None
        
        if self.db_type == 'postgresql':
            self._init_postgresql()
        elif self.db_type == 'mongodb':
            self._init_mongodb()
        else:
            logger.warning(f"Unsupported database type: {self.db_type}")

    def _init_postgresql(self):
        """Initialize PostgreSQL connection."""
        try:
            self.connection = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', 5432),
                database=os.getenv('DB_NAME'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD')
            )
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")

    def _init_mongodb(self):
        """Initialize MongoDB connection."""
        try:
            client = pymongo.MongoClient(
                host=os.getenv('DB_HOST', 'localhost'),
                port=int(os.getenv('DB_PORT', 27017))
            )
            self.connection = client[os.getenv('DB_NAME')]
            logger.info("Connected to MongoDB database")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")

    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute SQL query and return results."""
        if self.db_type != 'postgresql' or not self.connection:
            return []
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params or ())
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
                return []
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user by ID from database."""
        if self.db_type == 'postgresql':
            result = self.execute_query("SELECT * FROM users WHERE id = %s", (user_id,))
            return result[0] if result else None
        elif self.db_type == 'mongodb':
            return self.connection.users.find_one({"_id": user_id})
        return None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Retrieve user by email from database."""
        if self.db_type == 'postgresql':
            result = self.execute_query("SELECT * FROM users WHERE email = %s", (email,))
            return result[0] if result else None
        elif self.db_type == 'mongodb':
            return self.connection.users.find_one({"email": email})
        return None

    def delete_user_by_email(self, email: str) -> bool:
        """Delete user by email from database."""
        try:
            if self.db_type == 'postgresql':
                with self.connection.cursor() as cursor:
                    cursor.execute("DELETE FROM users WHERE email = %s", (email,))
                    self.connection.commit()
                    return cursor.rowcount > 0
            elif self.db_type == 'mongodb':
                result = self.connection.users.delete_one({"email": email})
                return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete user {email}: {e}")
            return False

    def cleanup_user_data(self, user_id: str):
        """Clean up all data associated with a user."""
        try:
            if self.db_type == 'postgresql':
                with self.connection.cursor() as cursor:
                    # Delete in reverse dependency order
                    tables = ['user_sessions', 'user_profiles', 'orders', 'users']
                    for table in tables:
                        cursor.execute(f"DELETE FROM {table} WHERE user_id = %s", (user_id,))
                    self.connection.commit()
            elif self.db_type == 'mongodb':
                collections = ['user_sessions', 'user_profiles', 'orders', 'users']
                for collection in collections:
                    self.connection[collection].delete_many({"user_id": user_id})
            
            logger.info(f"Cleaned up data for user: {user_id}")
        except Exception as e:
            logger.error(f"Cleanup failed for user {user_id}: {e}")

    def verify_data_integrity(self, user_id: str) -> Dict[str, bool]:
        """Verify data integrity across related tables."""
        checks = {
            'user_exists': False,
            'profile_exists': False,
            'email_verified': False
        }
        
        try:
            user = self.get_user_by_id(user_id)
            if user:
                checks['user_exists'] = True
                checks['email_verified'] = user.get('email_verified', False)
                
                # Check if profile exists
                if self.db_type == 'postgresql':
                    profile = self.execute_query(
                        "SELECT * FROM user_profiles WHERE user_id = %s", 
                        (user_id,)
                    )
                    checks['profile_exists'] = len(profile) > 0
                elif self.db_type == 'mongodb':
                    profile = self.connection.user_profiles.find_one({"user_id": user_id})
                    checks['profile_exists'] = profile is not None
        except Exception as e:
            logger.error(f"Data integrity check failed: {e}")
        
        return checks

    def close(self):
        """Close database connection."""
        if self.connection:
            if self.db_type == 'postgresql':
                self.connection.close()
            elif self.db_type == 'mongodb':
                self.connection.client.close()
            logger.info("Database connection closed")



